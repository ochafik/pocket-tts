# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Mimi neural audio codec for MLX."""

import logging
import math

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.conv import ConvDownsample1d, ConvTrUpsample1d
from pocket_tts_mlx.modules.dummy_quantizer import DummyQuantizer
from pocket_tts_mlx.modules.kv_cache import KVCache, RotatingKVCache
from pocket_tts_mlx.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts_mlx.modules.transformer import ProjectedTransformer

logger = logging.getLogger(__name__)


def get_extra_padding_for_conv1d(
    xs: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Calculate extra padding for conv1d."""
    length = xs.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


def pad_for_conv1d(
    x: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> mx.array:
    """Pad tensor for convolution to ensure last window is full."""
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    if extra_padding > 0:
        x = mx.pad(x, pad_width=((0, 0), (0, 0), (0, extra_padding)))
    return x


class MimiModel(nn.Module):
    """Mimi neural audio codec.

    Combines SEANet encoder/decoder with transformers and optional resampling.
    """

    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: DummyQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        encoder_transformer: ProjectedTransformer,
        decoder_transformer: ProjectedTransformer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoder_frame_rate = encoder_frame_rate

        self.dimension = encoder.dimension

        # Create resampling layers if needed
        self.downsample = None
        self.upsample = None
        if encoder_frame_rate != frame_rate:
            assert encoder_frame_rate > frame_rate, "Cannot upsample with conv."
            downsample_stride = int(encoder_frame_rate / frame_rate)
            self.downsample = ConvDownsample1d(downsample_stride, self.dimension)
            self.upsample = ConvTrUpsample1d(downsample_stride, self.dimension)

        # Create transformer caches
        self._encoder_cache = None
        self._decoder_cache = None

    @property
    def frame_size(self) -> int:
        """Number of audio samples per frame."""
        return int(self.sample_rate / self.frame_rate)

    def reset_state(self):
        """Reset all streaming state."""
        self.encoder.reset_state()
        self.decoder.reset_state()
        if self.downsample is not None:
            self.downsample.reset_state()
        if self.upsample is not None:
            self.upsample.reset_state()
        # Reset transformer caches
        if self._encoder_cache is not None:
            self.encoder_transformer.reset_cache(self._encoder_cache)
        if self._decoder_cache is not None:
            self.decoder_transformer.reset_cache(self._decoder_cache)

    def make_caches(self, batch_size: int = 1):
        """Create transformer caches."""
        self._encoder_cache = self.encoder_transformer.make_cache(batch_size)
        self._decoder_cache = self.decoder_transformer.make_cache(batch_size)

    def _to_framerate(self, x: mx.array) -> mx.array:
        """Convert from encoder frame rate to overall frame rate."""
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.downsample(x)

    def _to_encoder_framerate(self, x: mx.array) -> mx.array:
        """Convert from overall frame rate to encoder frame rate."""
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.upsample(x)

    def _to_encoder_framerate_step(self, x: mx.array) -> mx.array:
        """Streaming convert from overall frame rate to encoder frame rate."""
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.upsample.step(x)

    def encode_to_latent(self, x: mx.array) -> mx.array:
        """Encode audio to latent space.

        Args:
            x: Audio tensor [B, C, T]

        Returns:
            Latent tensor [B, D, T']
        """
        assert x.ndim == 3, f"Expected [B, C, T] but got shape {x.shape}"

        frame_size = self.frame_size

        # Pad to multiple of frame size
        x = pad_for_conv1d(x, frame_size, frame_size)

        # Reset state for full encoding
        self.encoder.reset_state()
        if self._encoder_cache is None:
            self.make_caches(x.shape[0])
        else:
            self.encoder_transformer.reset_cache(self._encoder_cache)

        # Encode through SEANet
        emb = self.encoder(x)

        # Through transformer
        (emb,) = self.encoder_transformer(emb, self._encoder_cache)

        # Downsample to target frame rate
        emb = self._to_framerate(emb)

        return emb

    def decode_from_latent(self, latent: mx.array) -> mx.array:
        """Decode latent to audio (full sequence).

        Args:
            latent: Latent tensor [B, D, T]

        Returns:
            Audio tensor [B, C, T']
        """
        # Reset state for full decoding
        self.decoder.reset_state()
        if self._decoder_cache is None:
            self.make_caches(latent.shape[0])
        else:
            self.decoder_transformer.reset_cache(self._decoder_cache)

        # Upsample to encoder frame rate
        emb = self._to_encoder_framerate(latent)

        # Through transformer
        (emb,) = self.decoder_transformer(emb, self._decoder_cache)

        # Decode through SEANet
        out = self.decoder(emb)

        return out

    def decode_from_latent_step(self, latent: mx.array) -> mx.array:
        """Decode one frame of latent to audio (streaming).

        Args:
            latent: Latent tensor [B, D, 1]

        Returns:
            Audio tensor [B, C, frame_size]
        """
        if self._decoder_cache is None:
            self.make_caches(latent.shape[0])

        # Upsample to encoder frame rate
        emb = self._to_encoder_framerate_step(latent)

        # Through transformer
        (emb,) = self.decoder_transformer(emb, self._decoder_cache)

        # Decode through SEANet (streaming)
        out = self.decoder.step(emb)

        return out
