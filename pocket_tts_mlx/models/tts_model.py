# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main TTS model combining FlowLM and Mimi for MLX."""

import copy
import logging
import queue
import statistics
import threading
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.conditioners.text import LUTConditioner, TokenizedText
from pocket_tts_mlx.models.flow_lm import FlowLMModel
from pocket_tts_mlx.models.mimi import MimiModel
from pocket_tts_mlx.modules.dummy_quantizer import DummyQuantizer
from pocket_tts_mlx.modules.mlp import SimpleMLPAdaLN
from pocket_tts_mlx.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts_mlx.modules.transformer import ProjectedTransformer, StreamingTransformer
from pocket_tts_mlx.utils.audio import audio_read, convert_audio
from pocket_tts_mlx.utils.loaders import (
    PREDEFINED_VOICES,
    download_if_necessary,
    load_predefined_voice,
    load_safetensors_weights,
    convert_conv_weights,
    remap_weight_key,
)

logger = logging.getLogger(__name__)

# Default generation parameters
DEFAULT_VARIANT = "b6369a24"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_LSD_DECODE_STEPS = 1
DEFAULT_NOISE_CLAMP = 3.0
DEFAULT_EOS_THRESHOLD = -4.0


class TTSModel(nn.Module):
    """Text-to-Speech model combining FlowLM and Mimi.

    This is the main user-facing class for TTS generation.
    """

    def __init__(
        self,
        flow_lm: FlowLMModel,
        mimi: MimiModel,
        temp: float = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        self.mimi = mimi
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self._sample_rate = sample_rate
        self.has_voice_cloning = True

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def reset_state(self):
        """Reset all streaming state."""
        self.flow_lm.reset_state()
        self.mimi.reset_state()

    def _encode_audio(self, audio: mx.array) -> mx.array:
        """Encode audio for voice conditioning.

        Args:
            audio: Audio tensor [B, C, T]

        Returns:
            Conditioning tensor [B, T', dim]
        """
        encoded = self.mimi.encode_to_latent(audio)
        # NCL -> NLC
        latents = encoded.swapaxes(-1, -2)
        # Project through speaker projection
        conditioning = latents @ self.flow_lm.speaker_proj_weight.T
        return conditioning

    def _run_flow_lm(
        self,
        text_tokens: mx.array | None = None,
        backbone_input_latents: mx.array | None = None,
        audio_conditioning: mx.array | None = None,
        temperature: float | None = None,
        lsd_decode_steps: int | None = None,
        noise_clamp: float | None = None,
        eos_threshold: float | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run FlowLM for one step.

        Args:
            text_tokens: Optional text tokens [B, T]
            backbone_input_latents: Optional input latents [B, T, ldim]
            audio_conditioning: Optional audio conditioning [B, T, dim]
            temperature: Sampling temperature (overrides instance default)
            lsd_decode_steps: LSD decoding steps (overrides instance default)
            noise_clamp: Noise clamp value (overrides instance default)
            eos_threshold: EOS threshold (overrides instance default)

        Returns:
            Tuple of (output_latent, is_eos)
        """
        B = 1  # Batch size

        if text_tokens is None:
            text_embeddings = mx.zeros((B, 0, self.flow_lm.dim))
        else:
            text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))

        if backbone_input_latents is None:
            backbone_input_latents = mx.zeros((B, 0, self.flow_lm.ldim))

        if audio_conditioning is not None:
            text_embeddings = mx.concatenate([text_embeddings, audio_conditioning], axis=1)

        output_embeddings, is_eos = self.flow_lm.sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            lsd_decode_steps=lsd_decode_steps if lsd_decode_steps is not None else self.lsd_decode_steps,
            temp=temperature if temperature is not None else self.temp,
            noise_clamp=noise_clamp if noise_clamp is not None else self.noise_clamp,
            eos_threshold=eos_threshold if eos_threshold is not None else self.eos_threshold,
        )
        return output_embeddings, is_eos

    def generate_audio(
        self,
        text: str,
        voice: str | Path | mx.array | None = None,
        max_duration_sec: float = 30.0,
        frames_after_eos: int = 3,
        temperature: float | None = None,
        lsd_decode_steps: int | None = None,
        noise_clamp: float | None = None,
        eos_threshold: float | None = None,
    ) -> mx.array:
        """Generate audio from text.

        Args:
            text: Input text to synthesize
            voice: Voice conditioning - can be:
                - str: Predefined voice name or path to audio file
                - Path: Path to audio file
                - mx.array: Pre-encoded voice embedding
                - None: No voice conditioning
            max_duration_sec: Maximum generation duration in seconds
            frames_after_eos: Frames to generate after EOS detection
            temperature: Sampling temperature (overrides instance default)
            lsd_decode_steps: LSD decoding steps (overrides instance default)
            noise_clamp: Noise clamp value (overrides instance default)
            eos_threshold: EOS threshold (overrides instance default)

        Returns:
            Audio tensor [T] at sample_rate
        """
        chunks = list(
            self.generate_audio_stream(
                text=text,
                voice=voice,
                max_duration_sec=max_duration_sec,
                frames_after_eos=frames_after_eos,
                temperature=temperature,
                lsd_decode_steps=lsd_decode_steps,
                noise_clamp=noise_clamp,
                eos_threshold=eos_threshold,
            )
        )
        return mx.concatenate(chunks, axis=0)

    def generate_audio_stream(
        self,
        text: str,
        voice: str | Path | mx.array | None = None,
        max_duration_sec: float = 30.0,
        frames_after_eos: int = 3,
        temperature: float | None = None,
        lsd_decode_steps: int | None = None,
        noise_clamp: float | None = None,
        eos_threshold: float | None = None,
    ) -> Iterator[mx.array]:
        """Generate audio from text with streaming output.

        Args:
            text: Input text to synthesize
            voice: Voice conditioning (see generate_audio)
            max_duration_sec: Maximum generation duration
            frames_after_eos: Frames after EOS
            temperature: Sampling temperature (overrides instance default)
            lsd_decode_steps: LSD decoding steps (overrides instance default)
            noise_clamp: Noise clamp value (overrides instance default)
            eos_threshold: EOS threshold (overrides instance default)

        Yields:
            Audio chunks [T_chunk]
        """
        # Reset state
        self.reset_state()
        self.flow_lm.make_cache(batch_size=1)
        self.mimi.make_caches(batch_size=1)

        # Prepare text
        text = prepare_text_prompt(text)

        # Handle voice conditioning
        if voice is not None:
            if isinstance(voice, str) and voice in PREDEFINED_VOICES:
                audio_conditioning = load_predefined_voice(voice)
                audio_conditioning = audio_conditioning.reshape(1, -1, self.flow_lm.dim)
            elif isinstance(voice, (str, Path)):
                # Load and encode audio file
                audio, sr = audio_read(str(voice))
                audio = convert_audio(audio, sr, self.sample_rate, 1)
                audio = audio.reshape(1, 1, -1)  # [B, C, T]
                audio_conditioning = self._encode_audio(audio)
            else:
                # Assume pre-encoded
                audio_conditioning = voice

            # Run FlowLM with audio conditioning
            self._run_flow_lm(audio_conditioning=audio_conditioning)
            mx.eval(self.flow_lm._cache)

        # Tokenize and condition on text
        prepared = self.flow_lm.conditioner.prepare(text)
        self._run_flow_lm(text_tokens=prepared.tokens)
        mx.eval(self.flow_lm._cache)

        # Autoregressive generation
        max_gen_len = int(max_duration_sec * 12.5)  # 12.5 Hz frame rate
        backbone_input = mx.full((1, 1, self.flow_lm.ldim), float("nan"))

        eos_step = None
        t_start = time.monotonic()
        total_samples = 0

        for step in range(max_gen_len):
            # Generate next latent
            next_latent, is_eos = self._run_flow_lm(
                backbone_input_latents=backbone_input,
                temperature=temperature,
                lsd_decode_steps=lsd_decode_steps,
                noise_clamp=noise_clamp,
                eos_threshold=eos_threshold,
            )

            # Decode to audio (build full graph before eval)
            mimi_input = next_latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
            mimi_input = mimi_input.swapaxes(-1, -2)  # NLC -> NCL
            quantized = self.mimi.quantizer(mimi_input)
            audio_frame = self.mimi.decode_from_latent_step(quantized)

            # Single eval for entire step (reduces graph compilation overhead)
            mx.eval(next_latent, is_eos, audio_frame)

            # Check EOS
            if bool(is_eos[0, 0]) and eos_step is None:
                eos_step = step
                logger.debug(f"EOS detected at step {step}")

            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            # Yield audio chunk (remove batch and channel dims)
            chunk = audio_frame[0, 0]
            total_samples += chunk.shape[0]
            yield chunk

            # Update input for next step
            backbone_input = next_latent

        # Log timing
        duration_ms = int(total_samples * 1000 / self.sample_rate)
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        if elapsed_ms > 0:
            rtf = duration_ms / elapsed_ms
            logger.info(f"Generated {duration_ms}ms audio in {elapsed_ms}ms ({rtf:.2f}x realtime)")

    @staticmethod
    def load_model(
        variant: str = DEFAULT_VARIANT,
        temp: float = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    ) -> "TTSModel":
        """Load a pre-trained TTS model.

        Args:
            variant: Model variant (config file name)
            temp: Sampling temperature
            lsd_decode_steps: LSD decoding steps
            noise_clamp: Noise clamping value
            eos_threshold: EOS detection threshold

        Returns:
            Loaded TTSModel
        """
        # Load config from our own config module
        from pocket_tts_mlx.utils.config import load_config

        # Config is in the pocket_tts_mlx/config directory
        config_path = Path(__file__).parents[1] / f"config/{variant}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                f"Available configs: {list((Path(__file__).parents[1] / 'config').glob('*.yaml'))}"
            )
        config = load_config(config_path)

        # Build model components
        flow_lm_config = config.flow_lm
        mimi_config = config.mimi.model_dump()

        # Build FlowLM
        d_model = flow_lm_config.transformer.d_model
        dim_feedforward = int(d_model * flow_lm_config.transformer.hidden_scale)
        ldim = mimi_config["quantizer"]["dimension"]

        flow_mlp = SimpleMLPAdaLN(
            in_channels=ldim,
            model_channels=flow_lm_config.flow.dim,
            out_channels=ldim,
            cond_channels=d_model,
            num_res_blocks=flow_lm_config.flow.depth,
            num_time_conds=2,
        )

        conditioner = LUTConditioner(
            n_bins=flow_lm_config.lookup_table.n_bins,
            tokenizer_path=str(flow_lm_config.lookup_table.tokenizer_path),
            dim=flow_lm_config.lookup_table.dim,
            output_dim=d_model,
        )

        transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=flow_lm_config.transformer.num_heads,
            num_layers=flow_lm_config.transformer.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(flow_lm_config.transformer.max_period),
            kind="flow_lm",
        )

        flow_lm = FlowLMModel(
            conditioner=conditioner,
            flow_net=flow_mlp,
            transformer=transformer,
            dim=d_model,
            ldim=ldim,
        )

        # Initialize speaker projection
        flow_lm.speaker_proj_weight = mx.zeros((1024, 512))

        # Build Mimi
        encoder = SEANetEncoder(**mimi_config["seanet"])
        decoder = SEANetDecoder(**mimi_config["seanet"])

        encoder_transformer = ProjectedTransformer(**mimi_config["transformer"])
        decoder_transformer = ProjectedTransformer(**mimi_config["transformer"])
        quantizer = DummyQuantizer(**mimi_config["quantizer"])

        mimi = MimiModel(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            channels=mimi_config["channels"],
            sample_rate=mimi_config["sample_rate"],
            frame_rate=mimi_config["frame_rate"],
            encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        )

        # Create TTS model
        tts_model = TTSModel(
            flow_lm=flow_lm,
            mimi=mimi,
            temp=temp,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            sample_rate=mimi_config["sample_rate"],
        )

        # Load weights
        weights_path = config.weights_path
        if weights_path is not None:
            logger.info(f"Loading weights from {weights_path}")
            try:
                weights_file = download_if_necessary(weights_path)
            except Exception:
                tts_model.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

            tts_model.load_pytorch_weights(str(weights_file))

        return tts_model

    def load_pytorch_weights(self, file_path: str):
        """Load weights from PyTorch safetensors file.

        Args:
            file_path: Path to safetensors file
        """
        raw_weights = load_safetensors_weights(file_path)

        weights = []
        for key, value in raw_weights.items():
            # Remap key
            new_key = remap_weight_key(key)

            # Skip keys that should be ignored
            if new_key is None:
                continue

            # Convert conv weights
            value = convert_conv_weights(new_key, value)

            weights.append((new_key, value))

        # Load into model
        self.load_weights(weights)

        # Update ConvTranspose1d modules that need expanded weights
        # Use MLX's filter_and_map to properly traverse the module tree
        from pocket_tts_mlx.modules.conv import ConvTranspose1d

        def _update_convtr_fn(module, name, _):
            if isinstance(module, ConvTranspose1d) and name == "weight":
                module.update_in_place()
            return True

        self.filter_and_map(_update_convtr_fn)


def prepare_text_prompt(text: str) -> str:
    """Prepare text for TTS generation.

    Args:
        text: Input text

    Returns:
        Formatted text
    """
    text = text.strip()
    if not text:
        raise ValueError("Text prompt cannot be empty")

    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")

    # Capitalize first letter
    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    # Add punctuation if missing
    if text[-1].isalnum():
        text = text + "."

    # Pad short texts for better generation
    if len(text.split()) < 5:
        text = " " * 8 + text

    return text
