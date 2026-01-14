# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Flow Language Model for TTS generation."""

import logging
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.conditioners.text import LUTConditioner, TokenizedText
from pocket_tts_mlx.modules.kv_cache import KVCache
from pocket_tts_mlx.modules.mlp import SimpleMLPAdaLN
from pocket_tts_mlx.modules.transformer import StreamingTransformer

logger = logging.getLogger(__name__)


def lsd_decode(
    v_t,
    x_0: mx.array,
    num_steps: int = 1,
) -> mx.array:
    """Rebuild data sample from starting point using Lagrangian Self Distillation.

    Reference: https://arxiv.org/pdf/2505.18825

    Args:
        v_t: Function taking (s, t, x) and returning the flow direction
        x_0: Starting point from the known distribution [B, D]
        num_steps: Number of integration steps

    Returns:
        x_1_hat: Reconstructed data sample [B, D]
    """
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps
        # Create time arrays with shape matching x_0
        s_arr = mx.full(x_0.shape[:-1] + (1,), s)
        t_arr = mx.full(x_0.shape[:-1] + (1,), t)
        flow_dir = v_t(s_arr, t_arr, current)
        current = current + flow_dir / num_steps
    return current


class FlowLMModel(nn.Module):
    """Transformer-based flow language model for TTS.

    Uses Lagrangian Self Distillation (LSD) for generating audio latents
    from text conditioning.
    """

    def __init__(
        self,
        conditioner: LUTConditioner,
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        dim: int = 128,
        ldim: int = 64,
    ):
        super().__init__()
        self.conditioner = conditioner
        self.ldim = ldim
        self.dim = dim

        self.flow_net = flow_net

        # Buffers for latent statistics (will be loaded from weights)
        self.emb_std = mx.ones((ldim,))
        self.emb_mean = mx.zeros((ldim,))

        # BOS embedding (learnable parameter)
        self.bos_emb = mx.zeros((ldim,))

        # Input projection from latent to transformer dim
        self.input_linear = nn.Linear(ldim, dim, bias=False)

        # Transformer backbone
        self.transformer = transformer

        # Output layers
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.out_eos = nn.Linear(dim, 1)

        # Speaker projection for voice cloning (will be loaded from weights)
        self.speaker_proj_weight = None  # [1024, 512]

        # Cache for streaming
        self._cache = None

    def reset_state(self):
        """Reset streaming state."""
        if self._cache is not None:
            self.transformer.reset_cache(self._cache)

    def make_cache(self, batch_size: int = 1):
        """Create transformer cache."""
        self._cache = self.transformer.make_cache(batch_size)
        return self._cache

    def backbone(
        self,
        input_: mx.array,
        text_embeddings: mx.array,
        sequence: mx.array,
    ) -> mx.array:
        """Run transformer backbone.

        Args:
            input_: Projected input [B, T, dim]
            text_embeddings: Text embeddings [B, T_text, dim]
            sequence: Original sequence for length calculation

        Returns:
            Transformer output [B, T, dim]
        """
        # Concatenate text and audio embeddings
        input_ = mx.concatenate([text_embeddings, input_], axis=1)

        # Run transformer
        if self._cache is None:
            self._cache = self.transformer.make_cache(input_.shape[0])

        transformer_out = self.transformer(input_, self._cache)

        # Apply output norm
        transformer_out = self.out_norm(transformer_out)

        # Remove the text prefix from outputs
        transformer_out = transformer_out[:, -sequence.shape[1] :]

        return transformer_out

    def __call__(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[mx.array, mx.array]:
        """Generate next latent from sequence.

        Args:
            sequence: Input latents [B, T, ldim] (NaN for BOS)
            text_embeddings: Text embeddings [B, T_text, dim]
            lsd_decode_steps: Number of LSD decoding steps
            temp: Sampling temperature
            noise_clamp: Optional noise clamping value
            eos_threshold: EOS detection threshold

        Returns:
            Tuple of (next_latent [B, ldim], is_eos [B])
        """
        # Replace NaN with BOS embedding
        is_bos = mx.isnan(sequence)
        sequence = mx.where(is_bos, self.bos_emb, sequence)

        # Project to transformer dimension
        input_ = self.input_linear(sequence)

        # Run backbone
        transformer_out = self.backbone(input_, text_embeddings, sequence)

        # Get last timestep output
        transformer_out = transformer_out[:, -1]

        # EOS prediction
        out_eos = self.out_eos(transformer_out) > eos_threshold

        # Generate noise for flow matching
        noise_shape = transformer_out.shape[:-1] + (self.ldim,)
        std = temp**0.5

        if noise_clamp is None:
            noise = mx.random.normal(noise_shape) * std
        else:
            # Truncated normal
            noise = mx.random.normal(noise_shape) * std
            noise = mx.clip(noise, -noise_clamp, noise_clamp)

        # Create conditioned flow function
        conditioned_flow = partial(self.flow_net, transformer_out)

        # Decode using LSD
        decoded = lsd_decode(conditioned_flow, noise, lsd_decode_steps)

        return decoded, out_eos

    def sample_next_latent(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[mx.array, mx.array]:
        """Sample next latent and return with batch dimension.

        Args:
            sequence: Input sequence [B, T, ldim]
            text_embeddings: Text embeddings [B, T_text, dim]
            lsd_decode_steps: Number of LSD steps
            temp: Temperature
            noise_clamp: Noise clamp value
            eos_threshold: EOS threshold

        Returns:
            Tuple of (next_latent [B, 1, ldim], is_eos [B, 1])
        """
        decoded, is_eos = self(
            sequence=sequence,
            text_embeddings=text_embeddings,
            lsd_decode_steps=lsd_decode_steps,
            temp=temp,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        )
        # Add time dimension
        return decoded[:, None, :], is_eos
