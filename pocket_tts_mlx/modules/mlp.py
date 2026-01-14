# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Flow matching MLP with adaptive layer normalization."""

import math

import mlx.core as mx
import mlx.nn as nn


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply adaptive modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Note: This implementation matches the PyTorch version in pocket_tts which
    uses variance (E[(x-mean)^2]) instead of the standard mean(x^2).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # Match PyTorch version: uses var instead of mean(x^2)
        var = mx.var(x, axis=-1, keepdims=True) + self.eps
        y = x * (self.alpha * mx.rsqrt(var))
        return y


class LayerNorm(nn.Module):
    """Layer normalization (reimplemented for consistency)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        # MLP: Linear -> SiLU -> Linear -> RMSNorm
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

        # Precompute frequency basis
        half = frequency_embedding_size // 2
        self.freqs = mx.exp(-math.log(max_period) * mx.arange(half) / half)

    def __call__(self, t: mx.array) -> mx.array:
        """Embed timestep.

        Args:
            t: Timestep tensor of any shape, will embed last dimension

        Returns:
            Embedded timestep [*, hidden_size]
        """
        # Sinusoidal embedding
        args = t * self.freqs
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        # MLP
        x = self.linear1(embedding)
        x = nn.silu(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x


class ResBlock(nn.Module):
    """Residual block with adaptive layer normalization (adaLN).

    Uses modulation from conditioning to adaptively scale and shift
    the layer normalization output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Layer norm before MLP
        self.in_ln = LayerNorm(channels, eps=1e-6)

        # MLP: Linear -> SiLU -> Linear
        self.mlp_linear1 = nn.Linear(channels, channels, bias=True)
        self.mlp_linear2 = nn.Linear(channels, channels, bias=True)

        # adaLN modulation: SiLU -> Linear (produces shift, scale, gate)
        self.adaLN_linear = nn.Linear(channels, 3 * channels, bias=True)

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor
            y: Conditioning tensor (time + context)

        Returns:
            Output tensor with residual connection
        """
        # Get modulation parameters from conditioning
        modulation = nn.silu(y)
        modulation = self.adaLN_linear(modulation)

        # Split into shift, scale, gate
        shift_mlp, scale_mlp, gate_mlp = mx.split(modulation, 3, axis=-1)

        # Apply modulated layer norm
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)

        # MLP
        h = self.mlp_linear1(h)
        h = nn.silu(h)
        h = self.mlp_linear2(h)

        # Gated residual connection
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """Final layer with adaptive layer normalization."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

        # adaLN modulation for final layer (produces shift and scale only)
        self.adaLN_linear = nn.Linear(model_channels, 2 * model_channels, bias=True)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor
            c: Conditioning tensor

        Returns:
            Output tensor
        """
        # Get modulation parameters
        modulation = nn.silu(c)
        modulation = self.adaLN_linear(modulation)
        shift, scale = mx.split(modulation, 2, axis=-1)

        # Apply modulated layer norm and final projection
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """MLP for flow matching with adaptive layer normalization.

    This is the core network for predicting flow vectors in the
    Lagrangian Self Distillation (LSD) framework.

    Args:
        in_channels: Input dimension (latent dim)
        model_channels: Hidden dimension
        out_channels: Output dimension (latent dim)
        cond_channels: Conditioning dimension (from transformer)
        num_res_blocks: Number of residual blocks
        num_time_conds: Number of time conditions (2 for LSD: start and target time)
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_time_conds: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        assert num_time_conds == 2, "LSD requires exactly 2 time conditions (s and t)"

        # Time embedding for each time condition
        self.time_embed = [
            TimestepEmbedder(model_channels) for _ in range(num_time_conds)
        ]

        # Condition embedding
        self.cond_embed = nn.Linear(cond_channels, model_channels)

        # Input projection
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Residual blocks
        self.res_blocks = [ResBlock(model_channels) for _ in range(num_res_blocks)]

        # Final layer
        self.final_layer = FinalLayer(model_channels, out_channels)

    def __call__(
        self,
        c: mx.array,
        s: mx.array,
        t: mx.array,
        x: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            c: Conditioning from transformer [B, D] or [B, T, D]
            s: Start time [B, 1] (values in [0, 1])
            t: Target time [B, 1] (values in [0, 1])
            x: Input latent [B, D] or [B, T, D]

        Returns:
            Flow prediction [B, D] or [B, T, D]
        """
        # Project input to model dimension
        x = self.input_proj(x)

        # Combine time embeddings (average for LSD)
        t_embed_s = self.time_embed[0](s)
        t_embed_t = self.time_embed[1](t)
        t_combined = (t_embed_s + t_embed_t) / self.num_time_conds

        # Embed conditioning
        c = self.cond_embed(c)

        # Combine time and conditioning
        y = t_combined + c

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x, y)

        # Final layer
        return self.final_layer(x, y)
