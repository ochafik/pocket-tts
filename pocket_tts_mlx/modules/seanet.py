# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""SEANet encoder/decoder for audio compression."""

import math

import mlx.core as mx
import mlx.nn as nn

from .conv import StreamingConv1d, StreamingConvTranspose1d


class ELU(nn.Module):
    """ELU activation as a module for consistent layer handling."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._alpha = alpha

    def __call__(self, x: mx.array) -> mx.array:
        return nn.elu(x, alpha=self._alpha)


class SEANetResnetBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] = [3, 1],
        dilations: list[int] = [1, 1],
        pad_mode: str = "constant",
        compress: int = 2,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), (
            "Number of kernel sizes should match number of dilations"
        )
        hidden = dim // compress
        self.block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            self.block.append(
                StreamingConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    pad_mode=pad_mode,
                )
            )
        self._dim = dim

    def reset_state(self):
        """Reset streaming state for all conv layers."""
        for layer in self.block:
            layer.reset_state()

    def __call__(self, x: mx.array) -> mx.array:
        """Full sequence forward pass."""
        v = x
        for layer in self.block:
            v = nn.elu(v, alpha=1.0)
            v = layer(v)
        assert x.shape == v.shape, (x.shape, v.shape)
        return x + v

    def step(self, x: mx.array) -> mx.array:
        """Streaming forward pass."""
        v = x
        for layer in self.block:
            v = nn.elu(v, alpha=1.0)
            v = layer.step(v)
        # Handle shape mismatch in streaming mode
        if x.shape[-1] != v.shape[-1]:
            # Align shapes for residual connection
            min_len = min(x.shape[-1], v.shape[-1])
            if min_len > 0:
                return x[..., :min_len] + v[..., :min_len]
            return v
        return x + v


class SEANetEncoder(nn.Module):
    """SEANet encoder with downsampling."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "constant",
        compress: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(math.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2

        mult = 1
        # First conv (no ELU before it)
        self.model = [
            StreamingConv1d(channels, mult * n_filters, kernel_size, pad_mode=pad_mode)
        ]

        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers (no ELU before them - they have internal ELU)
            for j in range(n_residual_layers):
                self.model.append(
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                )

            # Add ELU then downsampling layers
            self.model.append(ELU(alpha=1.0))
            self.model.append(
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    pad_mode=pad_mode,
                )
            )
            mult *= 2

        # Final ELU + conv
        self.model.append(ELU(alpha=1.0))
        self.model.append(
            StreamingConv1d(mult * n_filters, dimension, last_kernel_size, pad_mode=pad_mode)
        )

    def reset_state(self):
        """Reset streaming state for all layers."""
        for layer in self.model:
            if hasattr(layer, "reset_state"):
                layer.reset_state()

    def __call__(self, x: mx.array) -> mx.array:
        """Full sequence forward pass."""
        for layer in self.model:
            x = layer(x)
        return x

    def step(self, x: mx.array) -> mx.array:
        """Streaming forward pass."""
        for layer in self.model:
            if isinstance(layer, ELU):
                x = layer(x)
            elif hasattr(layer, "step"):
                x = layer.step(x)
            else:
                x = layer(x)
        return x


class SEANetDecoder(nn.Module):
    """SEANet decoder with upsampling."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "constant",
        compress: int = 2,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(math.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2

        mult = int(2 ** len(self.ratios))
        # First conv (no ELU before it)
        self.model = [
            StreamingConv1d(dimension, mult * n_filters, kernel_size, pad_mode=pad_mode)
        ]

        # Upsample to raw audio scale
        for ratio in self.ratios:
            # Add ELU then upsampling layers
            self.model.append(ELU(alpha=1.0))
            self.model.append(
                StreamingConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            )

            # Add residual layers (no ELU before them - they have internal ELU)
            for j in range(n_residual_layers):
                self.model.append(
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                )
            mult //= 2

        # Final ELU + conv
        self.model.append(ELU(alpha=1.0))
        self.model.append(
            StreamingConv1d(n_filters, channels, last_kernel_size, pad_mode=pad_mode)
        )

    def reset_state(self):
        """Reset streaming state for all layers."""
        for layer in self.model:
            if hasattr(layer, "reset_state"):
                layer.reset_state()

    def __call__(self, z: mx.array) -> mx.array:
        """Full sequence forward pass."""
        for layer in self.model:
            z = layer(z)
        return z

    def step(self, z: mx.array) -> mx.array:
        """Streaming forward pass."""
        for layer in self.model:
            if isinstance(layer, ELU):
                z = layer(z)
            elif hasattr(layer, "step"):
                z = layer.step(z)
            else:
                z = layer(z)
        return z
