# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dummy quantizer for TTS (simple projection without actual quantization)."""

import mlx.core as mx
import mlx.nn as nn

from .conv import Conv1d


class DummyQuantizer(nn.Module):
    """Simplified quantizer that only provides output projection.

    This removes all quantization logic since pocket-tts doesn't use actual quantization,
    just a projection from the latent dimension to the Mimi embedding dimension.
    """

    def __init__(self, dimension: int, output_dimension: int):
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        # Use 1x1 convolution for projection
        self.output_proj = Conv1d(dimension, output_dimension, kernel_size=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Project latents to embedding dimension.

        Args:
            x: Input tensor [B, dimension, T]

        Returns:
            Output tensor [B, output_dimension, T]
        """
        return self.output_proj(x)
