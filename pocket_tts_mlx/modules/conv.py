# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Streaming convolution modules for MLX."""

import math

import mlx.core as mx
import mlx.nn as nn


class Conv1d(nn.Module):
    """1D convolution with NCL input/output layout (matching PyTorch).

    Internally converts to MLX's NLC layout for computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        scale = 1 / (in_channels * kernel_size)
        # MLX weight shape: (out_channels, kernel_size, in_channels // groups)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        self.bias = mx.zeros(out_channels) if bias else None
        self._padding = padding
        self._groups = groups
        self._stride = stride
        self._dilation = dilation
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels

    def __call__(self, xs: mx.array) -> mx.array:
        # MLX uses NLC whereas PyTorch uses NCL
        y = mx.conv1d(
            xs.swapaxes(-1, -2),  # NCL -> NLC
            self.weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(-1, -2)  # NLC -> NCL


class ConvTranspose1d(nn.Module):
    """1D transposed convolution with NCL input/output layout.

    Internally converts to MLX's NLC layout for computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        scale = 1 / (in_channels * kernel_size)
        # MLX weight shape for conv_transpose: (out_channels // groups, kernel_size, in_channels)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels // groups, kernel_size, in_channels),
        )
        self.bias = mx.zeros(out_channels) if bias else None
        self._padding = padding
        self._groups = groups
        self._stride = stride
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Handle depthwise case
        if groups == in_channels and groups == out_channels:
            eye = (
                mx.eye(out_channels)
                .astype(self.weight.dtype)
                .reshape((out_channels, 1, out_channels))
            )
            eye = mx.repeat(eye, repeats=kernel_size, axis=1)
            self._expanded_weight = mx.repeat(self.weight, repeats=groups, axis=0) * eye
            self._expanded_groups = 1
        elif groups > 1:
            raise ValueError("groups > 1 (non-depthwise) not supported in ConvTranspose1d")
        else:
            self._expanded_weight = self.weight
            self._expanded_groups = groups

    def update_in_place(self):
        """Update expanded weight after loading weights."""
        groups = self._groups
        in_channels = self._in_channels
        out_channels = self._out_channels
        kernel_size = self._kernel_size
        if groups == in_channels and groups == out_channels:
            eye = (
                mx.eye(out_channels)
                .astype(self.weight.dtype)
                .reshape((out_channels, 1, out_channels))
            )
            eye = mx.repeat(eye, repeats=kernel_size, axis=1)
            self._expanded_weight = mx.repeat(self.weight, repeats=groups, axis=0) * eye
            self._expanded_groups = 1
        elif groups > 1:
            raise ValueError("groups > 1 (non-depthwise) not supported in ConvTranspose1d")
        else:
            self._expanded_weight = self.weight
            self._expanded_groups = groups

    def __call__(self, xs: mx.array) -> mx.array:
        y = mx.conv_transpose1d(
            xs.swapaxes(-1, -2),  # NCL -> NLC
            self._expanded_weight,
            stride=self._stride,
            padding=self._padding,
            groups=self._expanded_groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(-1, -2)  # NLC -> NCL


def get_extra_padding_for_conv1d(
    xs: mx.array,
    kernel_size: int,
    stride: int,
    padding_total: int,
) -> int:
    """Calculate extra padding needed for a convolution to ensure last window is full."""
    length = xs.shape[-1]
    nframes = max(length + padding_total - kernel_size, 0) / stride + 1.0
    ideal_len = (int(math.ceil(nframes)) - 1) * stride + kernel_size - padding_total
    return max(0, ideal_len - length)


def unpad1d(xs: mx.array, unpad_l: int, unpad_r: int) -> mx.array:
    """Remove padding from both sides of a 1D tensor."""
    left = unpad_l
    right = xs.shape[-1] - unpad_r
    return xs[..., left:right]


class StreamingConv1d(nn.Module):
    """Streaming 1D convolution with causal padding.

    Maintains state between calls for streaming inference.
    Use `step()` for streaming mode, `__call__()` for full sequence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ["constant", "edge"], f"Unsupported pad_mode: {pad_mode}"
        self._pad_mode = pad_mode
        self._kernel_size = kernel_size
        self._out_channels = out_channels
        self.conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self._prev_xs: mx.array | None = None
        self._left_pad_applied = False

    def reset_state(self):
        """Reset streaming state for a new sequence."""
        self._prev_xs = None
        self._left_pad_applied = False

    def __call__(self, xs: mx.array) -> mx.array:
        """Process full sequence with appropriate padding."""
        kernel_size = self._kernel_size
        effective_kernel = (kernel_size - 1) * self.conv._dilation + 1
        padding_total = effective_kernel - self.conv._stride
        extra_padding = get_extra_padding_for_conv1d(
            xs,
            kernel_size=effective_kernel,
            stride=self.conv._stride,
            padding_total=padding_total,
        )
        # Causal padding: all padding on the left
        padding_left = padding_total
        padding_right = extra_padding
        z = (0, 0)
        widths = [z, z, (padding_left, padding_right)]
        xs_padded = mx.pad(xs, pad_width=widths, mode=self._pad_mode)
        return self.conv(xs_padded)

    def step(self, xs: mx.array) -> mx.array:
        """Process one chunk in streaming mode.

        Args:
            xs: Input chunk of shape [B, C, T]

        Returns:
            Output chunk
        """
        b, _, length = xs.shape
        if length == 0:
            return mx.zeros((b, self._out_channels, 0))

        stride = self.conv._stride
        dilation = self.conv._dilation
        effective_kernel = (self._kernel_size - 1) * dilation + 1

        # Apply left padding on first call
        if not self._left_pad_applied:
            self._left_pad_applied = True
            padding_total = effective_kernel - stride
            xs = mx.pad(
                xs, pad_width=((0, 0), (0, 0), (padding_total, 0)), mode=self._pad_mode
            )

        # Concatenate with previous state
        if self._prev_xs is not None:
            xs = mx.concatenate([self._prev_xs, xs], axis=-1)

        length = xs.shape[-1]
        nframes = max(length + stride - effective_kernel, 0) // stride

        if nframes > 0:
            offset = nframes * stride
            self._prev_xs = xs[..., offset:]
            in_l = (nframes - 1) * stride + effective_kernel
            if in_l > 0:
                xs = xs[..., :in_l]
                return self.conv(xs)
            else:
                return mx.zeros((b, self._out_channels, 0))
        else:
            self._prev_xs = xs
            return mx.zeros((b, self._out_channels, 0))


class StreamingConvTranspose1d(nn.Module):
    """Streaming transposed 1D convolution.

    Maintains state between calls for streaming inference.
    Use `step()` for streaming mode, `__call__()` for full sequence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._kernel_size = kernel_size
        self._out_channels = out_channels
        self.convtr = ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self._prev_ys: mx.array | None = None

    def reset_state(self):
        """Reset streaming state for a new sequence."""
        self._prev_ys = None

    def __call__(self, xs: mx.array) -> mx.array:
        """Process full sequence with appropriate unpadding."""
        stride = self.convtr._stride
        padding_total = max(self._kernel_size - stride, 0)
        xs = self.convtr(xs)
        # Causal: unpad from the right
        return unpad1d(xs, unpad_l=0, unpad_r=padding_total)

    def step(self, xs: mx.array) -> mx.array:
        """Process one chunk in streaming mode.

        Args:
            xs: Input chunk of shape [B, C, T]

        Returns:
            Output chunk
        """
        b, _, length = xs.shape
        if length == 0:
            return mx.zeros((b, self._out_channels, 0))

        stride = self.convtr._stride
        ys = self.convtr(xs)
        ot = ys.shape[-1]

        # Add overlap from previous step
        if self._prev_ys is not None:
            prev_ys = self._prev_ys
            pt = prev_ys.shape[-1]
            # Remove bias from prev_ys before adding (bias was already applied)
            if self.convtr.bias is not None:
                prev_ys = prev_ys - self.convtr.bias[None, :, None]
            ys1 = ys[..., :pt] + prev_ys
            ys2 = ys[..., pt:]
            ys = mx.concatenate([ys1, ys2], axis=-1)

        # Split output and overlap for next step
        invalid_steps = self._kernel_size - stride
        ys, self._prev_ys = ys[..., : ot - invalid_steps], ys[..., ot - invalid_steps :]
        return ys


class ConvDownsample1d(nn.Module):
    """Convolutional downsampling layer."""

    def __init__(self, stride: int, dim: int):
        super().__init__()
        self.conv = StreamingConv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=2 * stride,
            stride=stride,
            dilation=1,
            groups=1,
            bias=False,
            pad_mode="edge",
        )

    def reset_state(self):
        self.conv.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        return self.conv(xs)

    def step(self, xs: mx.array) -> mx.array:
        return self.conv.step(xs)


class ConvTrUpsample1d(nn.Module):
    """Convolutional upsampling layer using transposed convolution."""

    def __init__(self, stride: int, dim: int):
        super().__init__()
        self.convtr = StreamingConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=2 * stride,
            stride=stride,
            groups=dim,  # Depthwise
            bias=False,
        )

    def reset_state(self):
        self.convtr.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        return self.convtr(xs)

    def step(self, xs: mx.array) -> mx.array:
        return self.convtr.step(xs)
