# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Streaming transformer modules for MLX."""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .kv_cache import KVCache, RotatingKVCache, create_attention_mask


def apply_rope(
    q: mx.array,
    k: mx.array,
    offset: int = 0,
    max_period: float = 10_000.0,
) -> tuple[mx.array, mx.array]:
    """Apply rotary positional embeddings to queries and keys.

    Args:
        q: Queries, shape [B, T, H, D]
        k: Keys, shape [B, T, H, D]
        offset: Current offset for streaming
        max_period: Maximum period for cos/sin

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape
    assert (B, T, D) == (Bk, Tk, Dk)
    assert D > 0 and D % 2 == 0

    ds = mx.arange(D // 2)
    freqs = mx.exp(ds * (-math.log(max_period) * 2 / D))

    ts = mx.arange(T) + offset
    ts = ts.reshape(-1, 1, 1)

    q = q.reshape(B, T, H, D // 2, 2)
    k = k.reshape(B, T, Hk, D // 2, 2)

    qr = q[..., 0]
    qi = q[..., 1]
    kr = k[..., 0]
    ki = k[..., 1]

    rotr = mx.cos(freqs * ts)
    roti = mx.sin(freqs * ts)

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    qo = mx.stack([qor, qoi], axis=-1)
    ko = mx.stack([kor, koi], axis=-1)

    return qo.reshape(B, T, H, D), ko.reshape(B, T, Hk, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def __call__(
        self, q: mx.array, k: mx.array, offset: int = 0
    ) -> tuple[mx.array, mx.array]:
        return apply_rope(q, k, offset, self.max_period)


class LayerScale(nn.Module):
    """Learnable layer scaling."""

    def __init__(self, dim: int, init: float = 1.0):
        super().__init__()
        self.scale = mx.full((dim,), init)

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


class StreamingMultiheadAttention(nn.Module):
    """Multi-head attention with streaming support (no context window)."""

    def __init__(self, embed_dim: int, num_heads: int, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = RotaryEmbedding(max_period=max_period)

        out_dim = 3 * embed_dim  # Q, K, V
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        """Forward pass with KV cache.

        Args:
            x: Input tensor [B, T, D]
            cache: KVCache for streaming

        Returns:
            Output tensor [B, T, D]
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        projected = self.in_proj(x)
        projected = projected.reshape(B, T, 3, self.num_heads, self.head_dim)
        q = projected[:, :, 0]  # [B, T, H, D]
        k = projected[:, :, 1]
        v = projected[:, :, 2]

        # Apply RoPE
        q, k = self.rope(q, k, offset=cache.offset)

        # Transpose for attention: [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Update cache and get full K, V
        k, v = cache.update_and_fetch(k, v)

        # Create causal mask
        mask = None
        if T > 1:
            # Create additive causal mask
            offset = cache.offset - T  # Offset before we added current tokens
            rinds = mx.arange(offset + T)
            linds = mx.arange(offset, offset + T) if offset else mx.arange(T)
            mask = linds[:, None] < rinds[None]
            mask = mask * -1e9
            mask = mask.astype(q.dtype)

        # Scaled dot-product attention
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Transpose back and reshape: [B, T, H*D]
        x = x.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x


class MimiStreamingMultiheadAttention(nn.Module):
    """Multi-head attention with streaming support and context window (for Mimi)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context: int,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.context = context
        self.rope = RotaryEmbedding(max_period=max_period)

        out_dim = 3 * embed_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x: mx.array, cache: RotatingKVCache) -> mx.array:
        """Forward pass with rotating KV cache.

        Args:
            x: Input tensor [B, T, D]
            cache: RotatingKVCache for streaming with context window

        Returns:
            Output tensor [B, T, D]
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        projected = self.in_proj(x)
        projected = projected.reshape(B, T, 3, self.num_heads, self.head_dim)
        q = projected[:, :, 0]
        k = projected[:, :, 1]
        v = projected[:, :, 2]

        # Apply RoPE
        q, k = self.rope(q, k, offset=cache.offset)

        # Transpose for attention: [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Update cache and get K, V within context window
        k, v = cache.update_and_fetch(k, v)

        # Create context-limited causal mask
        K_len = k.shape[2]
        Q_len = T

        # Position indices
        pos_q = mx.arange(cache.offset - T, cache.offset).reshape(1, Q_len, 1)
        pos_k = mx.arange(cache.offset - K_len, cache.offset).reshape(1, 1, K_len)

        # Mask: causal + context window
        delta = pos_q - pos_k
        attn_mask = (delta >= 0) & (delta < self.context)
        attn_mask = mx.where(attn_mask, mx.array(0.0), mx.array(-1e9))
        attn_mask = attn_mask.astype(q.dtype)

        # Scaled dot-product attention
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)

        # Transpose back and reshape
        x = x.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x


class StreamingTransformerLayer(nn.Module):
    """Transformer layer with streaming support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None = None,
        max_period: float = 10000.0,
        layer_scale: float | None = None,
        attention_kind: str = "mimi",
    ):
        super().__init__()

        if attention_kind == "mimi" and context is not None:
            self.self_attn = MimiStreamingMultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                context=context,
                max_period=max_period,
            )
        else:
            self.self_attn = StreamingMultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                max_period=max_period,
            )

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is not None:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None

    def __call__(self, x: mx.array, cache: KVCache | RotatingKVCache) -> mx.array:
        # Self-attention block
        h = self.norm1(x)
        h = self.self_attn(h, cache)
        if self.layer_scale_1 is not None:
            h = self.layer_scale_1(h)
        x = x + h

        # Feed-forward block
        h = self.norm2(x)
        h = nn.gelu(self.linear1(h))
        h = self.linear2(h)
        if self.layer_scale_2 is not None:
            h = self.layer_scale_2(h)
        x = x + h

        return x


@dataclass
class LayerCache:
    """Cache for a single transformer layer."""

    self_attn: KVCache | RotatingKVCache


class StreamingTransformer(nn.Module):
    """Streaming transformer with multiple layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        context: int | None = None,
        max_period: float = 10000.0,
        layer_scale: float | None = None,
        kind: str = "mimi",
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context = context
        self.max_period = max_period

        self.layers = [
            StreamingTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                context=context,
                max_period=max_period,
                layer_scale=layer_scale,
                attention_kind=kind,
            )
            for _ in range(num_layers)
        ]

    def make_cache(self, batch_size: int = 1) -> list[KVCache | RotatingKVCache]:
        """Create KV caches for all layers."""
        head_dim = self.d_model // self.num_heads
        caches = []
        for _ in range(self.num_layers):
            if self.context is not None:
                cache = RotatingKVCache(
                    head_dim=head_dim,
                    n_kv_heads=self.num_heads,
                    max_size=self.context,
                )
            else:
                cache = KVCache(head_dim=head_dim, n_kv_heads=self.num_heads)
            caches.append(cache)
        return caches

    def reset_cache(self, cache: list[KVCache | RotatingKVCache]):
        """Reset all caches."""
        for c in cache:
            c.reset()

    def __call__(
        self, x: mx.array, cache: list[KVCache | RotatingKVCache] | None = None
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor [B, T, D]
            cache: List of KVCache objects for each layer

        Returns:
            Output tensor [B, T, D]
        """
        if cache is None:
            cache = self.make_cache()

        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, layer_cache)

        return x


class ProjectedTransformer(nn.Module):
    """Transformer with input/output projections."""

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...],
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None,
        context: int,
        max_period: float,
        dim_feedforward: int,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_scale=layer_scale,
            context=context,
            max_period=max_period,
            dim_feedforward=dim_feedforward,
        )
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions

        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = []
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(None)  # Identity
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    def make_cache(self, batch_size: int = 1) -> list[KVCache | RotatingKVCache]:
        return self.transformer.make_cache(batch_size)

    def reset_cache(self, cache: list[KVCache | RotatingKVCache]):
        self.transformer.reset_cache(cache)

    def __call__(
        self, x: mx.array, cache: list[KVCache | RotatingKVCache] | None = None
    ) -> list[mx.array]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, T] (NCL format)
            cache: KV caches for streaming

        Returns:
            List of output tensors [B, C', T] for each output dimension
        """
        # NCL -> NLC (channels last)
        x = x.swapaxes(1, 2)

        if self.input_proj is not None:
            x = self.input_proj(x)

        z = self.transformer(x, cache)

        ys = []
        for output_proj in self.output_projs:
            if output_proj is None:
                y = z
            else:
                y = output_proj(z)
            # NLC -> NCL
            y = y.swapaxes(1, 2)
            ys.append(y)

        return ys
