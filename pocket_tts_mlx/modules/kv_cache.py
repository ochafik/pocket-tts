# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from mlx-examples: https://github.com/ml-explore/mlx-examples
"""KV Cache implementation for streaming transformers."""

from typing import Any

import mlx.core as mx


class KVCache:
    """Key-Value cache for efficient autoregressive generation.

    Stores and manages key/value tensors for attention layers,
    growing the cache as needed during generation.
    """

    def __init__(self, head_dim: int | tuple[int, int], n_kv_heads: int):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.step = 256  # Allocation step size

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Update cache with new keys/values and return full cached tensors.

        Args:
            keys: New keys of shape [B, n_kv_heads, S, head_dim]
            values: New values of shape [B, n_kv_heads, S, head_dim]

        Returns:
            Tuple of (cached_keys, cached_values) including new entries
        """
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        assert self.values is not None
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self):
        """Reset the cache for a new sequence."""
        self.offset = 0
        self.keys = None
        self.values = None

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Return current cache state."""
        return self.keys, self.values


class RotatingKVCache:
    """Rotating KV cache with fixed maximum size.

    Useful for models with limited context windows.
    Once the cache reaches max_size, old entries are rotated out.
    """

    def __init__(
        self,
        head_dim: int | tuple[int, int],
        n_kv_heads: int,
        max_size: int,
        keep: int = 0,
        step: int = 256,
    ):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keep = keep
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(
        self, trim_size: int, v: mx.array, append: mx.array | None = None
    ) -> mx.array:
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Update cache with new keys/values, rotating out old entries if needed."""
        prev = self.offset
        B, _, S = keys.shape[:3]

        # Prefill mode
        if S > 1:
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                trim_size = self.keys.shape[2] - self.max_size + 1
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
            self.offset += S
            self._idx = self.keys.shape[2]
            return self.keys, self.values

        # Generation mode
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, self.n_kv_heads, new_size, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, new_size, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + 1, :] = keys
        assert self.values is not None
        self.values[..., self._idx : self._idx + 1, :] = values
        self.offset += 1
        self._idx += 1

        # If buffer not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def reset(self):
        """Reset the cache for a new sequence."""
        self.offset = 0
        self._idx = 0
        self.keys = None
        self.values = None

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Return current cache state."""
        return self.keys, self.values


def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    """Create an additive causal mask for attention.

    Args:
        N: Sequence length
        offset: Offset for the mask (for cached sequences)

    Returns:
        Mask tensor where future positions have -1e9
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Any | None = None) -> mx.array | None:
    """Create attention mask for a sequence, accounting for cache.

    Args:
        h: Hidden states tensor [B, T, D]
        cache: Optional list of KVCache objects

    Returns:
        Attention mask or None if not needed
    """
    T = h.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if isinstance(c, RotatingKVCache):
                offset = min(c.max_size - 1, c.offset)
            else:
                offset = c.offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask
