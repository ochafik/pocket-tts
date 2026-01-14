# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Text conditioner using SentencePiece tokenization."""

import logging
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import sentencepiece

from pocket_tts_mlx.utils.loaders import download_if_necessary

logger = logging.getLogger(__name__)


class TokenizedText(NamedTuple):
    """Tokenized text container."""

    tokens: mx.array  # Shape: [1, T] or [B, T], dtype int32


class SentencePieceTokenizer:
    """SentencePiece tokenizer for text.

    Args:
        nbins: Vocabulary size (must match tokenizer's vocab size)
        tokenizer_path: Path to the SentencePiece model file
    """

    def __init__(self, nbins: int, tokenizer_path: str) -> None:
        logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        assert nbins == self.sp.vocab_size(), (
            f"SentencePiece tokenizer has vocab size={self.sp.vocab_size()} "
            f"but nbins={nbins} was specified"
        )

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text.

        Args:
            text: Input text string

        Returns:
            TokenizedText with shape [1, T]
        """
        token_ids = self.sp.encode(text, out_type=int)
        return TokenizedText(mx.array([token_ids], dtype=mx.int32))


class LUTConditioner(nn.Module):
    """Lookup Table conditioner for text.

    Uses an embedding table to convert tokens to vectors.

    Args:
        n_bins: Vocabulary size
        tokenizer_path: Path to SentencePiece model
        dim: Embedding dimension
        output_dim: Output dimension (should equal dim for pocket-tts)
    """

    def __init__(
        self,
        n_bins: int,
        tokenizer_path: str,
        dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.tokenizer = SentencePieceTokenizer(n_bins, tokenizer_path)
        # n_bins + 1 for padding token
        self.embed = nn.Embedding(n_bins + 1, dim)

    def prepare(self, x: str) -> TokenizedText:
        """Tokenize text string.

        Args:
            x: Input text

        Returns:
            TokenizedText containing token IDs
        """
        return self.tokenizer(x)

    def __call__(self, inputs: TokenizedText) -> mx.array:
        """Get embeddings for tokens.

        Args:
            inputs: TokenizedText containing token IDs [B, T]

        Returns:
            Embeddings [B, T, dim]
        """
        return self.embed(inputs.tokens)
