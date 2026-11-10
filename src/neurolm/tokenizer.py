"""Byte-level tokenizer.

A byte-level tokenizer needs no training corpus statistics, never emits an
out-of-vocabulary token and round-trips arbitrary UTF-8 text exactly. That makes
it the right default for a from-scratch language model whose first goal is a
correct, reproducible training loop rather than state-of-the-art compression.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

VOCAB_SIZE = 256
BOS = 256
EOS = 257
SPECIAL_TOKENS = 2
FULL_VOCAB_SIZE = VOCAB_SIZE + SPECIAL_TOKENS


class ByteTokenizer:
    """Encode text to bytes and decode bytes back to text."""

    vocab_size = FULL_VOCAB_SIZE

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special:
            ids = [BOS, *ids, EOS]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        byte_values = [int(i) for i in ids if 0 <= int(i) < VOCAB_SIZE]
        return bytes(byte_values).decode("utf-8", errors="replace")

    def encode_array(self, text: str) -> np.ndarray:
        return np.asarray(self.encode(text), dtype=np.int64)

    def read_corpus(self, path: str | Path) -> np.ndarray:
        """Load a UTF-8 corpus file and return the token id array."""
        text = Path(path).read_text(encoding="utf-8")
        return self.encode_array(text)


__all__ = ["ByteTokenizer", "VOCAB_SIZE", "FULL_VOCAB_SIZE", "BOS", "EOS"]
