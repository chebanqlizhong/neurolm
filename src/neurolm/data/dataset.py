"""Dataset utilities: contiguous token stream, splits and mini-batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..tokenizer import ByteTokenizer


@dataclass
class Batch:
    """A single training batch of input/target token ids."""

    inputs: np.ndarray  # (batch, block)
    targets: np.ndarray  # (batch, block)

    def __len__(self) -> int:
        return self.inputs.shape[0]


class TokenDataset:
    """A split of a token stream that yields random contiguous blocks.

    The corpus is tokenized once, split into train/validation regions and stored
    as a single ``int64`` array. Sampling a batch is then pure integer indexing,
    which keeps the training loop free of I/O and shuffling overhead.
    """

    def __init__(self, tokens: np.ndarray, block_size: int, seed: int = 0) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1-D array")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if tokens.size <= block_size + 1:
            raise ValueError(
                f"corpus too small: {tokens.size} tokens for block_size={block_size}"
            )
        self.tokens = tokens.astype(np.int64, copy=False)
        self.block_size = block_size
        self._rng = np.random.default_rng(seed)

    @property
    def num_blocks(self) -> int:
        return max(0, self.tokens.size - self.block_size - 1)

    def sample_batch(self, batch_size: int) -> Batch:
        starts = self._rng.integers(0, self.num_blocks, size=batch_size)
        offsets = np.arange(self.block_size)
        idx = starts[:, None] + offsets[None, :]
        inputs = self.tokens[idx]
        targets = self.tokens[idx + 1]
        return Batch(inputs=inputs, targets=targets)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        block_size: int,
        val_split: float = 0.1,
        seed: int = 0,
    ) -> tuple[TokenDataset, TokenDataset]:
        """Tokenize a corpus and return ``(train, validation)`` datasets."""
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0, 1)")
        tokenizer = ByteTokenizer()
        tokens = tokenizer.read_corpus(path)
        split = int(tokens.size * (1.0 - val_split))
        train_tokens = tokens[:split]
        val_tokens = tokens[split:] if split < tokens.size else tokens
        train = cls(train_tokens, block_size, seed=seed)
        val = cls(val_tokens, block_size, seed=seed + 1)
        return train, val


__all__ = ["Batch", "TokenDataset"]
