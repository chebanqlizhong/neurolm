"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurolm.config import ModelConfig  # noqa: E402


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=258,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(
        "The quick brown fox jumps over the lazy dog. " * 40
        + "A language model predicts the next token in a sequence. " * 40,
        encoding="utf-8",
    )
    return path
