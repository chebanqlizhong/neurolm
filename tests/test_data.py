"""Tests for the tokenizer and dataset."""

from __future__ import annotations

import numpy as np
import pytest

from neurolm.data.dataset import TokenDataset
from neurolm.tokenizer import BOS, EOS, ByteTokenizer


def test_roundtrip_ascii():
    tok = ByteTokenizer()
    text = "Hello, NeuroLM!"
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_unicode():
    tok = ByteTokenizer()
    text = "你好，世界 — naïve café 🚀"
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_added():
    tok = ByteTokenizer()
    ids = tok.encode("ab", add_special=True)
    assert ids[0] == BOS
    assert ids[-1] == EOS
    assert ids[1:-1] == [ord("a"), ord("b")]


def test_decode_ignores_special_tokens():
    tok = ByteTokenizer()
    assert tok.decode([BOS, ord("h"), ord("i"), EOS]) == "hi"


def test_batch_shapes():
    tokens = np.arange(200)
    ds = TokenDataset(tokens, block_size=16, seed=0)
    batch = ds.sample_batch(4)
    assert batch.inputs.shape == (4, 16)
    assert batch.targets.shape == (4, 16)
    # targets are the inputs shifted by one position
    assert np.allclose(batch.targets[:, :-1], batch.inputs[:, 1:])


def test_sampling_is_deterministic_with_seed():
    tokens = np.arange(500)
    a = TokenDataset(tokens, block_size=8, seed=7).sample_batch(3)
    b = TokenDataset(tokens, block_size=8, seed=7).sample_batch(3)
    assert np.array_equal(a.inputs, b.inputs)


def test_from_file_splits(corpus_file):
    train, val = TokenDataset.from_file(corpus_file, block_size=16, val_split=0.25)
    total = train.tokens.size + val.tokens.size
    assert val.tokens.size == pytest.approx(total * 0.25, abs=2)
    assert train.block_size == 16


def test_too_small_corpus_raises():
    with pytest.raises(ValueError):
        TokenDataset(np.arange(5), block_size=16)
