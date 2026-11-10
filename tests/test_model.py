"""Tests for the transformer model."""

from __future__ import annotations

import numpy as np

from neurolm.autograd import cross_entropy
from neurolm.models.gpt import GPT


def test_forward_output_shape(tiny_model_config):
    model = GPT(tiny_model_config, seed=0)
    idx = np.random.randint(0, tiny_model_config.vocab_size, size=(2, 5))
    logits = model.forward(idx, training=False)
    assert logits.shape == (2, 5, tiny_model_config.vocab_size)


def test_sequence_length_limit(tiny_model_config):
    model = GPT(tiny_model_config, seed=0)
    idx = np.random.randint(0, tiny_model_config.vocab_size, size=(1, 99))
    try:
        model.forward(idx, training=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError for over-long sequence")


def test_causality_no_future_leakage(tiny_model_config):
    """Changing a future token must not affect earlier logits."""
    model = GPT(tiny_model_config, seed=1)
    idx = np.array([[5, 10, 15, 20, 25]])
    logits_a = model.forward(idx, training=False).data

    idx2 = idx.copy()
    idx2[0, -1] = 200
    logits_b = model.forward(idx2, training=False).data

    # positions before the last must be identical
    assert np.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-6)
    assert not np.allclose(logits_a[0, -1], logits_b[0, -1])


def test_training_reduces_loss(tiny_model_config):
    model = GPT(tiny_model_config, seed=0)
    rng = np.random.default_rng(0)
    data = np.random.randint(0, tiny_model_config.vocab_size, size=(4, 16))

    def loss_value():
        logits = model.forward(data, training=False)
        return float(cross_entropy(logits, data).data)

    before = loss_value()
    lr = 5e-3
    for _ in range(40):
        model.zero_grad()
        logits = model.forward(data, training=True, rng=rng)
        loss = cross_entropy(logits, data)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.data = (param.data - lr * param.grad).astype(np.float32)
    after = loss_value()
    assert after < before, f"loss did not decrease: {before} -> {after}"


def test_parameter_names_unique(tiny_model_config):
    model = GPT(tiny_model_config, seed=0)
    names = [name for name, _ in model.named_parameters()]
    assert len(names) == len(set(names))
