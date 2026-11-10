"""Tests for the autograd engine, including a numerical gradient check."""

from __future__ import annotations

import numpy as np

from neurolm.autograd import (
    Tensor,
    cross_entropy,
    gelu,
    layer_norm,
    log_softmax,
    softmax,
)


def test_add_mul_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    out = ((a * b) + a).sum()
    out.backward()
    assert np.allclose(a.grad, [5.0, 6.0, 7.0])
    assert np.allclose(b.grad, [1.0, 2.0, 3.0])


def test_broadcast_backward():
    a = Tensor(np.ones((3, 4)), requires_grad=True)
    b = Tensor(np.arange(4.0), requires_grad=True)
    out = (a + b).sum()
    out.backward()
    assert np.allclose(a.grad, np.ones((3, 4)))
    assert np.allclose(b.grad, np.full(4, 3.0))


def test_matmul_backward_shapes():
    a = Tensor(np.random.rand(2, 3), requires_grad=True)
    b = Tensor(np.random.rand(3, 5), requires_grad=True)
    out = a.matmul(b).sum()
    out.backward()
    assert a.grad.shape == (2, 3)
    assert b.grad.shape == (3, 5)


def test_softmax_rows_sum_to_one():
    x = Tensor(np.random.rand(4, 7), requires_grad=True)
    probs = softmax(x, axis=-1)
    assert np.allclose(probs.data.sum(axis=-1), 1.0, atol=1e-6)


def test_log_softmax_matches_log_of_softmax():
    x = Tensor(np.random.rand(3, 5), requires_grad=True)
    a = log_softmax(x, axis=-1).data
    b = np.log(softmax(x, axis=-1).data)
    assert np.allclose(a, b, atol=1e-6)


def test_cross_entropy_uniform_is_log_vocab():
    logits = Tensor(np.zeros((2, 4, 5)), requires_grad=True)
    targets = np.random.randint(0, 5, size=(2, 4))
    loss = cross_entropy(logits, targets)
    assert np.isclose(float(loss.data), np.log(5), atol=1e-6)


def _numerical_grad(fn, params, eps=1e-3):
    grads = []
    for param in params:
        grad = np.zeros_like(param.data)
        it = np.nditer(param.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = float(param.data[idx])
            param.data[idx] = orig + eps
            plus = float(fn().data)
            param.data[idx] = orig - eps
            minus = float(fn().data)
            param.data[idx] = orig
            grad[idx] = (plus - minus) / (2 * eps)
            it.iternext()
        grads.append(grad)
    return grads


def test_gradient_check_composite():
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    w = Tensor(rng.normal(size=(4, 4)), requires_grad=True)

    def fn():
        x = layer_norm(a, Tensor(np.ones(4)), Tensor(np.zeros(4)))
        h = gelu(x.matmul(w) + b)
        return (softmax(h, axis=-1) ** 2).sum()

    out = fn()
    out.backward()
    numerical = _numerical_grad(fn, [a, b, w])
    for param, expected in zip([a, b, w], numerical, strict=True):
        assert np.allclose(param.grad, expected, atol=1e-2), param


def test_getitem_scatter_add():
    x = Tensor(np.zeros((4,)), requires_grad=True)
    idx = np.array([0, 2, 2, 3])
    out = x[idx].sum()
    out.backward()
    assert np.allclose(x.grad, [1.0, 0.0, 2.0, 1.0])
