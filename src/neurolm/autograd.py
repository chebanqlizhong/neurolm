"""A minimal reverse-mode autodiff engine built on NumPy.

The engine is deliberately small and dependency-free: every neural network
component in :mod:`neurolm.model` is expressed with the operations defined
here, so gradients are derived automatically. This keeps the training stack
transparent and easy to extend.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

ArrayLike = np.ndarray | float | int


def _accum(current: np.ndarray | None, new: np.ndarray) -> np.ndarray:
    """Accumulate a gradient, creating it when it does not exist yet."""
    if current is None:
        return new
    return current + new


def _unbroadcast(grad: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Sum ``grad`` back down to ``shape`` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """An N-dimensional array that records how it was produced."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Iterable[Tensor] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward = lambda: None
        self._prev: tuple[Tensor, ...] = tuple(_children)
        self._op = _op

    # -- introspection -----------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        return len(self.data)

    def item(self) -> float:
        return float(self.data)

    @staticmethod
    def _coerce(other: object) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    # -- elementwise arithmetic -------------------------------------------
    def __add__(self, other: object) -> Tensor:
        other = self._coerce(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, _unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other.grad = _accum(other.grad, _unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other: object) -> Tensor:
        other = self._coerce(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, _unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other.grad = _accum(other.grad, _unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: object) -> Tensor:
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> Tensor:
        return self._coerce(other) + (-self)

    def __truediv__(self, other: object) -> Tensor:
        other = self._coerce(other)
        return self * other.pow(-1.0)

    def __pow__(self, exponent: float) -> Tensor:
        return self.pow(exponent)

    def pow(self, exponent: float) -> Tensor:
        out = Tensor(
            self.data**exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exponent}",
        )

        def _backward() -> None:
            if self.requires_grad:
                local = exponent * (self.data ** (exponent - 1))
                self.grad = _accum(self.grad, out.grad * local)

        out._backward = _backward
        return out

    # -- matrix operations -------------------------------------------------
    def matmul(self, other: object) -> Tensor:
        other = self._coerce(other)
        out = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                self.grad = _accum(self.grad, _unbroadcast(grad, self.shape))
            if other.requires_grad:
                grad = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                other.grad = _accum(other.grad, _unbroadcast(grad, other.shape))

        out._backward = _backward
        return out

    __matmul__ = matmul

    def __rmatmul__(self, other: object) -> Tensor:
        return self._coerce(other).matmul(self)

    # -- shape manipulation ------------------------------------------------
    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        if not axes:
            axes = tuple(range(self.ndim))[::-1]
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])  # type: ignore[assignment]
        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )
        inverse = np.argsort(axes)

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, np.transpose(out.grad, inverse))

        out._backward = _backward
        return out

    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    for ax in sorted(np.atleast_1d(axis)):
                        grad = np.expand_dims(grad, ax)
                self.grad = _accum(self.grad, np.broadcast_to(grad, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            n = self.data.size
        else:
            n = int(np.prod([self.shape[a] for a in np.atleast_1d(axis)]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def __getitem__(self, key: object) -> Tensor:
        out = Tensor(
            self.data[key],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="getitem",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, key, out.grad)
                self.grad = _accum(self.grad, grad)

        out._backward = _backward
        return out

    # -- activations -------------------------------------------------------
    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad / self.data)

        out._backward = _backward
        return out

    # -- autodiff ----------------------------------------------------------
    def backward(self) -> None:
        """Run reverse-mode differentiation starting from this scalar node."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = None


# ---------------------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------------------
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=axis, keepdims=True)
    out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op="softmax")

    def _backward() -> None:
        if x.requires_grad:
            dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
            x.grad = _accum(x.grad, probs * (out.grad - dot))

    out._backward = _backward
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    logp = shifted - logsumexp
    out = Tensor(logp, requires_grad=x.requires_grad, _children=(x,), _op="log_softmax")
    probs = np.exp(logp)

    def _backward() -> None:
        if x.requires_grad:
            dot = np.sum(out.grad, axis=axis, keepdims=True)
            x.grad = _accum(x.grad, out.grad - probs * dot)

    out._backward = _backward
    return out


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    xd = x.data
    mu = xd.mean(axis=-1, keepdims=True)
    var = xd.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (xd - mu) * inv
    out = Tensor(
        xhat * gamma.data + beta.data,
        requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad,
        _children=(x, gamma, beta),
        _op="layernorm",
    )
    n = xd.shape[-1]
    reduce_axes = tuple(range(out.ndim - 1))

    def _backward() -> None:
        go = out.grad
        if gamma.requires_grad:
            gamma.grad = _accum(gamma.grad, np.sum(go * xhat, axis=reduce_axes))
        if beta.requires_grad:
            beta.grad = _accum(beta.grad, np.sum(go, axis=reduce_axes))
        if x.requires_grad:
            dxhat = go * gamma.data
            dvar = np.sum(dxhat * (xd - mu) * (-0.5) * inv**3, axis=-1, keepdims=True)
            dmu = np.sum(dxhat * -inv, axis=-1, keepdims=True) + dvar * np.mean(
                -2.0 * (xd - mu), axis=-1, keepdims=True
            )
            dx = dxhat * inv + dvar * 2.0 * (xd - mu) / n + dmu / n
            x.grad = _accum(x.grad, dx)

    out._backward = _backward
    return out


def gelu(x: Tensor) -> Tensor:
    """Gaussian error linear unit (tanh approximation)."""
    xd = x.data
    c = np.sqrt(2.0 / np.pi)
    inner = c * (xd + 0.044715 * xd**3)
    t = np.tanh(inner)
    out = Tensor(0.5 * xd * (1.0 + t), requires_grad=x.requires_grad, _children=(x,), _op="gelu")

    def _backward() -> None:
        if x.requires_grad:
            dinner = c * (1.0 + 3.0 * 0.044715 * xd**2)
            local = 0.5 * (1.0 + t) + 0.5 * xd * (1.0 - t**2) * dinner
            x.grad = _accum(x.grad, out.grad * local)

    out._backward = _backward
    return out


def embedding(weight: Tensor, idx: np.ndarray) -> Tensor:
    """Row lookup with scatter-add backward (used for token/position embeddings)."""
    idx = np.asarray(idx)
    out = Tensor(weight.data[idx], requires_grad=weight.requires_grad, _children=(weight,), _op="embedding")

    def _backward() -> None:
        if weight.requires_grad:
            grad = np.zeros_like(weight.data)
            np.add.at(grad, idx, out.grad)
            weight.grad = _accum(weight.grad, grad)

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy between ``logits`` ``(..., V)`` and integer ``targets``."""
    targets = np.asarray(targets)
    x = logits.data
    shifted = x - np.max(x, axis=-1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    logp = shifted - logsumexp
    probs = np.exp(logp)

    flat_logp = logp.reshape(-1, logp.shape[-1])
    flat_targets = targets.reshape(-1)
    rows = np.arange(flat_targets.size)
    loss = float(-flat_logp[rows, flat_targets].mean())

    out = Tensor(loss, requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy")

    def _backward() -> None:
        if logits.requires_grad:
            grad = probs.copy().reshape(-1, probs.shape[-1])
            grad[rows, flat_targets] -= 1.0
            grad /= flat_targets.size
            grad = grad.reshape(logits.shape)
            logits.grad = _accum(logits.grad, grad * out.grad)

    out._backward = _backward
    return out
