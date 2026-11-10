"""Parameter container and initialisation helpers.

Parameters are wrapped in :class:`Parameter`, a :class:`~neurolm.autograd.Tensor`
subclass that always requires gradients and carries a stable name so optimizers
and checkpoints can refer to it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..autograd import Tensor


class Parameter(Tensor):
    """A trainable tensor."""

    def __init__(self, data: np.ndarray, name: str = "") -> None:
        super().__init__(data, requires_grad=True)
        self.name = name

    @property
    def numel(self) -> int:
        return int(self.data.size)


class ParameterDict(dict):
    """An ordered ``name -> Parameter`` mapping with convenience helpers."""

    def parameters(self) -> Iterator[Parameter]:
        return iter(self.values())

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        return iter(self.items())

    def zero_grad(self) -> None:
        for param in self.values():
            param.grad = None

    def numel(self) -> int:
        return sum(p.numel for p in self.values())


def init_normal(
    shape: tuple[int, ...],
    name: str,
    std: float,
    rng: np.random.Generator,
    mean: float = 0.0,
) -> Parameter:
    """Gaussian-initialised parameter."""
    data = rng.normal(mean, std, size=shape).astype(np.float32)
    return Parameter(data, name=name)


def init_zeros(shape: tuple[int, ...], name: str) -> Parameter:
    return Parameter(np.zeros(shape, dtype=np.float32), name=name)


def init_ones(shape: tuple[int, ...], name: str) -> Parameter:
    return Parameter(np.ones(shape, dtype=np.float32), name=name)


__all__ = [
    "Parameter",
    "ParameterDict",
    "init_normal",
    "init_zeros",
    "init_ones",
]
