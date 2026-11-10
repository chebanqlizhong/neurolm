"""Optimization: AdamW, learning-rate schedule and gradient clipping."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..models.parameter import Parameter


@dataclass
class AdamWConfig:
    lr: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01


class AdamW:
    """Adam with decoupled weight decay.

    Decoupled weight decay (as in Loshchilov & Hutter, 2019) shrinks weights
    independently of the adaptive gradient, which regularizes far more
    predictably than adding ``wd * param`` to the gradient.
    """

    def __init__(self, params: list[Parameter], config: AdamWConfig | None = None) -> None:
        self.params = params
        self.config = config or AdamWConfig()
        self.step_count = 0
        self._m = [np.zeros_like(p.data) for p in params]
        self._v = [np.zeros_like(p.data) for p in params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def set_lr(self, lr: float) -> None:
        self.config.lr = lr

    @property
    def lr(self) -> float:
        return self.config.lr

    def step(self) -> None:
        self.step_count += 1
        cfg = self.config
        b1, b2 = cfg.betas
        bias1 = 1.0 - b1**self.step_count
        bias2 = 1.0 - b2**self.step_count

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad.astype(np.float32)
            self._m[i] = b1 * self._m[i] + (1.0 - b1) * grad
            self._v[i] = b2 * self._v[i] + (1.0 - b2) * (grad * grad)
            m_hat = self._m[i] / bias1
            v_hat = self._v[i] / bias2
            update = m_hat / (np.sqrt(v_hat) + cfg.eps)
            if cfg.weight_decay > 0.0:
                update = update + cfg.weight_decay * param.data
            param.data = (param.data - cfg.lr * update).astype(np.float32)


def cosine_lr(step: int, *, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def clip_gradients(params: list[Parameter], max_norm: float) -> float:
    """Clip gradients by global norm; returns the pre-clip norm."""
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(np.sum(param.grad.astype(np.float64) ** 2))
    norm = math.sqrt(total)
    if max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for param in params:
            if param.grad is not None:
                param.grad = (param.grad * scale).astype(np.float32)
    return norm


__all__ = ["AdamW", "AdamWConfig", "cosine_lr", "clip_gradients"]
