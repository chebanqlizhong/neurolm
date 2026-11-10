"""The training loop.

:class:`Trainer` owns a model, an optimizer and a pair of datasets and runs the
standard language-modeling loop: sample a batch, compute cross-entropy, call
backward, clip gradients, update weights, and periodically evaluate and
checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..autograd import cross_entropy
from ..config import RunConfig
from ..data.dataset import TokenDataset
from ..models.gpt import GPT
from .checkpoint import save_checkpoint
from .optimizer import AdamW, AdamWConfig, clip_gradients, cosine_lr


@dataclass
class StepMetrics:
    """Metrics emitted for a single training or evaluation step."""

    step: int
    loss: float
    lr: float
    grad_norm: float = 0.0
    elapsed: float = 0.0


@dataclass
class History:
    """Accumulated metrics for a run."""

    train: list[StepMetrics] = field(default_factory=list)
    val: list[StepMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "train": [vars(m) for m in self.train],
            "val": [vars(m) for m in self.val],
        }


class Trainer:
    """Runs the language-model training loop for a :class:`RunConfig`."""

    def __init__(
        self,
        config: RunConfig,
        model: GPT,
        train_data: TokenDataset,
        val_data: TokenDataset,
        out_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.out_dir = Path(out_dir or config.out_dir) / config.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(config.train.seed)
        self.optimizer = AdamW(
            model.parameters(),
            AdamWConfig(
                lr=config.train.learning_rate,
                betas=(config.train.beta1, config.train.beta2),
                eps=config.train.eps,
                weight_decay=config.train.weight_decay,
            ),
        )
        self.history = History()

    # -- core steps --------------------------------------------------------
    def train_step(self, step: int) -> StepMetrics:
        cfg = self.config.train
        lr = cosine_lr(
            step,
            max_steps=cfg.max_steps,
            lr=cfg.learning_rate,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps,
        )
        self.optimizer.set_lr(lr)

        batch = self.train_data.sample_batch(self.config.data.batch_size)
        self.model.zero_grad()
        logits = self.model.forward(batch.inputs, training=True, rng=self.rng)
        loss = cross_entropy(logits, batch.targets)
        loss.backward()
        grad_norm = clip_gradients(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()

        return StepMetrics(step=step, loss=float(loss.data), lr=lr, grad_norm=grad_norm)

    def evaluate(self, step: int) -> StepMetrics:
        cfg = self.config.train
        losses = []
        for _ in range(cfg.eval_batches):
            batch = self.val_data.sample_batch(self.config.data.batch_size)
            logits = self.model.forward(batch.inputs, training=False)
            loss = cross_entropy(logits, batch.targets)
            losses.append(float(loss.data))
        return StepMetrics(step=step, loss=float(np.mean(losses)), lr=self.optimizer.lr)

    # -- driver ------------------------------------------------------------
    def fit(self, callback=None) -> History:
        cfg = self.config.train
        start = time.time()
        for step in range(cfg.max_steps):
            metrics = self.train_step(step)
            metrics.elapsed = time.time() - start
            self.history.train.append(metrics)

            if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
                val = self.evaluate(step)
                self.history.val.append(val)
                if callback is not None:
                    callback(metrics, val)

            if (
                self.config.checkpoint_interval > 0
                and step > 0
                and step % self.config.checkpoint_interval == 0
            ):
                self.save(f"step_{step}.npz", step)

        self.save("final.npz", cfg.max_steps)
        return self.history

    def save(self, filename: str, step: int) -> Path:
        return save_checkpoint(self.out_dir / filename, self.model, step)


__all__ = ["Trainer", "StepMetrics", "History"]
