"""Evaluation utilities: loss, perplexity and sampling-based generation."""

from __future__ import annotations

import math

import numpy as np

from ..autograd import cross_entropy
from ..data.dataset import TokenDataset
from ..models.gpt import GPT
from ..tokenizer import ByteTokenizer


def evaluate_loss(
    model: GPT,
    dataset: TokenDataset,
    batch_size: int,
    num_batches: int,
) -> float:
    """Mean cross-entropy over ``num_batches`` validation batches."""
    losses = []
    for _ in range(num_batches):
        batch = dataset.sample_batch(batch_size)
        logits = model.forward(batch.inputs, training=False)
        losses.append(float(cross_entropy(logits, batch.targets).data))
    return float(np.mean(losses))


def perplexity(loss: float) -> float:
    """Perplexity is ``exp(loss)`` for natural-log cross-entropy."""
    return float(math.exp(loss))


@np.errstate(over="ignore")
def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    cutoff = np.sort(logits)[-k]
    return np.where(logits < cutoff, -np.inf, logits)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def generate(
    model: GPT,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
) -> str:
    """Autoregressively sample a completion for ``prompt``."""
    rng = np.random.default_rng(seed)
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    for _ in range(max_new_tokens):
        context = np.asarray([ids[-model.config.max_seq_len :]], dtype=np.int64)
        logits = model.forward(context, training=False)
        last = logits.data[0, -1, :]
        last = _top_k_filter(_apply_temperature(last, temperature), top_k)
        probs = _softmax_np(last.astype(np.float64))
        next_id = int(rng.choice(probs.shape[0], p=probs))
        ids.append(next_id)

    return tokenizer.decode(ids)


__all__ = ["evaluate_loss", "perplexity", "generate"]
