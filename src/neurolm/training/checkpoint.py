"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models.gpt import GPT


def save_checkpoint(
    path: str | Path,
    model: GPT,
    step: int,
) -> Path:
    """Persist model weights and the step counter to a ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: param.data for name, param in model.named_parameters()}
    arrays["_meta_step"] = np.asarray(step)
    np.savez(path, **arrays)
    return path


def load_checkpoint(path: str | Path, model: GPT) -> dict:
    """Load weights from a checkpoint into ``model`` (in place)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        for name, param in model.named_parameters():
            if name not in data:
                raise KeyError(f"Checkpoint missing parameter '{name}'")
            param.data = data[name].astype(np.float32)
        step = int(data["_meta_step"]) if "_meta_step" in data else 0
    return {"step": step}


__all__ = ["save_checkpoint", "load_checkpoint"]
