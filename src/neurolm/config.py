"""Configuration handling for training runs.

Configs are plain YAML files (see ``configs/``) parsed into typed dataclasses so
that every hyper-parameter is validated once, at load time, instead of being
re-checked deep inside the training loop.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    """Architecture of the decoder-only transformer."""

    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("d_model and n_layers must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class DataConfig:
    """Where the training text comes from and how it is batched."""

    corpus_path: str = "data/corpus.txt"
    val_split: float = 0.1
    batch_size: int = 16
    block_size: int = 128
    seed: int = 1337


@dataclass
class TrainConfig:
    """Optimization schedule and logging cadence."""

    max_steps: int = 500
    eval_interval: int = 50
    eval_batches: int = 20
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    seed: int = 1337
    device: str = "cpu"


@dataclass
class RunConfig:
    """Top-level run configuration."""

    run_name: str = "neurolm"
    out_dir: str = "runs"
    checkpoint_interval: int = 250
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(dc_type: type, payload: dict[str, Any] | None) -> Any:
    """Instantiate a (possibly nested) dataclass from a plain dict."""
    payload = payload or {}
    hints = get_type_hints(dc_type)
    known = {f.name: f for f in fields(dc_type)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {dc_type.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        ftype = hints[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML run configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _build(RunConfig, raw)


def to_dict(config: Any) -> dict[str, Any]:
    """Recursively convert a config dataclass to a plain dict."""
    if is_dataclass(config):
        return {f.name: to_dict(getattr(config, f.name)) for f in fields(config)}
    return config


def save_config(config: Any, path: str | Path) -> None:
    """Persist a config dataclass as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(config), handle, sort_keys=False)


__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "RunConfig",
    "load_config",
    "save_config",
    "to_dict",
    "dataclasses",
]
