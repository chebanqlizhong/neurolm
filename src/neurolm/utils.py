"""Shared helpers: seeding, logging and run directories."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "neurolm", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and the hash salt for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def humanize_number(value: float) -> str:
    """Format large counts with a compact suffix (e.g. ``1.2M``)."""
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.1f}{unit}"
    return str(int(value))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "get_logger",
    "set_seed",
    "humanize_number",
    "ensure_dir",
    "write_json",
]
