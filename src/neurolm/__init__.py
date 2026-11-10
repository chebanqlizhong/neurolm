"""NeuroLM — a from-scratch, dependency-light language model training system.

The package provides an end-to-end pipeline for training a small decoder-only
transformer from raw text: byte-level tokenization, dataset construction,
automatic differentiation, optimization, training loops, evaluation and text
generation.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
