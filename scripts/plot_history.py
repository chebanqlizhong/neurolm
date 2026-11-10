#!/usr/bin/env python3
"""Plot the training and validation loss curves from a run's ``history.json``.

Usage::

    python scripts/plot_history.py runs/small/history.json --out runs/small/loss.png

Requires ``matplotlib``. If it is not installed the script prints an ASCII
sparkline instead, so it stays useful in minimal environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ascii_curve(values: list[float], width: int = 60) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    return "".join(blocks[min(len(blocks) - 1, int((v - lo) / span * (len(blocks) - 1)))] for v in sampled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="path to history.json")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    args = parser.parse_args()

    payload = json.loads(args.history.read_text(encoding="utf-8"))
    train_loss = [m["loss"] for m in payload["train"]]
    val_loss = [m["loss"] for m in payload["val"]]
    val_steps = [m["step"] for m in payload["val"]]

    print(f"steps: {len(train_loss)}")
    print(f"train loss: {train_loss[0]:.4f} -> {train_loss[-1]:.4f}")
    print(f"val   loss: {val_loss[0]:.4f} -> {val_loss[-1]:.4f}")
    print("train " + _ascii_curve(train_loss))
    print("val   " + _ascii_curve(val_loss))

    if args.out is None:
        return 0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped PNG output")
        return 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, label="train", linewidth=1.2)
    ax.plot(val_steps, val_loss, label="val", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("NeuroLM training")
    ax.legend()
    ax.grid(alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
