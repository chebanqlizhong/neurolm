#!/usr/bin/env python3
"""从一次运行的 ``history.json`` 绘制训练与验证损失曲线。

用法::

    python scripts/plot_history.py runs/small/history.json --out runs/small/loss.png

需要 ``matplotlib``。若未安装，脚本会打印 ASCII 曲线，因此在极简环境中依然可用。
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
    parser.add_argument("history", type=Path, help="history.json 的路径")
    parser.add_argument("--out", type=Path, default=None, help="输出 PNG 路径")
    args = parser.parse_args()

    payload = json.loads(args.history.read_text(encoding="utf-8"))
    train_loss = [m["loss"] for m in payload["train"]]
    val_loss = [m["loss"] for m in payload["val"]]
    val_steps = [m["step"] for m in payload["val"]]

    print(f"步数: {len(train_loss)}")
    print(f"训练损失: {train_loss[0]:.4f} -> {train_loss[-1]:.4f}")
    print(f"验证损失: {val_loss[0]:.4f} -> {val_loss[-1]:.4f}")
    print("训练 " + _ascii_curve(train_loss))
    print("验证 " + _ascii_curve(val_loss))

    if args.out is None:
        return 0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib；已跳过 PNG 输出")
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
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
