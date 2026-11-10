"""NeuroLM 训练系统的命令行接口。

子命令
------
``train``      根据 YAML 配置训练模型。
``evaluate``   报告检查点的损失与困惑度。
``generate``   从检查点采样文本。
``info``       打印解析后的配置与模型规模。

除 PyYAML/NumPy 外，命令行不依赖任何第三方库，因此可在裸 Python 环境中运行：
``python -m neurolm train --config configs/small.yaml``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import RunConfig, load_config, save_config
from .data.dataset import TokenDataset
from .models.gpt import GPT
from .tokenizer import ByteTokenizer
from .training.checkpoint import load_checkpoint
from .training.evaluate import evaluate_loss, generate, perplexity
from .training.trainer import Trainer
from .utils import get_logger, humanize_number, set_seed, write_json

logger = get_logger("neurolm.cli")


def _build_model(config: RunConfig) -> GPT:
    model = GPT(config.model, seed=config.train.seed)
    logger.info(
        "模型：%d 个参数 (%s)，词表=%d，层数=%d，d_model=%d",
        model.numel(),
        humanize_number(model.numel()),
        config.model.vocab_size,
        config.model.n_layers,
        config.model.d_model,
    )
    return model


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    set_seed(config.train.seed)
    logger.info("运行 '%s' -> %s", config.run_name, config.out_dir)

    train_data, val_data = TokenDataset.from_file(
        config.data.corpus_path,
        block_size=config.data.block_size,
        val_split=config.data.val_split,
        seed=config.data.seed,
    )
    logger.info(
        "语料：训练 %d 个 token，验证 %d 个 token",
        train_data.tokens.size,
        val_data.tokens.size,
    )

    model = _build_model(config)
    trainer = Trainer(config, model, train_data, val_data)
    save_config(config, trainer.out_dir / "config.yaml")

    def on_eval(train_metrics, val_metrics) -> None:
        logger.info(
            "步数 %5d | 训练 %.4f | 验证 %.4f | 困惑度 %.2f | 学习率 %.2e | |梯度| %.2f",
            train_metrics.step,
            train_metrics.loss,
            val_metrics.loss,
            perplexity(val_metrics.loss),
            train_metrics.lr,
            train_metrics.grad_norm,
        )

    history = trainer.fit(callback=on_eval)
    write_json(trainer.out_dir / "history.json", history.to_dict())
    logger.info("训练完成；产物位于 %s", trainer.out_dir)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model = _build_model(config)
    load_checkpoint(args.checkpoint, model)
    _, val_data = TokenDataset.from_file(
        config.data.corpus_path,
        block_size=config.data.block_size,
        val_split=config.data.val_split,
        seed=config.data.seed,
    )
    loss = evaluate_loss(model, val_data, config.data.batch_size, args.batches)
    logger.info("损失 %.4f | 困惑度 %.2f", loss, perplexity(loss))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model = _build_model(config)
    load_checkpoint(args.checkpoint, model)
    tokenizer = ByteTokenizer()
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model = _build_model(config)
    logger.info("配置：%s", args.config)
    for name, param in model.named_parameters():
        logger.info("  %-24s %s", name, tuple(param.shape))
    logger.info("参数总数：%d", model.numel())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurolm", description="从零训练一个小型语言模型。")
    parser.add_argument("--version", action="version", version="neurolm 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--config", required=True, type=Path, help="YAML 配置文件路径")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="评估检查点")
    p_eval.add_argument("--config", required=True, type=Path, help="YAML 配置文件路径")
    p_eval.add_argument("--checkpoint", required=True, type=Path, help="检查点 .npz 路径")
    p_eval.add_argument("--batches", type=int, default=20, help="评估使用的批次数")
    p_eval.set_defaults(func=cmd_evaluate)

    p_gen = sub.add_parser("generate", help="从检查点采样文本")
    p_gen.add_argument("--config", required=True, type=Path, help="YAML 配置文件路径")
    p_gen.add_argument("--checkpoint", required=True, type=Path, help="检查点 .npz 路径")
    p_gen.add_argument("--prompt", default="", help="提示文本")
    p_gen.add_argument("--max-new-tokens", type=int, default=200, help="最多生成多少 token")
    p_gen.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    p_gen.add_argument("--top-k", type=int, default=0, help="top-k 采样（0 表示关闭）")
    p_gen.add_argument("--seed", type=int, default=0, help="随机种子")
    p_gen.set_defaults(func=cmd_generate)

    p_info = sub.add_parser("info", help="打印配置与模型规模")
    p_info.add_argument("--config", required=True, type=Path, help="YAML 配置文件路径")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
