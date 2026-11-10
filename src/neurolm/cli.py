"""Command-line interface for the NeuroLM training system.

Subcommands
-----------
``train``      Train a model from a YAML config.
``evaluate``   Report loss and perplexity of a checkpoint.
``generate``   Sample text from a checkpoint.
``info``       Print the resolved configuration and model size.

The CLI has no third-party dependency beyond PyYAML/NumPy, so it can run in a
bare Python environment: ``python -m neurolm train --config configs/small.yaml``.
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
        "model: %d parameters (%s), vocab=%d, layers=%d, d_model=%d",
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
    logger.info("run '%s' -> %s", config.run_name, config.out_dir)

    train_data, val_data = TokenDataset.from_file(
        config.data.corpus_path,
        block_size=config.data.block_size,
        val_split=config.data.val_split,
        seed=config.data.seed,
    )
    logger.info(
        "corpus: %d train tokens, %d val tokens", train_data.tokens.size, val_data.tokens.size
    )

    model = _build_model(config)
    trainer = Trainer(config, model, train_data, val_data)
    save_config(config, trainer.out_dir / "config.yaml")

    def on_eval(train_metrics, val_metrics) -> None:
        logger.info(
            "step %5d | train %.4f | val %.4f | ppl %.2f | lr %.2e | |g| %.2f",
            train_metrics.step,
            train_metrics.loss,
            val_metrics.loss,
            perplexity(val_metrics.loss),
            train_metrics.lr,
            train_metrics.grad_norm,
        )

    history = trainer.fit(callback=on_eval)
    write_json(trainer.out_dir / "history.json", history.to_dict())
    logger.info("training complete; artifacts in %s", trainer.out_dir)
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
    logger.info("loss %.4f | perplexity %.2f", loss, perplexity(loss))
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
    logger.info("config: %s", args.config)
    for name, param in model.named_parameters():
        logger.info("  %-24s %s", name, tuple(param.shape))
    logger.info("total parameters: %d", model.numel())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurolm", description="Train a small language model from scratch.")
    parser.add_argument("--version", action="version", version="neurolm 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model")
    p_train.add_argument("--config", required=True, type=Path)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p_eval.add_argument("--config", required=True, type=Path)
    p_eval.add_argument("--checkpoint", required=True, type=Path)
    p_eval.add_argument("--batches", type=int, default=20)
    p_eval.set_defaults(func=cmd_evaluate)

    p_gen = sub.add_parser("generate", help="sample text from a checkpoint")
    p_gen.add_argument("--config", required=True, type=Path)
    p_gen.add_argument("--checkpoint", required=True, type=Path)
    p_gen.add_argument("--prompt", default="")
    p_gen.add_argument("--max-new-tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=0)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.set_defaults(func=cmd_generate)

    p_info = sub.add_parser("info", help="print config and model size")
    p_info.add_argument("--config", required=True, type=Path)
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
