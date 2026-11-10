# NeuroLM

**A from-scratch language model training system in pure NumPy.**

NeuroLM trains a small decoder-only transformer end to end: tokenization,
dataset construction, automatic differentiation, the optimizer, the training
loop, evaluation and text generation. It has no deep-learning framework
dependency — only NumPy and PyYAML — so every part of the stack is readable and
hackable.

> This repository is under active development. See
> [`docs/ROADMAP.md`](docs/ROADMAP.md) for what is planned next and
> [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) for the running history of changes.

---

## Why from scratch?

Most tutorials hide the interesting parts behind a framework. NeuroLM keeps
them visible:

- **A real autodiff engine.** [`src/neurolm/autograd.py`](src/neurolm/autograd.py)
  implements reverse-mode automatic differentiation over a small tensor library.
  Gradients are verified against numerical differentiation in the test suite.
- **A real transformer.** Multi-head causal attention, pre-norm residual blocks
  and a feed-forward network, all expressed with the autodiff primitives.
- **A real training loop.** AdamW with decoupled weight decay, cosine learning
  rate schedule with warmup, global gradient clipping, periodic validation and
  checkpointing.

## Features

| Area | What you get |
|------|--------------|
| Tokenization | Byte-level tokenizer: no vocabulary training, no out-of-vocabulary tokens, exact UTF-8 round-trip |
| Model | Decoder-only transformer, configurable depth/width/heads |
| Autodiff | Reverse-mode engine with broadcasting, matmul, layer norm, GELU, softmax, cross-entropy |
| Optimization | AdamW, cosine LR schedule, gradient clipping |
| Training | Batched sampling, train/val split, evaluation, checkpointing, JSON history |
| Inference | Temperature and top-k sampling from any checkpoint |
| Tooling | CLI, YAML configs, Makefile, pytest suite, GitHub Actions CI |

## Quick start

```bash
# 1. Install (only NumPy + PyYAML)
pip install -e ".[dev]"

# 2. Train the tiny config — finishes in seconds on CPU
make train-tiny

# 3. Train the small config (~90s on a laptop CPU)
make train-small

# 4. Sample from the trained model
make generate PROMPT="The transformer "
```

Or use the CLI directly:

```bash
export PYTHONPATH=src
python -m neurolm train     --config configs/small.yaml
python -m neurolm evaluate  --config configs/small.yaml --checkpoint runs/small/final.npz
python -m neurolm generate  --config configs/small.yaml --checkpoint runs/small/final.npz \
                            --prompt "The model " --max-new-tokens 200 --temperature 0.8
```

## Results

Training the `small` config on the bundled corpus (`data/corpus.txt`):

| Step | Train loss | Val loss | Val perplexity |
|-----:|-----------:|---------:|---------------:|
| 0    | 5.557      | 5.524    | 250.7          |
| 50   | 2.704      | 2.722    | 15.2           |
| 100  | 2.449      | 2.532    | 12.6           |
| 299  | 1.970      | 2.699    | 14.9           |

Perplexity falls from ~250 (random) to ~13, and sampled text shows emerging
English-like structure. Reproduce with `make train-small`.

## Project layout

```
.
├── configs/              # YAML run configurations (tiny, small)
├── data/                 # Training corpus
├── docs/                 # Architecture, getting started, roadmap
├── scripts/              # Helper scripts (plotting, train-and-sample)
├── src/neurolm/
│   ├── autograd.py       # Reverse-mode autodiff engine
│   ├── config.py         # Typed, validated configuration
│   ├── tokenizer.py      # Byte-level tokenizer
│   ├── cli.py            # Command-line interface
│   ├── data/             # Dataset and batching
│   ├── models/           # Transformer (GPT), parameters
│   └── training/         # Optimizer, trainer, checkpoints, evaluation
└── tests/                # pytest suite (36 tests)
```

## Documentation

- [Getting started](docs/GETTING_STARTED.md) — install, train, sample, tune.
- [Architecture](docs/ARCHITECTURE.md) — how the pieces fit together.
- [Roadmap](docs/ROADMAP.md) — planned work and how to extend the system.
- [Development log](DEVELOPMENT_LOG.md) — chronological record of changes.
- [Changelog](CHANGELOG.md) — release-oriented summary.

## Requirements

- Python 3.10+
- NumPy ≥ 1.24, PyYAML ≥ 6.0
- No GPU required (runs on CPU)

## License

MIT — see [LICENSE](LICENSE).
