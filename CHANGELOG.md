# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Subword (BPE) tokenizer.
- Resume training from optimizer state, not just weights.
- Top-p (nucleus) sampling and repetition penalty.
- Machine-readable per-step metrics (JSONL/CSV).

## [0.1.0] - 2026-11-10

The first release of NeuroLM: a from-scratch, dependency-light language model
training system.

### Added
- **Autodiff engine** (`neurolm.autograd`): reverse-mode automatic
  differentiation over a NumPy-backed `Tensor`, with arithmetic, matmul,
  reshape, transpose, reductions, indexing, `exp`/`log`, and composite nodes
  for `softmax`, `log_softmax`, `layer_norm`, `gelu`, `embedding` and
  `cross_entropy`.
- **Configuration** (`neurolm.config`): typed, validated YAML configs.
- **Tokenizer** (`neurolm.tokenizer`): byte-level tokenizer with exact UTF-8
  round-trip and `<bos>`/`<eos>` tokens.
- **Data pipeline** (`neurolm.data`): token-stream dataset with train/val split
  and random contiguous block sampling.
- **Model** (`neurolm.models`): decoder-only transformer with pre-norm residual
  blocks, causal multi-head attention, GELU feed-forward network and optional
  weight tying.
- **Training** (`neurolm.training`): AdamW optimizer with decoupled weight
  decay, cosine learning-rate schedule with warmup, global-norm gradient
  clipping, training loop with validation and checkpointing, loss/perplexity
  evaluation, and temperature/top-k text generation.
- **CLI** (`neurolm.cli`): `train`, `evaluate`, `generate`, `info` subcommands.
- **Configs**: `configs/tiny.yaml` and `configs/small.yaml`.
- **Corpus**: `data/corpus.txt`.
- **Tooling**: `Makefile`, `scripts/plot_history.py`,
  `scripts/train_and_sample.sh`, GitHub Actions CI.
- **Documentation**: `README.md`, `docs/ARCHITECTURE.md`,
  `docs/GETTING_STARTED.md`, `docs/ROADMAP.md`, `DEVELOPMENT_LOG.md`.
- **Tests**: 36 tests covering autodiff (including finite-difference gradient
  checks), data, model, configuration and training.

### Changed
- Repository repurposed from a hello-world demonstration into a language model
  training system. `hello.py` is retained as a minimal toolchain smoke test.

[Unreleased]: https://github.com/chebanqlizhong/neurolm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/chebanqlizhong/neurolm/releases/tag/v0.1.0
