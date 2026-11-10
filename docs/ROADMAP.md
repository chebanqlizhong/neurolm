# Roadmap

NeuroLM is an actively developed project. This document tracks where it is,
where it is going, and how you can extend it. It is updated as work progresses;
see [`DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md) for the chronological record.

## Guiding principles

1. **Readable over clever.** Every component should be understandable in one
   sitting.
2. **Dependency-light.** NumPy and PyYAML, nothing else required.
3. **Verified.** New behavior comes with tests; numerical claims are checked.
4. **Reproducible.** Fixed seeds, resolved configs and logged metrics per run.

## Status

Legend: ✅ done · 🚧 in progress · ⬜ planned

### Core

- ✅ Reverse-mode autodiff engine with gradient checks
- ✅ Byte-level tokenizer with exact UTF-8 round-trip
- ✅ Decoder-only transformer (pre-norm, causal attention, weight tying)
- ✅ AdamW with decoupled weight decay
- ✅ Cosine LR schedule with warmup
- ✅ Global-norm gradient clipping
- ✅ Training loop with validation, checkpointing, JSON history
- ✅ Temperature / top-k sampling
- ✅ CLI (`train`, `evaluate`, `generate`, `info`)
- ✅ Typed, validated YAML configuration
- ✅ pytest suite + GitHub Actions CI

### Near term

- ⬜ **Subword tokenizer.** Train a BPE vocabulary to shorten sequences.
- ⬜ **Learning-rate and gradient reporting.** Surface metrics per step in a
  machine-readable log (CSV/JSONL).
- ⬜ **Resume training.** Load optimizer state (Adam moments) from a checkpoint,
  not just weights.
- ⬜ **Mixed precision.** Keep master weights in float32, run matmuls in
  bfloat16-style reduced precision where beneficial.
- ⬜ **Attention improvements.** Grouped-query attention and rotary position
  embeddings.

### Medium term

- ⬜ **Multi-device training.** Data-parallel gradient averaging across
  processes.
- ⬜ **Instruction tuning.** Supervised fine-tuning on prompt/response pairs.
- ⬜ **Evaluation harness.** Held-out benchmarks beyond validation perplexity.
- ⬜ **Sampling strategies.** Nucleus (top-p), repetition penalty, beam search.
- ⬜ **Quantization.** int8 weight-only inference.

### Longer term

- ⬜ **Retrieval-augmented generation** hooks.
- ⬜ **Preference optimization** (DPO-style) on top of a fine-tuned model.
- ⬜ **Model cards** auto-generated from each run's config and metrics.

## How to contribute

1. Pick an item above (or propose one).
2. Add a config or extend an existing one under `configs/`.
3. Write tests first where practical — especially for new autodiff operations,
   where a finite-difference check is the bar.
4. Update [`DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md) and, if behavior is
   user-visible, this roadmap and the [changelog](../CHANGELOG.md).
5. Keep the diff focused. Small, reviewable changes land faster.

## Non-goals

- Competing with production frameworks on speed or scale.
- Supporting every architecture. NeuroLM targets small decoder-only language
  models that train on a single CPU or GPU.
