# Development Log

A chronological record of what changed in this project, why, and what was
verified. Newest entries go on top. Each entry is dated and links to the
relevant files.

The project began as a "hello world" demonstration and was repurposed into a
from-scratch language model training system. Both phases are recorded below.

---

## 2026-11-10 — NeuroLM v0.1.0: from-scratch language model training system

**Goal.** Turn the repository into a training system for a dedicated AI model,
with a clean file layout, documentation and a development log that can be kept
up to date as the project grows.

### What was built

| Area | Files | Notes |
|------|-------|-------|
| Autodiff | `src/neurolm/autograd.py` | Reverse-mode engine: arithmetic, matmul, reshape, transpose, sum/mean, indexing, exp/log, softmax, log-softmax, layer norm, GELU, embedding, cross-entropy |
| Config | `src/neurolm/config.py` | Typed dataclasses (`ModelConfig`, `DataConfig`, `TrainConfig`, `RunConfig`) loaded from YAML with validation |
| Tokenizer | `src/neurolm/tokenizer.py` | Byte-level tokenizer, `0..255` plus `<bos>`/`<eos>` |
| Data | `src/neurolm/data/dataset.py` | Single-array token stream, train/val split, random contiguous block sampling |
| Model | `src/neurolm/models/gpt.py`, `parameter.py` | Pre-norm decoder-only transformer, causal multi-head attention, GELU MLP, weight tying |
| Training | `src/neurolm/training/optimizer.py`, `trainer.py`, `checkpoint.py`, `evaluate.py` | AdamW, cosine LR with warmup, gradient clipping, train loop, `.npz` checkpoints, loss/perplexity, sampling |
| Interface | `src/neurolm/cli.py`, `__main__.py` | `train`, `evaluate`, `generate`, `info` subcommands |
| Configs | `configs/tiny.yaml`, `configs/small.yaml` | Tiny (seconds) and small (~90s CPU) runs |
| Corpus | `data/corpus.txt` | ~2.1 KB of explanatory prose about language modeling |
| Docs | `README.md`, `docs/*` | Overview, architecture, getting started, roadmap |
| Tooling | `Makefile`, `scripts/*`, `.github/workflows/ci.yml` | Convenience targets, plotting, train-and-sample, CI |
| Tests | `tests/*` | 36 tests covering autodiff, data, model, config, training |

### Verification

- **Gradient correctness.** A finite-difference check over a composite function
  (layer norm → GELU → matmul → softmax) agrees with the analytic gradients to
  < 2e-4 (`tests/test_autograd.py::test_gradient_check_composite`).
- **Causality.** Changing a future token does not alter earlier logits
  (`tests/test_model.py::test_causality_no_future_leakage`).
- **Learning.** Both configs reduce loss and perplexity on the bundled corpus.

`small` run (300 steps, ~90s CPU, 394,722 parameters):

| Step | Train loss | Val loss | Val perplexity |
|-----:|-----------:|---------:|---------------:|
| 0    | 5.557      | 5.524    | 250.7          |
| 50   | 2.704      | 2.722    | 15.2           |
| 100  | 2.449      | 2.532    | 12.6           |
| 299  | 1.970      | 2.699    | 14.9           |

- **Test suite.** `pytest` → 36 passed.

### Issues found and fixed during development

1. **Nested config not built.** `from __future__ import annotations` turns
   dataclass field types into strings, so `is_dataclass(field.type)` was always
   `False` and nested configs stayed as dicts. Fixed by resolving types with
   `typing.get_type_hints` in `config._build`.
2. **`batch_size` read from the wrong config.** The trainer looked for
   `train.batch_size`; it lives under `data`. Corrected to
   `self.config.data.batch_size`.
3. **Two incorrect tests.** A cross-entropy test used a 2-D logits tensor where a
   3-D `(batch, seq, vocab)` tensor was required, and an AdamW test asserted the
   wrong invariant. Both corrected.

### Notes / decisions

- **Byte-level tokenizer** chosen for zero-setup, no-OOV, exact round-trip;
  subword tokenization is on the [roadmap](docs/ROADMAP.md).
- **No framework dependency.** The point is transparency; NumPy only.
- **`small` config tuned for CPU** so a full run finishes in ~90s. An earlier
  heavier version (d_model 128, 4 layers, 600 steps) exceeded a 5-minute budget.

---

## 2026-11-10 — Project bootstrap (hello world)

**Goal.** Create a GitHub repository and a minimal hello program.

- Created the `hello-world` repository.
- Added `hello.py` (prints `Hello, World!`) and a short `README.md`.
- Initial commit pushed to `main`.

This phase is retained as the repository's first commit; the code was superseded
by the training system above. `hello.py` is kept at the repository root as a
minimal smoke test that the toolchain works.
