# AGENTS.md

Repository-specific notes for AI agents and contributors. Keep this file current
when build commands, layout or conventions change.

## What this project is

NeuroLM: a from-scratch, dependency-light language model training system in pure
NumPy. No PyTorch/TensorFlow. Repurposed from an initial hello-world repo.

## Environment

- Python 3.10+ (developed on 3.13).
- Dependencies: `numpy`, `pyyaml` (runtime); `pytest`, `ruff` (dev).
- No GPU needed. Everything runs on CPU.

## Commands

```bash
pip install -e ".[dev]"        # install
python3 -m pytest -q           # run the test suite (36 tests)
python3 -m ruff check src tests # lint (CI runs this)
make train-tiny                # seconds, smoke test
make train-small               # ~90s on CPU
PYTHONPATH=src python3 -m neurolm <train|evaluate|generate|info> --config <cfg>
```

`PYTHONPATH=src` is required unless the package is installed with `pip install -e .`.

## Layout

- `src/neurolm/autograd.py` — reverse-mode autodiff engine (the foundation).
- `src/neurolm/{config,tokenizer,cli}.py` — config, byte tokenizer, CLI.
- `src/neurolm/data/` — token dataset and batching.
- `src/neurolm/models/` — `Parameter`, `GPT` transformer.
- `src/neurolm/training/` — optimizer, trainer, checkpoint, evaluate.
- `configs/` — YAML run configs; `data/` — corpus; `tests/` — pytest suite.

## Conventions

- Type-annotated Python, `from __future__ import annotations` at the top.
- Format/lint with ruff; CI fails on lint errors.
- New autodiff operations require a finite-difference gradient check in
  `tests/test_autograd.py`.
- Configs are typed dataclasses; unknown keys must raise.
- Prefer readable code over clever code; this project is meant to be learnable.

## Known gotchas

- `from __future__ import annotations` makes dataclass field types strings.
  `config._build` uses `typing.get_type_hints` to resolve nested dataclasses —
  don't switch back to `field.type`.
- `batch_size` lives under `data`, not `train`, in the config.
- Training artifacts (`runs/`, `*.npz`) are gitignored.

## Workflow for changes

1. Implement + add/adjust tests.
2. `python3 -m pytest -q` and `python3 -m ruff check src tests` must pass.
3. Update `DEVELOPMENT_LOG.md`; update `CHANGELOG.md` and `docs/ROADMAP.md` for
   user-visible changes.
