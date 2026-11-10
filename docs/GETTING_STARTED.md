# Getting started

This guide takes you from a clean checkout to a trained model and sampled text.

## 1. Install

NeuroLM needs Python 3.10+ and two runtime dependencies (NumPy, PyYAML).

```bash
git clone https://github.com/chebanqlizhong/neurolm.git neurolm
cd neurolm
pip install -e ".[dev]"     # "dev" adds pytest and ruff
```

If you prefer not to install the package, you can run everything from the source
tree by putting `src/` on the path:

```bash
export PYTHONPATH=src
```

## 2. Train

The fastest way to confirm everything works is the `tiny` config, which finishes
in seconds:

```bash
make train-tiny
```

For a model that actually learns something recognizable, use `small` (about 90
seconds on a laptop CPU):

```bash
make train-small
```

You will see a log line each time validation runs:

```
step     0 | train 5.5573 | val 5.5244 | ppl 250.73 | lr 7.50e-05 | |g| 1.77
step    50 | train 2.7045 | val 2.7219 | ppl 15.21  | lr 2.99e-03 | |g| 0.58
step   299 | train 1.9704 | val 2.6986 | ppl 14.86  | lr 3.00e-04 | |g| 0.99
```

Artifacts land in `runs/<run_name>/`:

```
runs/small/
├── config.yaml     # the resolved configuration
├── history.json    # per-step train and validation metrics
├── step_150.npz    # periodic checkpoint
└── final.npz       # final weights
```

## 3. Evaluate

Measure loss and perplexity of a checkpoint:

```bash
python -m neurolm evaluate \
  --config configs/small.yaml \
  --checkpoint runs/small/final.npz \
  --batches 20
```

## 4. Generate

Sample text from a checkpoint:

```bash
python -m neurolm generate \
  --config configs/small.yaml \
  --checkpoint runs/small/final.npz \
  --prompt "The transformer " \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

Sampling controls:

| Flag | Effect |
|------|--------|
| `--temperature` | Lower (< 1) is conservative, higher (> 1) is more random |
| `--top-k` | Restrict sampling to the `k` most likely tokens (0 = disabled) |
| `--seed` | Makes sampling reproducible |

## 5. Plot the loss curve

```bash
python scripts/plot_history.py runs/small/history.json --out runs/small/loss.png
```

`matplotlib` is optional; without it the script prints an ASCII curve.

## 6. Configure your own run

Copy a config and edit it:

```bash
cp configs/small.yaml configs/my_run.yaml
```

The important knobs:

```yaml
run_name: my_run        # names the output directory
model:
  d_model: 128          # embedding width (must divide by n_heads)
  n_heads: 4            # attention heads
  n_layers: 6           # transformer blocks
  max_seq_len: 128      # maximum context length
data:
  corpus_path: data/corpus.txt
  block_size: 128       # tokens per training example (<= max_seq_len)
  batch_size: 16
train:
  max_steps: 1000
  learning_rate: 3.0e-3
```

Then:

```bash
python -m neurolm train --config configs/my_run.yaml
```

## 7. Use your own corpus

Point `data.corpus_path` at any UTF-8 text file. Bigger and more varied is
better. A rough rule: you want at least a few hundred thousand characters for
the model to learn anything beyond spelling.

## 8. Run the tests

```bash
make test      # pytest
make lint      # ruff
```

## Troubleshooting

**"corpus too small"** — the corpus must be longer than `block_size + 2` tokens.
Lower `block_size` or add more text.

**"d_model must be divisible by n_heads"** — pick a `d_model` that divides
evenly by `n_heads` (e.g. 128 and 4).

**Training is slow** — reduce `d_model`, `n_layers`, `batch_size` and
`max_seq_len`. The `tiny` config is the floor.

**Sampled text is gibberish** — expected for a very small model on a small
corpus. Train longer, use a larger model, or add more data. Lower the
temperature to make the output more conservative.
