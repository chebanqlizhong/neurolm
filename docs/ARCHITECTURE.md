# Architecture

This document explains how NeuroLM is put together and why each piece is
designed the way it is. It is meant to be read alongside the source; every
section points at the relevant file.

## Overview

NeuroLM is a small, layered system. Data flows in one direction, and each layer
has a single responsibility:

```
       raw text
          │
          ▼
   ┌─────────────┐   token ids
   │  tokenizer  │ ───────────────┐
   └─────────────┘                │
                                  ▼
                          ┌───────────────┐
                          │  TokenDataset │  samples (inputs, targets)
                          └───────────────┘
                                  │
                                  ▼
   ┌─────────────┐   logits  ┌───────────┐   loss
   │    GPT      │ ◄──────── │  Trainer  │ ────────► backward
   └─────────────┘           └───────────┘
        ▲                          │
        │ parameters               │ gradients
        │                          ▼
   ┌─────────────┐           ┌───────────┐
   │  Parameter  │ ◄──────── │   AdamW   │
   └─────────────┘           └───────────┘
```

## Layer 1 — Autodiff (`src/neurolm/autograd.py`)

The foundation is a reverse-mode automatic differentiation engine. A
[`Tensor`](../src/neurolm/autograd.py) wraps a NumPy array and records the
operation that produced it. Calling `backward()` on a scalar node:

1. builds a topological order of every tensor in the computation graph,
2. seeds the output gradient with ones,
3. walks the graph in reverse, calling each node's local `_backward`.

Each operation (add, multiply, matmul, reshape, sum, …) defines its own vector–
Jacobian product. Broadcasting is handled by `_unbroadcast`, which sums a
gradient back to the original shape. Composite operations (`softmax`,
`log_softmax`, `layer_norm`, `gelu`, `cross_entropy`, `embedding`) are written
as dedicated nodes with analytically derived backward passes for numerical
stability.

**Correctness.** `tests/test_autograd.py` checks gradients against finite
differences for a composite function that exercises layer norm, GELU, matmul
and softmax.

## Layer 2 — Tokenization (`src/neurolm/tokenizer.py`)

The `ByteTokenizer` maps text to the integers `0..255` and reserves `256`/`257`
for begin/end-of-sequence markers. A byte-level vocabulary:

- needs no training pass over the corpus,
- can never emit an unknown token,
- round-trips any UTF-8 string exactly.

The trade-off is sequence length: bytes are less compact than subword tokens, so
sequences are longer for the same text. For a small, readable system this is the
right default; subword tokenizers are on the [roadmap](ROADMAP.md).

## Layer 3 — Data (`src/neurolm/data/dataset.py`)

The corpus is tokenized once into a single `int64` array. `TokenDataset` splits
it into train/validation regions and samples random contiguous blocks:

- **inputs** are `tokens[i : i+block]`,
- **targets** are `tokens[i+1 : i+block+1]`.

The one-position shift is what makes the task "predict the next token". Sampling
is pure integer indexing, so the training loop never blocks on I/O.

## Layer 4 — Model (`src/neurolm/models/`)

`GPT` is a decoder-only transformer. For an input of shape `(batch, seq)`:

```
tok_emb[idx] + pos_emb[0..seq]        # embeddings
  └─ dropout
      └─ for each block:
           x = x + attn(ln1(x))       # causal self-attention
           x = x + mlp(ln2(x))        # feed-forward
  └─ ln_f
      └─ lm_head                        # (batch, seq, vocab) logits
```

Key design choices:

- **Pre-norm.** Layer normalization is applied *before* each sub-layer. This
  keeps gradients well-scaled in deep stacks and is the standard modern recipe.
- **Causal mask.** A fixed upper-triangular `-1e9` mask is added to attention
  scores so position *t* can only attend to positions `≤ t`. The mask is cached
  per sequence length.
- **Weight tying.** The output projection can share the token embedding matrix,
  which reduces parameters and often improves small models.

`Parameter` is a `Tensor` subclass that always requires gradients and carries a
stable name, so optimizers and checkpoints can refer to weights by name.

## Layer 5 — Training (`src/neurolm/training/`)

The `Trainer` owns the model, optimizer and datasets:

1. sample a batch,
2. forward pass → logits,
3. cross-entropy loss against targets,
4. `loss.backward()`,
5. global-norm gradient clipping,
6. `AdamW.step()`,
7. periodically evaluate and checkpoint.

**AdamW** uses decoupled weight decay, so regularization does not interfere with
the adaptive learning rate. The learning rate follows **linear warmup then
cosine decay** (`cosine_lr`). Validation loss is measured over several batches
and reported as perplexity (`exp(loss)`).

## Layer 6 — Interface (`src/neurolm/cli.py`)

The CLI exposes four subcommands — `train`, `evaluate`, `generate`, `info` — and
reads a validated YAML config (`src/neurolm/config.py`). Configs are parsed into
typed dataclasses at load time, so a typo in a config file fails immediately with
a clear message instead of surfacing deep inside training.

## Extension points

| To change… | Edit |
|------------|------|
| Model size/shape | `configs/*.yaml`, `ModelConfig` |
| Architecture (new layer type) | `src/neurolm/models/gpt.py` |
| Optimizer or schedule | `src/neurolm/training/optimizer.py` |
| Tokenization scheme | `src/neurolm/tokenizer.py` |
| New autodiff op | `src/neurolm/autograd.py` + a gradient-check test |
