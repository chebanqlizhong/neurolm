"""A decoder-only transformer language model.

The architecture follows the now-standard pre-norm GPT recipe:

    token embedding + position embedding
      -> N x [ LayerNorm -> causal self-attention -> residual
               LayerNorm -> feed-forward          -> residual ]
      -> LayerNorm -> linear language-model head

Every operation is expressed with :mod:`neurolm.autograd`, so calling
``logits.backward()`` produces gradients for all parameters.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..autograd import Tensor, _accum, embedding, gelu, layer_norm, softmax
from .parameter import Parameter, ParameterDict, init_normal, init_ones, init_zeros


def _dropout(x: Tensor, p: float, training: bool, rng: np.random.Generator) -> Tensor:
    """Inverted dropout as a differentiable tensor op."""
    if not training or p <= 0.0:
        return x
    keep = 1.0 - p
    mask = (rng.random(x.shape) < keep).astype(np.float32) / keep
    out = Tensor(x.data * mask, requires_grad=x.requires_grad, _children=(x,), _op="dropout")

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad * mask)

    out._backward = _backward
    return out


class CausalSelfAttention:
    """Multi-head scaled dot-product attention with a causal mask."""

    def __init__(self, config, rng: np.random.Generator, prefix: str) -> None:
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.attn_dropout = config.dropout
        std = 0.02
        self.params = ParameterDict(
            {
                "Wq": init_normal((config.d_model, config.d_model), f"{prefix}.Wq", std, rng),
                "Wk": init_normal((config.d_model, config.d_model), f"{prefix}.Wk", std, rng),
                "Wv": init_normal((config.d_model, config.d_model), f"{prefix}.Wv", std, rng),
                "Wo": init_normal((config.d_model, config.d_model), f"{prefix}.Wo", std, rng),
                "bq": init_zeros((config.d_model,), f"{prefix}.bq"),
                "bk": init_zeros((config.d_model,), f"{prefix}.bk"),
                "bv": init_zeros((config.d_model,), f"{prefix}.bv"),
                "bo": init_zeros((config.d_model,), f"{prefix}.bo"),
            }
        )

    def _split_heads(self, x: Tensor, batch: int, seq: int) -> Tensor:
        return x.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: Tensor, mask: np.ndarray, training: bool, rng) -> Tensor:
        batch, seq, _ = x.shape
        p = self.params
        q = (x.matmul(p["Wq"]) + p["bq"])
        k = (x.matmul(p["Wk"]) + p["bk"])
        v = (x.matmul(p["Wv"]) + p["bv"])

        q = self._split_heads(q, batch, seq)
        k = self._split_heads(k, batch, seq)
        v = self._split_heads(v, batch, seq)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q.matmul(k.transpose(0, 1, 3, 2)) * scale) + mask
        weights = softmax(scores, axis=-1)
        weights = _dropout(weights, self.attn_dropout, training, rng)

        context = weights.matmul(v)  # (B, H, T, hd)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        return context.matmul(p["Wo"]) + p["bo"]


class FeedForward:
    """Position-wise MLP with GELU activation."""

    def __init__(self, config, rng: np.random.Generator, prefix: str) -> None:
        std = 0.02
        self.params = ParameterDict(
            {
                "W1": init_normal((config.d_model, config.d_ff), f"{prefix}.W1", std, rng),
                "b1": init_zeros((config.d_ff,), f"{prefix}.b1"),
                "W2": init_normal((config.d_ff, config.d_model), f"{prefix}.W2", std, rng),
                "b2": init_zeros((config.d_model,), f"{prefix}.b2"),
            }
        )

    def __call__(self, x: Tensor) -> Tensor:
        p = self.params
        hidden = gelu(x.matmul(p["W1"]) + p["b1"])
        return hidden.matmul(p["W2"]) + p["b2"]


class Block:
    """A pre-norm transformer block."""

    def __init__(self, config, rng: np.random.Generator, index: int) -> None:
        prefix = f"blocks.{index}"
        self.ln1_g = init_ones((config.d_model,), f"{prefix}.ln1.g")
        self.ln1_b = init_zeros((config.d_model,), f"{prefix}.ln1.b")
        self.ln2_g = init_ones((config.d_model,), f"{prefix}.ln2.g")
        self.ln2_b = init_zeros((config.d_model,), f"{prefix}.ln2.b")
        self.attn = CausalSelfAttention(config, rng, f"{prefix}.attn")
        self.mlp = FeedForward(config, rng, f"{prefix}.mlp")
        self.dropout = config.dropout

    def __call__(self, x: Tensor, mask: np.ndarray, training: bool, rng) -> Tensor:
        h = layer_norm(x, self.ln1_g, self.ln1_b)
        h = self.attn(h, mask, training, rng)
        x = x + _dropout(h, self.dropout, training, rng)

        h = layer_norm(x, self.ln2_g, self.ln2_b)
        h = self.mlp(h)
        x = x + _dropout(h, self.dropout, training, rng)
        return x

    def parameters(self) -> Iterator[Parameter]:
        yield from (self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b)
        yield from self.attn.params.parameters()
        yield from self.mlp.params.parameters()


class GPT:
    """A small decoder-only transformer language model."""

    def __init__(self, config, seed: int = 0) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        std = 0.02
        residual_std = std / np.sqrt(2 * config.n_layers)

        self.tok_emb = init_normal(
            (config.vocab_size, config.d_model), "tok_emb", std, rng
        )
        self.pos_emb = init_normal(
            (config.max_seq_len, config.d_model), "pos_emb", std, rng
        )
        self.blocks = [Block(config, rng, i) for i in range(config.n_layers)]
        self.ln_f_g = init_ones((config.d_model,), "ln_f.g")
        self.ln_f_b = init_zeros((config.d_model,), "ln_f.b")
        self.lm_head = init_normal(
            (config.d_model, config.vocab_size), "lm_head", residual_std, rng
        )
        self.lm_head_b = init_zeros((config.vocab_size,), "lm_head.b")

        self._mask_cache: dict[int, np.ndarray] = {}

    # -- parameter management ---------------------------------------------
    def parameters(self) -> list[Parameter]:
        params = [self.tok_emb, self.pos_emb]
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend([self.ln_f_g, self.ln_f_b, self.lm_head, self.lm_head_b])
        return params

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        for param in self.parameters():
            yield param.name, param

    def numel(self) -> int:
        return sum(p.numel for p in self.parameters())

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    # -- forward -----------------------------------------------------------
    def _causal_mask(self, seq: int) -> np.ndarray:
        if seq not in self._mask_cache:
            mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
            self._mask_cache[seq] = mask
        return self._mask_cache[seq]

    def forward(self, idx: np.ndarray, training: bool = False, rng=None) -> Tensor:
        """Return logits of shape ``(batch, seq, vocab)``."""
        if rng is None:
            rng = np.random.default_rng(0)
        idx = np.asarray(idx)
        batch, seq = idx.shape
        if seq > self.config.max_seq_len:
            raise ValueError(
                f"序列长度 {seq} 超过了 max_seq_len {self.config.max_seq_len}"
            )

        positions = np.arange(seq)
        tok = embedding(self.tok_emb, idx)
        pos = embedding(self.pos_emb, positions)
        x = tok + pos
        x = _dropout(x, self.config.dropout, training, rng)

        mask = self._causal_mask(seq)
        for block in self.blocks:
            x = block(x, mask, training, rng)

        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x.matmul(self.lm_head) + self.lm_head_b
        return logits


__all__ = ["GPT", "Block", "CausalSelfAttention", "FeedForward"]
