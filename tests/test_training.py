"""Tests for the optimizer, schedule, checkpointing and training loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurolm.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from neurolm.data.dataset import TokenDataset
from neurolm.models.gpt import GPT
from neurolm.models.parameter import Parameter
from neurolm.tokenizer import ByteTokenizer
from neurolm.training.checkpoint import load_checkpoint, save_checkpoint
from neurolm.training.evaluate import generate, perplexity
from neurolm.training.optimizer import AdamW, AdamWConfig, clip_gradients, cosine_lr
from neurolm.training.trainer import Trainer


def test_cosine_schedule_shape():
    kwargs = dict(max_steps=100, lr=1.0, min_lr=0.1, warmup_steps=10)
    assert cosine_lr(0, **kwargs) < cosine_lr(9, **kwargs)  # warmup increases
    peak = cosine_lr(10, **kwargs)
    assert peak == pytest.approx(1.0, abs=0.05)
    assert cosine_lr(99, **kwargs) == pytest.approx(0.1, abs=0.02)
    assert cosine_lr(1000, **kwargs) == 0.1


def test_adamw_step_follows_negative_gradient():
    param = Parameter(np.array([1.0, -2.0, 3.0], dtype=np.float32), name="p")
    opt = AdamW([param], AdamWConfig(lr=0.1, weight_decay=0.0))
    before = param.data.copy()
    for _ in range(50):
        param.grad = np.ones_like(param.data)
        opt.step()
    # every coordinate must move in the negative gradient direction
    assert np.all(param.data < before)


def test_adamw_weight_decay_shrinks():
    param = Parameter(np.array([5.0], dtype=np.float32), name="p")
    opt = AdamW([param], AdamWConfig(lr=0.01, weight_decay=0.5))
    for _ in range(20):
        param.grad = np.zeros_like(param.data)
        opt.step()
    assert param.data[0] < 5.0


def test_clip_gradients_scales_norm():
    params = [Parameter(np.zeros(3, dtype=np.float32), name=f"p{i}") for i in range(2)]
    for p in params:
        p.grad = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    norm = clip_gradients(params, max_norm=1.0)
    assert norm == pytest.approx(5.0 * np.sqrt(2), rel=1e-5)
    total = np.sqrt(sum(float(np.sum(p.grad**2)) for p in params))
    assert total <= 1.0 + 1e-5


def test_checkpoint_roundtrip(tmp_path: Path, tiny_model_config):
    model = GPT(tiny_model_config, seed=0)
    path = save_checkpoint(tmp_path / "ckpt.npz", model, step=42)
    restored = GPT(tiny_model_config, seed=99)
    meta = load_checkpoint(path, restored)
    assert meta["step"] == 42
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), restored.named_parameters(), strict=True):
        assert n1 == n2
        assert np.allclose(p1.data, p2.data)


def _tiny_run_config(corpus_path: Path) -> RunConfig:
    return RunConfig(
        run_name="test_run",
        out_dir="runs",
        checkpoint_interval=5,
        model=ModelConfig(
            vocab_size=258, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=32, dropout=0.0
        ),
        data=DataConfig(
            corpus_path=str(corpus_path), block_size=32, batch_size=4, val_split=0.1
        ),
        train=TrainConfig(max_steps=10, eval_interval=5, eval_batches=2, learning_rate=5e-3),
    )


def test_trainer_end_to_end(tmp_path: Path, corpus_file: Path):
    config = _tiny_run_config(corpus_file)
    config.out_dir = str(tmp_path)
    train, val = TokenDataset.from_file(corpus_file, block_size=32, val_split=0.1)
    model = GPT(config.model, seed=0)
    trainer = Trainer(config, model, train, val)
    history = trainer.fit()
    assert len(history.train) == config.train.max_steps
    assert (trainer.out_dir / "final.npz").exists()
    # validation loss should be finite and improve over the run
    assert np.isfinite(history.val[0].loss)
    assert history.val[-1].loss < history.val[0].loss


def test_perplexity_matches_exp():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(np.log(10)) == pytest.approx(10.0)


def test_generate_returns_string_and_is_deterministic(tiny_model_config):
    model = GPT(tiny_model_config, seed=0)
    tok = ByteTokenizer()
    a = generate(model, tok, "The ", max_new_tokens=20, seed=3)
    b = generate(model, tok, "The ", max_new_tokens=20, seed=3)
    assert isinstance(a, str)
    assert a == b
    assert len(a) >= len("The ")
