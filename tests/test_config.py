"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurolm.config import ModelConfig, RunConfig, load_config, save_config, to_dict


def test_defaults_are_valid():
    config = RunConfig()
    assert config.model.vocab_size == 256
    assert config.train.max_steps > 0


def test_head_dim_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=10, n_heads=4)


def test_load_config_roundtrip(tmp_path: Path):
    original = RunConfig(run_name="unit")
    path = tmp_path / "cfg.yaml"
    save_config(original, path)
    loaded = load_config(path)
    assert loaded.run_name == "unit"
    assert to_dict(loaded) == to_dict(original)


def test_nested_override(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"run_name": "x", "model": {"d_model": 64, "n_heads": 8}}),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.d_model == 64
    assert config.model.n_heads == 8
    assert config.model.n_layers == 4  # default preserved


def test_unknown_key_rejected(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"model": {"bogus": 1}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_shipped_configs_are_valid():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    for cfg_path in config_dir.glob("*.yaml"):
        config = load_config(cfg_path)
        assert config.model.vocab_size > 0
        assert config.data.block_size <= config.model.max_seq_len
