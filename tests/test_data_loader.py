"""Smoke tests for src/data_loader.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_config, resolve_paths


def test_load_config_returns_dict() -> None:
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg and "baseline_xgb" in cfg


def test_resolve_paths_has_expected_attributes() -> None:
    cfg = load_config()
    paths = resolve_paths(cfg)
    for name in ["raw_dir", "parquet_dir", "train_numeric", "train_date", "test_numeric"]:
        assert hasattr(paths, name), f"missing path attribute: {name}"


def test_config_target_and_id_present() -> None:
    cfg = load_config()
    assert cfg["loading"]["target_col"] == "Response"
    assert cfg["loading"]["id_col"] == "Id"
