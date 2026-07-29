"""Thin wrapper so scripts can share train.load_train_frame without duplicating logic.

TODO: probably should move load_train_frame into src/ properly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_engineered_sample(cfg: dict, sample_n: int) -> tuple[pd.DataFrame, pd.Series]:
    from train import load_train_frame
    X, y, _ = load_train_frame(cfg, sample_n=sample_n)
    return X, y
