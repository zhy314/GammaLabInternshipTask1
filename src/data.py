"""Shared data loader for the Task1 pipeline.

All experiments import ``load_data`` so that paths, feature names and
dtypes stay consistent across models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES = ["f1", "f2", "f3", "f4"]
TARGET = "target"
ID = "id"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) read from ``data/``."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def xy(train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split a training DataFrame into feature matrix / label vector."""
    return train[FEATURES].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=int)


def test_x(test: pd.DataFrame) -> np.ndarray:
    return test[FEATURES].to_numpy(dtype=float)
