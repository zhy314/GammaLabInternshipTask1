"""Evaluation and submission helpers shared by every experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import FEATURES, ID, load_data, test_x, xy

SEED = 42
N_SPLITS = 5
SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent


def cv_accuracy(estimator, X: np.ndarray, y: np.ndarray,
                n_splits: int = N_SPLITS, seed: int = SEED) -> Dict[str, float]:
    """Return mean/std 5-fold stratified accuracy."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}


def grid_report(name: str, results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Pretty-print a grid-search summary and return it as a DataFrame."""
    df = pd.DataFrame(results).sort_values("mean", ascending=False).reset_index(drop=True)
    print(f"\n=== {name} grid search (top 10) ===")
    print(df.head(10).to_string(index=False))
    return df


def make_submission(estimator, filename: str, *, retrain: bool = True) -> Path:
    """Fit on the full training set and write ``submissions/<filename>``."""
    train, test = load_data()
    X, y = xy(train)
    if retrain:
        estimator.fit(X, y)
    preds = estimator.predict(test_x(test)).astype(int)
    out = pd.DataFrame({ID: test[ID].values, "prediction": preds})
    path = SUBMISSIONS_DIR / filename
    out.to_csv(path, index=False)
    # Mirror the latest/best to repo root for one-click submission.
    out.to_csv(ROOT / "submission.csv", index=False)
    print(f"wrote {path} (also mirrored to submission.csv)")
    return path
