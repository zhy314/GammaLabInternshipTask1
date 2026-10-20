"""Experiment 03 - Gradient Boosted Decision Trees.

We try both sklearn's :class:`GradientBoostingClassifier` and LightGBM so we
can compare a slow, well-tested implementation with the fast industry
standard. Grid is kept small on purpose: with 156 rows we need to aggressively
regularise (shallow trees, small learning rate, early stopping).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

from src.data import load_data, xy
from src.evaluate import SEED, cv_accuracy, grid_report, make_submission


def main() -> None:
    train, _ = load_data()
    X, y = xy(train)

    results = []

    # --- sklearn GradientBoostingClassifier -----------------------------
    for n_estimators in (100, 200, 400):
        for max_depth in (2, 3, 4):
            for lr in (0.03, 0.05, 0.1):
                est = GradientBoostingClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=lr, subsample=0.9, random_state=SEED)
                s = cv_accuracy(est, X, y)
                results.append({
                    "impl": "sk-gb", "n_est": n_estimators, "max_depth": max_depth,
                    "lr": lr, "num_leaves": "-", "min_child": "-",
                    "mean": s["mean"], "std": s["std"],
                })

    # --- LightGBM --------------------------------------------------------
    for n_estimators in (200, 400, 800):
        for num_leaves in (7, 15, 31):
            for lr in (0.03, 0.05, 0.1):
                for min_child in (5, 10, 20):
                    est = lgb.LGBMClassifier(
                        n_estimators=n_estimators, num_leaves=num_leaves,
                        learning_rate=lr, min_child_samples=min_child,
                        subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                        reg_lambda=1.0, random_state=SEED, n_jobs=1, verbose=-1)
                    s = cv_accuracy(est, X, y)
                    results.append({
                        "impl": "lgbm", "n_est": n_estimators,
                        "max_depth": "-", "lr": lr, "num_leaves": num_leaves,
                        "min_child": min_child,
                        "mean": s["mean"], "std": s["std"],
                    })

    df = grid_report("GBDT", results)
    best = df.iloc[0]
    print(f"\n>>> best: {best.to_dict()}")

    if best.impl == "sk-gb":
        best_est = GradientBoostingClassifier(
            n_estimators=int(best.n_est), max_depth=int(best.max_depth),
            learning_rate=float(best.lr), subsample=0.9, random_state=SEED)
    else:
        best_est = lgb.LGBMClassifier(
            n_estimators=int(best.n_est), num_leaves=int(best.num_leaves),
            learning_rate=float(best.lr), min_child_samples=int(best.min_child),
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=SEED, n_jobs=1, verbose=-1)

    # Only promote to submission.csv if it actually beats LogReg (0.9296).
    is_best = float(best["mean"]) > 0.9296
    make_submission(best_est, "submission_gbdt.csv", mirror_as_best=is_best)
    if not is_best:
        print("GBDT did NOT beat LogReg -> submission.csv left unchanged.")

    df.to_csv(ROOT / "experiments" / "exp03_gbdt_grid.csv", index=False)


if __name__ == "__main__":
    main()
