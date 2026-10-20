"""Experiment 01 - Logistic Regression.

Grid-search ``C`` and penalty with and without 2nd degree polynomial
features, then write the best submission.

Run from the repo root with the d2l conda env:

    D:\\Soft\\miniconda3\\envs\\d2l\\python.exe experiments/exp01_logreg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.data import load_data, xy
from src.evaluate import SEED, cv_accuracy, grid_report, make_submission


def build(poly_degree: int, C: float, penalty: str) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if poly_degree > 1:
        steps.append(("poly", PolynomialFeatures(
            degree=poly_degree, interaction_only=False, include_bias=False)))
        # Scale again after expansion so penalties act on comparable magnitudes.
        steps.append(("scaler2", StandardScaler()))
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    steps.append(("clf", LogisticRegression(
        C=C, penalty=penalty, solver=solver, max_iter=5000, random_state=SEED)))
    return Pipeline(steps)


def main() -> None:
    train, _ = load_data()
    X, y = xy(train)

    # Reference: baseline kNN(5) under the same CV protocol.
    knn_cv = cv_accuracy(
        Pipeline([("scaler", StandardScaler()),
                  ("knn", KNeighborsClassifier(n_neighbors=5))]),
        X, y)
    print(f"[reference] baseline kNN(k=5) 5-fold acc = "
          f"{knn_cv['mean']:.4f} +/- {knn_cv['std']:.4f}")

    results = []
    for poly in (1, 2, 3):
        for penalty in ("l2", "l1"):
            for C in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0):
                est = build(poly, C, penalty)
                score = cv_accuracy(est, X, y)
                results.append({
                    "poly": poly, "penalty": penalty, "C": C,
                    "mean": score["mean"], "std": score["std"],
                })

    df = grid_report("LogReg", results)
    best = df.iloc[0]
    print(f"\n>>> best: poly={int(best.poly)} penalty={best.penalty} "
          f"C={best.C} -> {best['mean']:.4f} +/- {best['std']:.4f}")

    best_est = build(int(best.poly), float(best.C), best.penalty)
    # LogReg is currently the CV champion -> mirror to submission.csv.
    make_submission(best_est, "submission_logreg.csv", mirror_as_best=True)

    # Persist the grid for the doc.
    df.to_csv(ROOT / "experiments" / "exp01_logreg_grid.csv", index=False)


if __name__ == "__main__":
    main()
