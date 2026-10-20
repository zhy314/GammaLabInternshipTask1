"""Experiment 04 - Ensemble (soft voting + stacking).

Combines the three tuned base models (LogReg, SVM, GBDT) to see whether
blending their decisions can beat the LogReg champion.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from src.data import load_data, xy
from src.evaluate import SEED, cv_accuracy, grid_report, make_submission


def logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler2", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0,
                                   solver="lbfgs", max_iter=5000,
                                   random_state=SEED)),
    ])


def svm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=10.0, probability=True, random_state=SEED)),
    ])


def gbdt_estimator() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.9, random_state=SEED)


def main() -> None:
    train, _ = load_data()
    X, y = xy(train)

    base = [
        ("logreg", logreg_pipeline()),
        ("svm", svm_pipeline()),
        ("gbdt", gbdt_estimator()),
    ]

    results = []

    # 1) Soft voting with a few weight schemes.
    weight_grid = [
        (1, 1, 1),
        (2, 1, 1),  # bias toward LogReg (strongest base)
        (3, 1, 1),
        (2, 2, 1),
        (3, 2, 1),
        (1, 0, 0),  # pure LogReg sanity check
    ]
    for w in weight_grid:
        est = VotingClassifier(estimators=base, voting="soft", weights=list(w))
        s = cv_accuracy(est, X, y)
        results.append({"strategy": f"soft w={w}", "mean": s["mean"], "std": s["std"]})

    # 2) Hard voting (majority).
    est = VotingClassifier(estimators=base, voting="hard")
    s = cv_accuracy(est, X, y)
    results.append({"strategy": "hard majority", "mean": s["mean"], "std": s["std"]})

    # 3) Stacking with LogReg meta-learner (passthrough off / on).
    for passthrough in (False, True):
        est = StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(C=1.0, max_iter=5000, random_state=SEED),
            cv=5, passthrough=passthrough, n_jobs=-1,
        )
        s = cv_accuracy(est, X, y)
        results.append({"strategy": f"stacking passthrough={passthrough}",
                        "mean": s["mean"], "std": s["std"]})

    df = grid_report("Ensemble", results)
    best = df.iloc[0]
    print(f"\n>>> best ensemble: {best.strategy} -> "
          f"{best['mean']:.4f} +/- {best['std']:.4f}")

    # Rebuild the chosen ensemble.
    strategy = best.strategy
    if strategy.startswith("soft w="):
        w = eval(strategy.replace("soft w=", ""))  # tuple of ints
        best_est = VotingClassifier(estimators=base, voting="soft", weights=list(w))
    elif strategy == "hard majority":
        best_est = VotingClassifier(estimators=base, voting="hard")
    else:
        passthrough = "True" in strategy
        best_est = StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(C=1.0, max_iter=5000, random_state=SEED),
            cv=5, passthrough=passthrough, n_jobs=-1,
        )

    # Promote only if it beats the LogReg champion.
    is_best = float(best["mean"]) > 0.9296
    make_submission(best_est, "submission_ensemble.csv", mirror_as_best=is_best)
    if not is_best:
        print("Ensemble did NOT beat LogReg -> submission.csv left unchanged.")

    df.to_csv(ROOT / "experiments" / "exp04_ensemble_grid.csv", index=False)


if __name__ == "__main__":
    main()
