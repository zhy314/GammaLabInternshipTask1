"""Experiment 02 - Support Vector Classifier (RBF + linear + poly).

Grid-search ``C`` / ``gamma`` / kernel, pick best by 5-fold CV accuracy,
and emit ``submissions/submission_svm.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import load_data, xy
from src.evaluate import SEED, cv_accuracy, grid_report, make_submission


def build(kernel: str, C: float, gamma, degree: int = 3) -> Pipeline:
    kwargs = dict(C=C, kernel=kernel, probability=True, random_state=SEED)
    if kernel in ("rbf", "poly", "sigmoid"):
        kwargs["gamma"] = gamma
    if kernel == "poly":
        kwargs["degree"] = degree
    return Pipeline([("scaler", StandardScaler()), ("clf", SVC(**kwargs))])


def main() -> None:
    train, _ = load_data()
    X, y = xy(train)

    results = []

    # Linear kernel - sanity check vs. LogReg.
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        est = build("linear", C, gamma="scale")
        s = cv_accuracy(est, X, y)
        results.append({"kernel": "linear", "C": C, "gamma": "-", "degree": "-",
                        "mean": s["mean"], "std": s["std"]})

    # RBF - the main workhorse for small, low-dim, non-linear data.
    for C in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0):
        for gamma in ("scale", "auto", 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
            est = build("rbf", C, gamma=gamma)
            s = cv_accuracy(est, X, y)
            results.append({"kernel": "rbf", "C": C, "gamma": gamma, "degree": "-",
                            "mean": s["mean"], "std": s["std"]})

    # Polynomial - compare against LogReg + PolynomialFeatures.
    for C in (0.1, 1.0, 10.0):
        for degree in (2, 3):
            for gamma in ("scale", 0.25, 1.0):
                est = build("poly", C, gamma=gamma, degree=degree)
                s = cv_accuracy(est, X, y)
                results.append({"kernel": "poly", "C": C, "gamma": gamma,
                                "degree": degree, "mean": s["mean"], "std": s["std"]})

    df = grid_report("SVM", results)
    best = df.iloc[0]
    print(f"\n>>> best: kernel={best.kernel} C={best.C} gamma={best.gamma} "
          f"degree={best.degree} -> {best['mean']:.4f} +/- {best['std']:.4f}")

    if best.kernel == "linear":
        gamma = "scale"
    elif best.gamma in ("scale", "auto"):
        gamma = best.gamma
    else:
        gamma = float(best.gamma)
    degree = 3 if best.degree == "-" else int(best.degree)
    best_est = build(best.kernel, float(best.C), gamma, degree=degree)
    make_submission(best_est, "submission_svm.csv")

    df.to_csv(ROOT / "experiments" / "exp02_svm_grid.csv", index=False)


if __name__ == "__main__":
    main()
