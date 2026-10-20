# 05 - Ensemble (voting + stacking)

## Why ensemble

The three previously tuned models encode different inductive biases:

* **LogReg (poly=2, L2, C=1)** – smooth quadratic linear boundary.
* **SVM (linear, C=10)** – max-margin linear boundary.
* **GBDT (sklearn GB, 400 trees, depth=4, lr=0.03)** – piecewise constant,
  interaction-aware.

When base models err on *different* samples, averaging / voting their outputs
reduces the variance of the combined classifier. With only 156 training rows,
variance reduction is often more valuable than squeezing out a slightly
better single-model score.

## Strategies compared

Protocol: `StratifiedKFold(5, shuffle=True, random_state=42)` on the training
set, same as every other experiment.

| Strategy | 5-fold CV mean | CV std |
|---|---:|---:|
| **Hard majority vote (LogReg + SVM + GBDT)** | **0.9363** | 0.0445 |
| Soft vote weights (1, 0, 0) – LogReg only | 0.9296 | 0.0235 |
| Soft vote weights (3, 1, 1) | 0.9236 | 0.0548 |
| Soft vote weights (3, 2, 1) | 0.9234 | 0.0377 |
| Soft vote weights (2, 1, 1) | 0.9171 | 0.0585 |
| Stacking (meta=LogReg, passthrough=True) | 0.9169 | 0.0429 |
| Soft vote weights (2, 2, 1) | 0.9169 | 0.0429 |
| Stacking (meta=LogReg, passthrough=False) | 0.9042 | 0.0442 |
| Soft vote weights (1, 1, 1) | 0.8978 | 0.0505 |

Full table: `experiments/exp04_ensemble_grid.csv`.

## Observations

* **Hard voting wins** because the three base learners disagree on different
  samples. When at least two of them get a point right, the majority rule
  recovers it, even on inputs where LogReg alone fails.
* **Soft voting underperforms hard voting** here. GBDT emits poorly
  calibrated probabilities (its CV mean is only 0.898), so averaging in its
  high-confidence wrong probabilities actively hurts.
* **Stacking** is slightly worse than the LogReg baseline because the meta
  model has very little data (156 fold-OOF rows) to learn weights from and
  can itself overfit.
* **Variance trade-off**: hard voting pays ~2× the std of pure LogReg
  (0.0445 vs 0.0235). The mean gain (+0.7 pt) is real but small; on the
  hidden test set the single LogReg model could still beat it on a given
  draw. We follow the project rules — pick the best CV mean — and promote
  the hard-vote ensemble to `submission.csv`, but keep the LogReg submission
  in `submissions/submission_logreg.csv` as a safety net.

## Final leaderboard (internal, by 5-fold CV)

| Rank | Model | CV mean | CV std |
|:---:|---|---:|---:|
| 1 | **Hard-vote (LogReg + SVM + GBDT)** | **0.9363** | 0.0445 |
| 2 | LogReg (poly=2, L2, C=1)           | 0.9296 | 0.0235 |
| 3 | SVM (linear, C=10)                 | 0.9234 | 0.0516 |
| 4 | GBDT (sklearn GB)                  | 0.8982 | 0.0694 |
| 5 | Baseline kNN(k=5)                  | 0.8268 | 0.0635 |

## Deliverables

* Script: `experiments/exp04_ensemble.py`
* Grid CSV: `experiments/exp04_ensemble_grid.csv`
* Submission: `submissions/submission_ensemble.csv` (promoted to
  `submission.csv`).
