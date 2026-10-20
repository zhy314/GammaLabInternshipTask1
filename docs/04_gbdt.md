# 04 - Gradient Boosted Decision Trees

## Why GBDT

Even though the EDA suggests a near-linear boundary, GBDT is the de-facto
industry baseline for tabular data: it is non-parametric, captures feature
interactions automatically, is scale-invariant, and typically gives a strong
lower-bound on what's possible. We run it to confirm that the previous
LogReg/SVM results are not missing a non-linear structure.

We benchmark two implementations in parallel:

* `sklearn.ensemble.GradientBoostingClassifier` – the reference pure-Python
  implementation.
* `lightgbm.LGBMClassifier` – the fast histogram-based gradient booster
  widely used in production.

## Grid search

Protocol identical to the earlier experiments
(`StratifiedKFold(5, shuffle=True, random_state=42)`).

| Hyper-parameter | Values tried | Rationale |
|---|---|---|
| `n_estimators` | 100 – 800 | More rounds → more capacity, but more overfit risk on 156 rows. |
| `learning_rate` | 0.03, 0.05, 0.1 | Smaller lr + more trees usually generalizes better. |
| `max_depth` (sklearn) / `num_leaves` (lgbm) | depth 2–4 / leaves 7, 15, 31 | Shallow trees are essential when the dataset is tiny. |
| `min_child_samples` (lgbm) | 5, 10, 20 | Forces each leaf to see enough samples (strong regularizer). |
| `subsample`, `colsample_bytree` | 0.9 | Mild stochastic regularisation. |
| `reg_lambda` (lgbm) | 1.0 | L2 on leaf weights. |

Top configurations (full grid in `experiments/exp03_gbdt_grid.csv`):

| impl | n_est | max_depth | lr | num_leaves | min_child | CV mean | CV std |
|:---:|---:|:---:|---:|:---:|:---:|---:|---:|
| **sk-gb** | **400** | **4** | **0.03** | – | – | **0.8982** | 0.0694 |
| sk-gb | 200 | 2 | 0.05 | – | – | 0.8980 | 0.0642 |
| sk-gb | 400 | 3 | 0.03 | – | – | 0.8980 | 0.0642 |
| sk-gb | 100 | 2 | 0.10 | – | – | 0.8915 | 0.0582 |
| lgbm  | 200 | – | 0.10 | 7 | 10 | 0.8786 | 0.0540 |

## Verdict

| Model | 5-fold CV accuracy | CV std |
|---|---:|---:|
| Baseline kNN(5) | 0.8268 | 0.0635 |
| LogReg (poly=2, L2, C=1) | **0.9296** | **0.0235** |
| SVM (linear/RBF best)    | 0.9234 | 0.0516 |
| GBDT (best sklearn GB)   | 0.8982 | 0.0694 |

* **GBDT lags both linear models by ~3 points** and shows ~3× the variance of
  LogReg. On 156 rows, every tree sees ~125 training samples; even a
  depth-2 stump risks memorising noise once we stack hundreds of them.
* **LightGBM < sklearn-GB** here, because LightGBM's leaf-wise growth is more
  aggressive and needs more data to shine.
* **This confirms the EDA intuition**: the signal is dominated by the almost
  linear contributions of `f1` / `f2`; added tree flexibility only helps on
  training noise. We therefore keep the GBDT prediction as an *ensembling
  input* (`04_ensemble.md`) but do **not** promote it to `submission.csv`.

## Deliverables

* Script: `experiments/exp03_gbdt.py`
* Grid CSV: `experiments/exp03_gbdt_grid.csv`
* Submission: `submissions/submission_gbdt.csv` (NOT promoted to best).
