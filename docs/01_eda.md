# 01 - Exploratory Data Analysis

## Dataset at a glance

| Item | Value |
|---|---|
| Task | Binary classification (`target ∈ {0, 1}`) |
| Train rows | 156 |
| Test rows | 84 |
| Features | 4 numeric (`f1`, `f2`, `f3`, `f4`) |
| Missing values | 0 in both train and test |
| Duplicate IDs | 0 |
| Train/test ID overlap | 0 |
| Class balance | 1: 86 (55.1%), 0: 70 (44.9%) |

The features are already roughly standardized (per-column mean ≈ 0, std ≈ 1,
range ≈ ±3), so additional scaling matters mainly for distance / margin based
models (kNN, SVM, LogReg) and not for tree ensembles.

## Signal per feature (Pearson correlation with `target`)

| Feature | Corr |
|---|---|
| `f1` | **+0.609** (strong) |
| `f2` | **+0.454** (moderate) |
| `f3` | −0.162 (weak) |
| `f4` | −0.053 (near zero) |

Interpretation: `f1` and `f2` carry most of the linear signal, while `f3` and
`f4` may still contribute through non-linear interactions. We keep all four
features and let the models decide how to weight them.

## Modelling implications

1. **Very small dataset (156 rows)** – prefer models with strong implicit
   regularization (linear / RBF-SVM / shallow boosted trees). Deep networks or
   heavy ensembles are expected to overfit. All evaluation uses
   `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. **Public leaderboard = only part of the hidden test set.** We tune purely
   with cross-validation on the training set and never probe the leaderboard.
3. **Standardization only from train statistics** – implemented via
   `sklearn.pipeline` so no information leaks from CV folds into training.
4. **Baseline to beat**: `baseline.py` ships a kNN(k=5) model that we treat as
   the lower bound. Any new submission must beat it on 5-fold CV before being
   promoted to `submission.csv`.
