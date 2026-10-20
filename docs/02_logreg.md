# 02 - Logistic Regression

## Why Logistic Regression first

The EDA (see `01_eda.md`) shows that `f1` (ρ ≈ 0.61) and `f2` (ρ ≈ 0.45) carry
strong *linear* signal with respect to `target`, that the dataset is tiny
(156 rows) and that features are already roughly standardized. On such a
dataset a well-regularised linear model is expected to be hard to beat:

* linear boundaries have low variance, which matters when every CV fold has
  only ~31 held-out samples;
* `L2` penalty acts as a tunable bias/variance knob (`C` = 1/λ);
* a small polynomial expansion (`degree=2`) lets us test whether a
  slightly non-linear boundary improves the fit without jumping to a full
  non-parametric model.

The model is wrapped in a `Pipeline(StandardScaler, [PolynomialFeatures,
StandardScaler], LogisticRegression)` so that scaling is fitted inside each
CV fold — no leakage from validation data.

## Grid search

Evaluation protocol: `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)` on the training set, scoring = accuracy. Reference baseline
`kNN(k=5)` was re-run under the same CV: **0.8268 ± 0.0635**.

| Hyper-parameter | Values tried | Rationale |
|---|---|---|
| `poly` (degree of `PolynomialFeatures`) | 1, 2, 3 | 1 = pure linear; 2 picks up pairwise interactions and squared terms; 3 checks whether more flexibility helps or just overfits. |
| `penalty` | `l2`, `l1` | `l2` is the standard safe choice; `l1` adds feature selection in case `f3`/`f4` are pure noise. |
| `C` (inverse regularization strength) | 0.01, 0.1, 0.5, 1, 2, 5, 10, 100 | Spans heavy regularization to almost unregularized across 4 orders of magnitude. |

Top rows of the grid (full table saved as
`experiments/exp01_logreg_grid.csv`):

| poly | penalty | C | CV mean | CV std |
|---:|:---:|---:|---:|---:|
| **2** | **l2** | **1.0** | **0.9296** | **0.0235** |
| 1 | l2 | 0.1 | 0.9169 | 0.0475 |
| 2 | l2 | 0.5 | 0.9169 | 0.0429 |
| 1 | l2 | 0.5 | 0.9167 | 0.0436 |
| 1 | l2 | 10 | 0.9167 | 0.0327 |

## Chosen configuration

```
StandardScaler
PolynomialFeatures(degree=2, include_bias=False)
StandardScaler
LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
```

* **5-fold CV accuracy:** **0.9296 ± 0.0235**
* **Gain over baseline kNN(5):** +10.28 points (and much lower variance).
* **Observations:**
  * Degree 3 did not improve further and started inflating variance — a sign
    that quadratic interactions already capture the true boundary.
  * `l1` and `l2` were tied for most `C` values, suggesting the four features
    all contribute at least weakly; nothing is reliably zeroed out.
  * The model is extremely stable (std ≈ 0.024) which is important when the
    public leaderboard uses only part of the hidden test set.

## Deliverables

* Script: `experiments/exp01_logreg.py`
* Grid CSV: `experiments/exp01_logreg_grid.csv`
* Submission: `submissions/submission_logreg.csv` (also mirrored to
  `submission.csv` as the current best).
