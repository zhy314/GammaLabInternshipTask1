# 03 - Support Vector Classifier

## Why SVM

With only 156 training rows and 4 numeric features, a kernel SVM is a
textbook fit:

* **Max-margin objective** gives built-in regularization and generalizes well
  on small datasets.
* **RBF kernel** can carve arbitrary smooth decision boundaries without
  needing explicit feature engineering, which complements the polynomial
  LogReg experiment from `02_logreg.md`.
* A **linear kernel** cross-check tells us whether the extra flexibility of
  RBF actually helps, or whether the problem is essentially linear.
* A **polynomial kernel** mirrors the LogReg+PolynomialFeatures trick from
  the previous experiment, allowing a direct apples-to-apples comparison.

Features are standardized inside the Pipeline so that the RBF kernel's
distance computation is isotropic across features.

## Grid search

Protocol: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the
training set.

| Hyper-parameter | Values tried | Rationale |
|---|---|---|
| `kernel` | `linear`, `rbf`, `poly` | Covers linear, locally flexible, and globally polynomial boundaries. |
| `C` | 0.01 – 100 (log spaced) | Trade-off between margin width and training error. |
| `gamma` (RBF/poly) | `scale`, `auto`, 0.05, 0.1, 0.25, 0.5, 1, 2 | Controls the "reach" of each support vector; small γ = smooth, large γ = wiggly. |
| `degree` (poly only) | 2, 3 | Mirrors the LogReg polynomial sweep. |

Top configurations (full table in `experiments/exp02_svm_grid.csv`):

| kernel | C | gamma | degree | CV mean | CV std |
|:---:|---:|:---:|:---:|---:|---:|
| **linear** | **10** | – | – | **0.9234** | 0.0516 |
| rbf | 0.5 | auto | – | 0.9234 | 0.0516 |
| rbf | 0.5 | scale | – | 0.9234 | 0.0516 |
| rbf | 0.5 | 0.25 | – | 0.9234 | 0.0516 |
| linear | 100 | – | – | 0.9169 | 0.0556 |

The RBF ties with the linear SVM (small γ ≈ linear kernel limit), confirming
that the decision boundary is close to linear. We select the **linear SVM
with `C=10`** as the representative best configuration (equivalent accuracy,
simpler model).

## Verdict

| Model | 5-fold CV accuracy | CV std |
|---|---:|---:|
| Baseline kNN(5) | 0.8268 | 0.0635 |
| LogReg (poly=2, L2, C=1) | **0.9296** | **0.0235** |
| SVM (linear/RBF best)    | 0.9234 | 0.0516 |

SVM reaches a similar level but with ~2× the variance of the LogReg champion,
so `submission.csv` at the repo root stays pointed at the LogReg prediction.
The SVM prediction is still written to `submissions/submission_svm.csv` so
that we can use it later in ensembling (`04_ensemble.md`).

## Deliverables

* Script: `experiments/exp02_svm.py`
* Grid CSV: `experiments/exp02_svm_grid.csv`
* Submission: `submissions/submission_svm.csv` (NOT promoted to best).
