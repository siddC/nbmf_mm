# NBMF‑MM

[![PyPI](https://img.shields.io/pypi/v/nbmf-mm.svg)](https://pypi.org/project/nbmf-mm/)
[![License](https://img.shields.io/pypi/l/nbmf-mm.svg)](./LICENSE.md)
[![Python](https://img.shields.io/pypi/pyversions/nbmf-mm.svg)](https://pypi.org/project/nbmf-mm/)

**NBMF‑MM** is a fast, scikit‑learn‑style implementation of **mean‑parameterized Bernoulli (binary) matrix factorization** using a **Majorization–Minimization (MM)** solver.

- Two orientations (symmetric):
  - **`orientation="dir-beta"`** (a.k.a. Aspect Bernoulli): **columns of `H`** lie on the simplex; Beta prior on `W`.
  - **`orientation="beta-dir"`** (a.k.a. Binary ICA): **rows of `W`** lie on the simplex; Beta prior on `H`.
- **Masked training** for matrix completion / hold‑out validation.
- Optional acceleration: **NumExpr** (elementwise ops) and **Numba** (simplex projections), if installed.

> **Goal:** provide a fast, robust approximation you can dimension‑sweep in place of fully Bayesian NBMF.

---

## Installation

```bash
pip install nbmf-mm
# optional extras
pip install "nbmf-mm[sklearn]"   # scikit-learn integration (BaseEstimator/TransformerMixin, NNDSVD init)
pip install "nbmf-mm[docs]"      # docs build
```
---

## Quick Start

```python
import numpy as np
from nbmf_mm import NBMF  # alias of BernoulliNMF_MM

rng = np.random.default_rng(0)
X = (rng.random((100, 500)) < 0.25).astype(float)   # binary data in {0,1}

# Aspect Bernoulli: columns of H are on the simplex; W has a Beta prior
model = NBMF(
    n_components=20,
    orientation="dir-beta",
    alpha=1.2, beta=1.2,
    random_state=0, max_iter=2000, tol=1e-6,
    use_numexpr=True, use_numba=True
).fit(X)

W = model.W_                 # shape (n_samples, n_components)
H = model.components_        # shape (n_components, n_features)
Xhat = model.inverse_transform(W)  # probabilities in (0,1)

# Transform new data using fixed components H
X_new = (rng.random((10, 500)) < 0.25).astype(float)
W_new = model.transform(X_new)     # shape (10, n_components)
```

### Masking / hold-out
```python
mask = (rng.random(X.shape) < 0.9).astype(float)  # observe 90% entries
model = NBMF(n_components=20, orientation="dir-beta").fit(X, mask=mask)

# fast metrics
print("score (−NLL per obs):", model.score(X, mask=mask))
print("perplexity:", model.perplexity(X, mask=mask))
```

## Command-line (CLI)
```bash
nbmf-mm fit \
  --input X.npz --rank 30 \
  --orientation dir-beta --alpha 1.2 --beta 1.2 \
  --max-iter 2000 --tol 1e-6 --seed 0 \
  --mask train_mask.npz \
  --out model_rank30.npz
```
Outputs W, H, Xhat, objective_history, and summary metrics.

