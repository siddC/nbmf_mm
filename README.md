# NBMF‑MM

[![CI](https://github.com/siddC/nbmf_mm/actions/workflows/ci.yml/badge.svg)](https://github.com/siddC/nbmf_mm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
![Python versions](https://img.shields.io/badge/python-3.9–3.12-blue)

**NBMF‑MM** is a fast, scikit‑learn‑style implementation of **mean‑parameterized Bernoulli (binary) matrix factorization** using a **Majorization–Minimization (MM)** solver.

- Two symmetric orientations:
  - **`orientation="dir-beta"`** (default, *Aspect Bernoulli*): **columns of `H`** lie on the simplex; Beta prior on `W`.
  - **`orientation="beta-dir"`** (*Binary ICA*): **rows of `W`** lie on the simplex; Beta prior on `H`.
- **Masked training** for matrix completion / hold‑out validation.
- Optional acceleration: **NumExpr** (elementwise ops) and **Numba** (simplex projection).
- **Projection options**: default **Duchi** simplex projection (fast) with an opt‑in **legacy “normalize”** method for parity with older behavior.

---

## Installation

From PyPI (when released):

```bash
pip install nbmf-mm
```
From source:
```bash
pip install "git+https://github.com/siddC/nbmf_mm"
```

Optional extras:
```bash
# scikit-learn integration & NNDSVD-style init (if you enable it later)
pip install "nbmf-mm[sklearn]"

# docs build stack
pip install "nbmf-mm[docs]"
```

---

## Quick Start

```python
import numpy as np
from nbmf_mm import NBMF

rng = np.random.default_rng(0)
X = (rng.random((100, 500)) < 0.25).astype(float)   # binary {0,1} or probabilities in [0,1]

# Aspect Bernoulli (default): H columns on simplex; W has a Beta prior
model = NBMF(
    n_components=20,
    orientation="dir-beta",
    alpha=1.2, beta=1.2,
    random_state=0,
    max_iter=2000, tol=1e-6,
    # fast defaults:
    projection_method="duchi",      # Euclidean simplex projection (recommended)
    projection_backend="auto",      # prefer Numba if installed
    use_numexpr=True,               # use NumExpr if installed
).fit(X)

W = model.W_                 # shape (n_samples, n_components)
H = model.components_        # shape (n_components, n_features)
Xhat = model.inverse_transform(W)  # probabilities in (0,1)

# Transform new data using fixed components H
X_new = (rng.random((10, 500)) < 0.25).astype(float)
W_new = model.transform(X_new)     # shape (10, n_components)

# Masked training / hold-out validation
mask = (rng.random(X.shape) < 0.9).astype(float)  # observe 90% of entries
model = NBMF(n_components=20).fit(X, mask=mask)

print("score (−NLL per observed entry):", model.score(X, mask=mask))
print("perplexity:", model.perplexity(X, mask=mask))
```

---

## Why two orientations?

- dir-beta (Aspect Bernoulli) — H columns are on the simplex → each feature (e.g., gene) has interpretable mixture memberships across latent aspects. W carries sample‑specific propensities with a Beta prior.

- beta-dir (Binary ICA) — W rows are on the simplex; H is Beta‑constrained.

Both solve the same Bernoulli mean‑parameterized factorization with different geometric constraints; pick the one that best matches your interpretability needs.

---

## API (scikit-learn style)
- `NBMF(...).fit(X, mask=None) -> self`
- `fit_transform(X, mask=None) -> W`
- `transform(X, mask=None, max_iter=500, tol=1e-6) -> W` (estimate `W` for new `X` with learned `H` fixed)
- `inverse_transform(W) -> Xhat` (reconstructed probabilities in (0,1))
- `score(X, mask=None) -> float` (negative NLL per observed entry; higher is better)
- `perplexity(X, mask=None) -> float` (exp of average NLL per observed entry; lower is better)

## Key parameters
- `n_components` (int) — rank 𝐾
- `orientation` ∈ {`"dir-beta"`, `"beta-dir"`}.
- `alpha, beta` (float > 0) — Beta prior hyperparameters (on `W` for dir-beta, on `H` for `beta-dir`).
- `projection_method` ∈ {`"duchi"`, `"normalize"`} — default `"duchi"` (fast & stable). `"normalize"` gives legacy behavior (nonnegativity + renormalization).
- `projection_backend` ∈ {`"auto"`, `"numba"`, `"numpy"`} — backend for `"duchi"` projection.
- `use_numexpr` (bool) — use NumExpr if available.
---

## Command-line (CLI)
After installation, a console script nbmf-mm is available:
```bash
nbmf-mm fit \
  --input X.npz --rank 30 \
  --orientation dir-beta --alpha 1.2 --beta 1.2 \
  --max-iter 2000 --tol 1e-6 --seed 0 --n-init 1 \
  --mask train_mask.npz \
  --out model_rank30.npz
```
This writes an `.npz` with `W`, `H`, `Xhat`, `objective_history`, and `n_iter`.

Input formats: `.npz` (expects key `arr_0`) or `.npy`. Masks are optional and must match `X` shape.
---

## Data requirements
- `X` must be in [0,1] (binary recommended; probabilistic inputs are allowed).
- `mask` (optional) must be the same shape as X, with values in [0,1] (typically {0,1}).
**Sparse inputs** (scipy.sparse) and masks are accepted and densified internally in this version.
---

## Performance notes
- The default **Duchi** projection gives an 𝑂(𝑑*log⁡(𝑑))
O(dlogd) per‑row/column simplex projection and is accelerated with **Numba** when installed.

**NumExpr** speeds large elementwise expressions.

Both accelerations are optional and degrade gracefully if not present.
---

## Reproducibility
- Set `random_state` (int) for reproducible initialization.
- Use `n_init > 1` to run several random restarts and keep the best NLL.
---

## References
- **Simplex projection** (default):
  - J. Duchi, S. Shalev‑Shwartz, Y. Singer, T. Chandra (2008).
  Efficient Projections onto the ℓ₁‑Ball for Learning in High Dimensions. ICML 2008.
  
  - W. Wang, M. Á. Carreira‑Perpiñán (2013).
  Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application. arXiv:1309.1541.
  
  - Bayesian NBMF (related, slower but fully Bayesian):
    - See the `NBMF` project by alumbreras for reference implementations of Bayesian variants.