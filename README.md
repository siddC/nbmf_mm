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
