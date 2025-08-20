# nbmf‑mm — Mean‑parameterized Bernoulli NMF via Majorization–Minimization

[![CI](https://github.com/siddC/nbmf_mm/actions/workflows/ci.yml/badge.svg)](https://github.com/siddC/nbmf_mm/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/nbmf-mm.svg)](https://pypi.org/project/nbmf-mm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

**nbmf‑mm** implements *mean‑parameterized* Bernoulli matrix factorization
\(\Theta = W H \in (0,1)^{M\times N}\) with a **Majorization–Minimization (MM)**
solver, following **Magron & Févotte (2022)**. It exposes a scikit‑learn‑style
API and two symmetric orientations:

- **`orientation="dir-beta"` (Aspect Bernoulli; default)**  
  Columns of **H** lie on the probability simplex (sum‑to‑1); **W** has a
  Beta\((\alpha,\beta)\) prior.

- **`orientation="beta-dir"` (Binary ICA / bICA)**  
  Rows of **W** lie on the probability simplex; **H** has a
  Beta\((\alpha,\beta)\) prior.

**Projection choices** for the simplex‑constrained factor:

- **`projection_method="normalize"` (default; theory‑first)** – paper‑exact
  multiplicative step with the **/N** (or **/M**) normalizer that preserves the
  simplex in exact arithmetic and enjoys the classical MM monotonicity
  guarantee.

- **`projection_method="duchi"` (fast)** – Euclidean projection to the simplex
  (Duchi et al., 2008) performed **after** the multiplicative step. This is
  typically near‑identical to `"normalize"` numerically, but the formal MM
  monotonicity guarantee applies to `"normalize"`.

> Masked training (matrix completion) is supported: only observed entries
> contribute to the likelihood and updates. In the simplex steps the paper‑exact
> **/N** (or **/M**) normalizer is naturally replaced by **per‑row** (or
> **per‑column**) observed counts, preserving the simplex under masking.

---

## Installation

```bash
pip install nbmf-mm
