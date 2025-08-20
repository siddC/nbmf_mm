Drop-in "paper-exact" MM helpers and a pytest suite for nbmf_mm
===============================================================

What is in this bundle?
-----------------------
- `src/nbmf_mm/_mm_exact.py`:
  Pure-NumPy **reference** implementation of the exact MM updates from
  Magron & Févotte (2022), covering both orientations:
  * `beta-dir`: rows of W on the simplex, Beta prior on H (paper orientation).
  * `dir-beta`: columns of H on the simplex, Beta prior on W (swapped).

  These helpers use the **closed-form normalizers** (/N or /M) for the
  simplex-constrained factor and the **C/(C+D)** ratio for the Beta-constrained
  factor. They are intended for **unit tests and parity checks**, not speed.

- `tests/`:
  * `test_mm_equivalence.py` – core tests:
      - Monotonic decrease of the objective with the exact MM helpers.
      - Equivalence of `projection_method="duchi"` vs `"normalize"` in your
        estimator (same reconstruction & perplexity within tight tolerances).
      - Orientation swap symmetry check (transpose trick).
  * `test_one_step_and_masking.py` – one-step simplex preservation and masked
    training monotonicity tests.
  * `test_animals_optional.py` – auto-skips unless `animals.rda` + `pyreadr`
    are available; compares perplexity between projection methods.
  * `test_strict_parity_optional.py` – auto-skips unless your `NBMF` supports
    `init_W`/`init_H`; when available, it checks **one-step** factor parity
    against the paper-exact update.

How to use
----------
1. Create directories:

