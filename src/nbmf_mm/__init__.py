"""
NBMF-MM: Bernoulli Mean-Parameterized Non-negative Binary Matrix Factorization with Majorization-Minimization

Public API for nbmf_mm.

A scikit-learn compatible implementation of Bernoulli mean-parameterized
Non-negative Binary Matrix Factorization using Majorization-Minimization.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Prefer runtime package metadata; fall back to the file written by setuptools_scm
try:
    __version__ = _pkg_version("nbmf-mm")
except PackageNotFoundError:  # not installed or during editable dev
    try:
        from ._version import version as __version__  # type: ignore
    except Exception:  # noqa: BLE001
        __version__ = "0.0.0"

# Public API
from .estimator import NBMF

__all__ = ["NBMF", "__version__"]
