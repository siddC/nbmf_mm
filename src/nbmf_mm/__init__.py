"""
NBMF-MM: Bernoulli Mean-Parameterized Non-negative Binary Matrix Factorization with Majorization-Minimization

Public API for nbmf_mm.

A scikit-learn compatible implementation of Bernoulli mean-parameterized
Non-negative Binary Matrix Factorization using Majorization-Minimization.
"""

from ._version import __version__
from .estimator import NBMF

# setuptools_scm runtime version
try:
    from ._version import version as __version__  # auto-generated
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ["NBMF", "__version__"]