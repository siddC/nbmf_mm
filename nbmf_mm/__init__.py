"""
NBMF-MM: Non-negative Binary Matrix Factorization via Majorization-Minimization

Implementation of the algorithm from:
P. Magron and C. Févotte, "A majorization-minimization algorithm for
nonnegative binary matrix factorization," IEEE Signal Processing Letters, 2022.
"""

import numpy as np
from ._base import NBMFMM, NBMF
from ._solver import nbmf_mm_solver

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = ['NBMFMM', 'NBMF', 'nbmf_mm_solver']