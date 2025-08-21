"""
NBMF-MM: Non-negative Binary Matrix Factorization via Majorization-Minimization

Implementation of the algorithm from:
P. Magron and C. Févotte, "A majorization-minimization algorithm for
nonnegative binary matrix factorization," IEEE Signal Processing Letters, 2022.
"""

from .nbmf_mm import NBMFMM, NBMF

__version__ = '0.2.0'
__all__ = ['NBMFMM', 'NBMF']