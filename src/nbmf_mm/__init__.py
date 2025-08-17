"""
NBMF-MM: Bernoulli Mean-Parameterized Non-negative Binary Matrix Factorization with Majorization-Minimization

Public API for nbmf_mm.
A scikit-learn compatible implementation of Bernoulli mean-parameterized 
Non-negative Binary Matrix Factorization using Majorization-Minimization.
"""
from ._version import __version__
from .estimator import BernoulliNMF_MM

__all__ = ["BernoulliNMF_MM", "__version__"]
