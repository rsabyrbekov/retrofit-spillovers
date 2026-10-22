"""Household retrofit spillovers: a spatial autoregressive (SAR) toolkit.

Simulate households whose retrofit spending responds to neighbors' spending,
then estimate the model by OLS and by exact maximum likelihood.
"""
from .dgp import torus_weights, torus_eigenvalues, simulate, TRUE_BETA, VARIABLES
from .sar import fit_ols, fit_sar
from .effects import impacts
from .moran import moran_test

__all__ = ["torus_weights", "torus_eigenvalues", "simulate", "TRUE_BETA", "VARIABLES",
           "fit_ols", "fit_sar", "impacts", "moran_test"]
__version__ = "0.1.0"
