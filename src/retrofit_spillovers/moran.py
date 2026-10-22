"""Moran's I for regression residuals with a permutation test."""
from __future__ import annotations

import numpy as np


def moran_test(e: np.ndarray, W, permutations: int = 999, seed: int = 0) -> dict:
    """Moran's I = (n/S0) e'We / e'e; one-sided permutation p-value for positive autocorrelation."""
    e = np.asarray(e) - np.mean(e)
    n = len(e)
    S0 = W.sum()
    ee = e @ e
    I = n / S0 * (e @ (W @ e)) / ee
    rng = np.random.default_rng(seed)
    perm = np.empty(permutations)
    for t in range(permutations):
        p = rng.permutation(e)
        perm[t] = n / S0 * (p @ (W @ p)) / ee
    return {"I": float(I), "z": float((I - perm.mean()) / perm.std(ddof=1)),
            "p_value": float((np.sum(perm >= I) + 1) / (permutations + 1))}
