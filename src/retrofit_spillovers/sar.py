"""OLS and exact maximum likelihood for the spatial lag (SAR) model.

    y = rho W y + X beta + eps,   eps ~ N(0, sigma^2 I)

The concentrated log-likelihood is maximized over rho alone; beta and sigma^2
have closed forms given rho. The log-determinant uses the eigenvalues of W
(Ord, 1975). Standard errors come from the analytic information matrix
(Anselin, 1988), valid for symmetric W.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats


@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    resid: np.ndarray
    r2: float


@dataclass
class SARResult:
    rho: float
    beta: np.ndarray
    sigma2: float
    cov: np.ndarray            # covariance of (beta, rho, sigma2)
    loglik: float
    lr_stat: float             # LR test of rho = 0
    profile_ci: tuple          # 95% profile-likelihood CI for rho
    eigs: np.ndarray = field(repr=False)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def lr_pvalue(self) -> float:
        return float(stats.chi2.sf(self.lr_stat, 1))


def fit_ols(y: np.ndarray, X: np.ndarray) -> OLSResult:
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (n - k)
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
    return OLSResult(b, np.sqrt(np.diag(XtX_inv) * s2), e, float(r2))


def fit_sar(y: np.ndarray, X: np.ndarray, W, eigs: np.ndarray | None = None) -> SARResult:
    """Exact ML for the SAR model. Pass `eigs` if known (e.g. torus) to skip eigendecomposition."""
    n, k = X.shape
    Wd = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
    if eigs is None:
        eigs = np.linalg.eigvals(Wd).real
    lam_min, lam_max = eigs.min(), eigs.max()
    lo, hi = 1 / lam_min + 1e-6, 1 / lam_max - 1e-6

    XtX_inv = np.linalg.inv(X.T @ X)
    Wy = Wd @ y
    b0, bL = XtX_inv @ X.T @ y, XtX_inv @ X.T @ Wy
    e0, eL = y - X @ b0, Wy - X @ bL
    A, B, C = e0 @ e0, e0 @ eL, eL @ eL

    def loglik(r):
        s2 = (A - 2 * r * B + r * r * C) / n
        return -n / 2 * (np.log(2 * np.pi) + 1) - n / 2 * np.log(s2) + np.sum(np.log(1 - r * eigs))

    opt = minimize_scalar(lambda r: -loglik(r), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    rho = float(opt.x)
    beta = b0 - rho * bL
    s2 = (A - 2 * rho * B + rho * rho * C) / n
    ll = loglik(rho)

    # analytic information matrix for (beta, rho, sigma2)
    G = Wd @ np.linalg.inv(np.eye(n) - rho * Wd)
    GXb = G @ (X @ beta)
    trG, trG2, trGtG = np.trace(G), np.sum(G * G.T), np.sum(G * G)
    I = np.zeros((k + 2, k + 2))
    I[:k, :k] = X.T @ X / s2
    I[:k, k] = I[k, :k] = X.T @ GXb / s2
    I[k, k] = trG2 + trGtG + GXb @ GXb / s2
    I[k, k + 1] = I[k + 1, k] = trG / s2
    I[k + 1, k + 1] = n / (2 * s2 ** 2)
    cov = np.linalg.inv(I)

    # 95% profile-likelihood CI
    grid = np.linspace(max(lo, rho - 0.6), min(hi, rho + 0.6), 2401)
    keep = grid[np.array([loglik(g) for g in grid]) >= ll - stats.chi2.ppf(0.95, 1) / 2]
    ci = (float(keep.min()), float(keep.max()))
    return SARResult(rho, beta, float(s2), cov, float(ll), float(2 * (ll - loglik(0.0))), ci, eigs)
