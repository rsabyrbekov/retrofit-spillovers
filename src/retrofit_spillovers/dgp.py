"""Data-generating process: a linear-quadratic network game among households.

Household i chooses retrofit spending y_i to maximize
    U_i = a_i y_i - y_i^2 / 2 + rho * y_i * sum_j w_ij y_j,
with private return a_i = x_i' beta + eps_i. The best response is
    y_i = a_i + rho * sum_j w_ij y_j,
so the unique Nash equilibrium (|rho| < 1) is the SAR model
    y = (I - rho W)^{-1} (X beta + eps).
Spending is measured in $100s.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

VARIABLES = ["const", "log_income", "homeowner", "building_age_dec", "voucher"]
TRUE_BETA = np.array([-36.0, 4.0, 3.0, 0.8, 5.0])


def torus_weights(N: int = 20) -> sp.csr_matrix:
    """Row-standardized rook contiguity on an N x N torus (every cell has 4 neighbors)."""
    n = N * N
    rows, cols = [], []
    for i in range(n):
        r, c = divmod(i, N)
        for rr, cc in (((r - 1) % N, c), ((r + 1) % N, c), (r, (c - 1) % N), (r, (c + 1) % N)):
            rows.append(i)
            cols.append(rr * N + cc)
    W = sp.csr_matrix((np.full(len(rows), 0.25), (rows, cols)), shape=(n, n))
    return W


def torus_eigenvalues(N: int = 20) -> np.ndarray:
    """Closed-form eigenvalues of the torus W: (cos(2*pi*i/N) + cos(2*pi*j/N)) / 2."""
    k = np.cos(2 * np.pi * np.arange(N) / N)
    return ((k[:, None] + k[None, :]) / 2).ravel()


def simulate(N: int = 20, rho: float = 0.55, sigma: float = 3.0, beta: np.ndarray = TRUE_BETA,
             assign: str = "random", seed: int = 1) -> dict:
    """Simulate one cross-section of households.

    assign: "random" (voucher lottery, 25% of households) or "cluster"
            (four 5x5 blocks of the grid; requires N == 20).
    Returns dict with y, X (n x 5, including constant), W, and the true parameters.
    """
    rng = np.random.default_rng(seed)
    W = torus_weights(N)
    n = N * N
    # spatially sorted income: smooth white noise over the grid
    z = rng.standard_normal(n)
    for _ in range(4):
        z = 0.3 * z + 0.7 * (W @ z)
    z = (z - z.mean()) / z.std()
    log_income = 10.9 + 0.32 * z + 0.12 * rng.standard_normal(n)
    homeowner = (rng.random(n) < 0.6).astype(float)
    age = rng.random(n) * 6
    eps = rng.standard_normal(n)
    voucher = np.zeros(n)
    if assign == "random":
        voucher[rng.permutation(n)[: n // 4]] = 1
    elif assign == "cluster":
        if N != 20:
            raise ValueError("cluster assignment is defined for N=20")
        sr, sc = rng.integers(0, N, 2)
        for br, bc in ((0, 0), (0, 10), (10, 0), (10, 10)):
            for a in range(5):
                for b in range(5):
                    voucher[((sr + br + a) % N) * N + (sc + bc + b) % N] = 1
    else:
        raise ValueError("assign must be 'random' or 'cluster'")
    X = np.column_stack([np.ones(n), log_income, homeowner, age, voucher])
    a = X @ beta + sigma * eps
    y = spla.spsolve((sp.identity(n, format="csc") - rho * W).tocsc(), a)
    return {"y": y, "X": X, "W": W, "N": N, "rho": rho, "sigma": sigma, "beta": np.asarray(beta)}
