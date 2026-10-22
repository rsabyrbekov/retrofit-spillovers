"""Direct, indirect and total impacts (LeSage & Pace, 2009).

For the SAR model, dy/dx_k' = beta_k (I - rho W)^{-1}. The average direct
effect is beta_k * mean(diag((I - rho W)^{-1})), the average total effect is
beta_k / (1 - rho) for row-standardized W, and indirect = total - direct.
Inference by simulating from the asymptotic distribution of (beta, rho).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sar import SARResult


def _avg_diag(rho: float, eigs: np.ndarray) -> float:
    # trace((I - rho W)^{-1}) / n via eigenvalues
    return float(np.mean(1 / (1 - rho * eigs)))


def impacts(res: SARResult, names: list[str], skip_const: bool = True,
            draws: int = 1000, seed: int = 0) -> pd.DataFrame:
    k = len(res.beta)
    theta = np.r_[res.beta, res.rho]
    V = res.cov[: k + 1, : k + 1]
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(theta, V, size=draws)
    sims[:, k] = np.clip(sims[:, k], res.eigs.min() ** -1 + 1e-6, 0.999)
    ad = np.array([_avg_diag(r, res.eigs) for r in sims[:, k]])
    ad_hat = _avg_diag(res.rho, res.eigs)
    rows = []
    for j in range(k):
        if skip_const and names[j] in ("const", "Intercept"):
            continue
        b = res.beta[j]
        point = np.array([b * ad_hat, b / (1 - res.rho) - b * ad_hat, b / (1 - res.rho)])
        bs = sims[:, j]
        draw = np.column_stack([bs * ad, bs / (1 - sims[:, k]) - bs * ad, bs / (1 - sims[:, k])])
        se = draw.std(axis=0, ddof=1)
        rows.append({"variable": names[j], "direct": point[0], "direct_se": se[0],
                     "indirect": point[1], "indirect_se": se[1],
                     "total": point[2], "total_se": se[2]})
    return pd.DataFrame(rows).set_index("variable")
