"""Simulate one sample, estimate OLS and SAR, print results and save figures.

Usage: python scripts/run_demo.py [--rho 0.55] [--sigma 3] [--assign random|cluster] [--seed 1]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrofit_spillovers import (simulate, torus_eigenvalues, fit_ols, fit_sar, impacts,
                                 moran_test, VARIABLES)  # noqa: E402

VOUCHER_USD = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.55)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--assign", default="random")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    d = simulate(rho=a.rho, sigma=a.sigma, assign=a.assign, seed=a.seed)
    y, X, W = d["y"], d["X"], d["W"]
    ols = fit_ols(y, X)
    mor = moran_test(ols.resid, W)
    sar = fit_sar(y, X, W, eigs=torus_eigenvalues(d["N"]))
    imp = impacts(sar, VARIABLES)

    est = pd.DataFrame({"true": np.r_[d["beta"], a.rho],
                        "ols": np.r_[ols.beta, 0.0], "ols_se": np.r_[ols.se, np.nan],
                        "sar": np.r_[sar.beta, sar.rho], "sar_se": sar.se[: len(sar.beta) + 1]},
                       index=VARIABLES + ["rho"])
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n== Estimates (spending in $100s) ==\n", est)
    print(f"\nMoran's I on OLS residuals: {mor['I']:.3f} (z={mor['z']:.1f}, p={mor['p_value']:.3f})")
    print(f"LR test rho=0: chi2(1)={sar.lr_stat:.1f}, p={sar.lr_pvalue:.2g}; "
          f"95% profile CI for rho: [{sar.profile_ci[0]:.3f}, {sar.profile_ci[1]:.3f}]")
    print("\n== Impacts ==\n", imp)
    v = imp.loc["voucher"]
    print(f"\nPer ${VOUCHER_USD} voucher: OLS ${ols.beta[4]*100:,.0f} | SAR direct ${v.direct*100:,.0f}"
          f" + spillover ${v.indirect*100:,.0f} = total ${v.total*100:,.0f}"
          f" (true ${d['beta'][4]/(1-a.rho)*100:,.0f})")

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    N = d["N"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    im = ax[0].imshow(y.reshape(N, N) * 100, cmap="viridis")
    vr, vc = np.divmod(np.where(X[:, 4] == 1)[0], N)
    ax[0].scatter(vc, vr, s=8, c="orange", label="voucher")
    ax[0].set_title("Retrofit spending ($)")
    ax[0].legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    r = ols.resid
    lim = np.abs(r).max() * 100
    im2 = ax[1].imshow(r.reshape(N, N) * 100, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1].set_title(f"OLS residuals ($), Moran's I = {mor['I']:.2f}")
    fig.colorbar(im2, ax=ax[1], fraction=0.046)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "maps.png", dpi=150)
    print(f"\nSaved {out/'maps.png'}")


if __name__ == "__main__":
    main()
