"""Monte Carlo: bias and RMSE of OLS vs SAR-ML for the voucher's total effect.

Usage: python scripts/monte_carlo.py [--reps 300]
Writes results/monte_carlo.csv and results/monte_carlo.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrofit_spillovers import simulate, torus_eigenvalues, fit_ols, fit_sar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    eigs = torus_eigenvalues(20)
    rows = []
    for assign in ("random", "cluster"):
        for rho in (0.0, 0.3, 0.55, 0.75):
            true_total = 5.0 / (1 - rho)
            o_tot, s_tot, s_rho, s_inc, o_inc, cover = [], [], [], [], [], []
            for rep in range(a.reps):
                d = simulate(rho=rho, assign=assign, seed=10_000 + rep)
                ols = fit_ols(d["y"], d["X"])
                sar = fit_sar(d["y"], d["X"], d["W"], eigs=eigs)
                o_tot.append(ols.beta[4])           # what a naive evaluation reports
                s_tot.append(sar.beta[4] / (1 - sar.rho))
                s_rho.append(sar.rho)
                o_inc.append(ols.beta[1]); s_inc.append(sar.beta[1])
                cover.append(sar.profile_ci[0] <= rho <= sar.profile_ci[1])
            o_tot, s_tot = np.array(o_tot), np.array(s_tot)
            rows.append({"assignment": assign, "rho": rho, "true_total_voucher": true_total,
                         "ols_mean": o_tot.mean(), "ols_rmse": np.sqrt(np.mean((o_tot - true_total) ** 2)),
                         "sar_mean": s_tot.mean(), "sar_rmse": np.sqrt(np.mean((s_tot - true_total) ** 2)),
                         "rho_hat_mean": np.mean(s_rho), "rho_ci_coverage": np.mean(cover),
                         "income_ols_mean": np.mean(o_inc), "income_sar_mean": np.mean(s_inc)})
            print(rows[-1])
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "monte_carlo.csv", index=False)
    (out / "monte_carlo.md").write_text(df.round(3).to_markdown(index=False))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
