import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrofit_spillovers import (simulate, torus_weights, torus_eigenvalues, fit_ols, fit_sar,
                                 impacts, moran_test, VARIABLES)


def test_torus_eigenvalues_match_numerical():
    W = torus_weights(8).toarray()
    assert np.allclose(np.sort(np.linalg.eigvalsh(W)), np.sort(torus_eigenvalues(8)))


def test_sar_recovers_parameters():
    d = simulate(rho=0.55, seed=3)
    res = fit_sar(d["y"], d["X"], d["W"], eigs=torus_eigenvalues(20))
    assert abs(res.rho - 0.55) < 4 * res.se[5]
    assert np.all(np.abs(res.beta[1:] - d["beta"][1:]) < 4 * res.se[1:5])


def test_eigs_closed_form_equals_numeric_path():
    d = simulate(rho=0.4, seed=5)
    a = fit_sar(d["y"], d["X"], d["W"], eigs=torus_eigenvalues(20))
    b = fit_sar(d["y"], d["X"], d["W"])
    assert abs(a.rho - b.rho) < 1e-6


def test_total_impact_identity():
    d = simulate(rho=0.5, seed=7)
    res = fit_sar(d["y"], d["X"], d["W"], eigs=torus_eigenvalues(20))
    imp = impacts(res, VARIABLES, draws=200)
    row = imp.loc["voucher"]
    assert np.isclose(row.direct + row.indirect, row.total)
    assert np.isclose(row.total, res.beta[4] / (1 - res.rho))


def test_moran_detects_omitted_lag():
    d = simulate(rho=0.6, seed=9)
    m = moran_test(fit_ols(d["y"], d["X"]).resid, d["W"], permutations=199)
    assert m["I"] > 0.1 and m["p_value"] < 0.05


def test_matches_pysal_spreg():
    spreg = pytest.importorskip("spreg")
    d = simulate(rho=0.55, seed=11)
    from libpysal.weights import WSP
    w = WSP(d["W"]).to_W()
    m = spreg.ML_Lag(d["y"][:, None], d["X"][:, 1:], w=w, method="full")
    ours = fit_sar(d["y"], d["X"], d["W"], eigs=torus_eigenvalues(20))
    assert abs(float(m.rho) - ours.rho) < 1e-3
    assert np.allclose(np.ravel(m.betas)[:-1], ours.beta, atol=1e-2)
