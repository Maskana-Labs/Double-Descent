"""Sanity checks. Run with:  python test_experiment.py   (pytest also works)."""

import numpy as np

from experiment import fit_ridge_svd, make_features, one_trial, sweep


def test_min_norm_matches_lstsq_when_overdetermined():
    rng = np.random.default_rng(0)
    Phi, y = rng.standard_normal((40, 10)), rng.standard_normal(40)
    expected = np.linalg.lstsq(Phi, y, rcond=None)[0]
    np.testing.assert_allclose(fit_ridge_svd(Phi, y, 0.0), expected, atol=1e-8)


def test_min_norm_interpolates_and_is_shortest_when_underdetermined():
    rng = np.random.default_rng(1)
    Phi, y = rng.standard_normal((20, 80)), rng.standard_normal(20)
    w = fit_ridge_svd(Phi, y, 0.0)
    assert np.max(np.abs(Phi @ w - y)) < 1e-8  # fits training data exactly
    np.testing.assert_allclose(w, np.linalg.pinv(Phi) @ y, atol=1e-8)  # pinv = min-norm solution


def test_ridge_matches_closed_form():
    rng = np.random.default_rng(2)
    Phi, y = rng.standard_normal((30, 12)), rng.standard_normal(30)
    lam = 0.3
    expected = np.linalg.solve(Phi.T @ Phi + lam * np.eye(12), Phi.T @ y)
    np.testing.assert_allclose(fit_ridge_svd(Phi, y, lam), expected, atol=1e-8)


def test_feature_shape_and_scale():
    X = np.random.default_rng(3).standard_normal((5, 4))
    W = np.random.default_rng(4).standard_normal((4, 16))
    F = make_features(X, W)
    assert F.shape == (5, 16) and (F >= 0).all()


def test_trial_and_sweep_are_reproducible():
    a = one_trial(30, (0.0, 0.1), 20, 50, 5, 0.5, np.random.default_rng(7))
    b = one_trial(30, (0.0, 0.1), 20, 50, 5, 0.5, np.random.default_rng(7))
    assert a == b
    kw = dict(n_train=15, n_test=50, d=4, p_max=40, seeds=3, lams=(0.0,), seed=5)
    assert sweep(**kw) == sweep(**kw)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
