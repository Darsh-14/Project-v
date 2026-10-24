import numpy as np
import pytest

from projectv.monitor.asm import fit_nominal, mahalanobis
from projectv.monitor.metrics import (
    auprc,
    auroc,
    cluster_bootstrap_ci,
    detection_steps,
    ewma,
    prefix_score,
    tpr_at_fpr,
)


def _streams(n_ok=200, n_fail=200, T=50, shift=3.0, seed=0):
    """Per-step raw scores: failures drift upward after step 5."""
    rng = np.random.default_rng(seed)
    ok = rng.normal(0, 1, (n_ok, T))
    fail = rng.normal(0, 1, (n_fail, T))
    fail[:, 5:] += shift
    m = np.concatenate([ok, fail])
    y = np.concatenate([np.zeros(n_ok), np.ones(n_fail)])
    return m, y


def test_ewma_matches_recursion():
    m = np.array([[1.0, 2.0, 3.0, 4.0]])
    e = ewma(m, alpha=0.5)
    ref = [1.0]
    for x in m[0, 1:]:
        ref.append(0.5 * x + 0.5 * ref[-1])
    np.testing.assert_allclose(e[0], ref)


def test_ewma_alpha_one_is_identity():
    m = np.random.default_rng(0).normal(size=(3, 10))
    np.testing.assert_allclose(ewma(m, 1.0), m)


def test_prefix_score_uses_only_prefix():
    e = np.zeros((1, 50))
    e[0, 30] = 10.0
    assert prefix_score(e, 10)[0] == 0.0
    assert prefix_score(e, 31)[0] == 10.0


def test_auroc_separable_is_one():
    m, y = _streams(shift=50.0)
    s = prefix_score(ewma(m, 0.3), 25)
    assert auroc(s, y) == pytest.approx(1.0)
    assert auprc(s, y) == pytest.approx(1.0)


def test_auroc_shuffled_labels_is_chance():
    m, y = _streams(n_ok=2000, n_fail=2000, shift=3.0)
    s = prefix_score(ewma(m, 0.3), 25)
    y_shuf = np.random.default_rng(1).permutation(y)
    assert abs(auroc(s, y_shuf) - 0.5) <= 0.05


def test_tpr_at_fpr_threshold_from_calib():
    rng = np.random.default_rng(0)
    calib_ok = rng.normal(0, 1, 10_000)
    test_s = np.concatenate([rng.normal(0, 1, 5000), rng.normal(100, 1, 5000)])
    test_y = np.concatenate([np.zeros(5000), np.ones(5000)])
    tpr, thr, fpr = tpr_at_fpr(calib_ok, test_s, test_y, fpr=0.05)
    assert thr == pytest.approx(np.quantile(calib_ok, 0.95))
    assert tpr == pytest.approx(1.0)
    assert abs(fpr - 0.05) < 0.01


def test_detection_steps():
    e = np.array([[0, 0, 5, 9], [0, 0, 0, 0]], dtype=float)
    d = detection_steps(e, threshold=4.0)
    assert d[0] == 2
    assert np.isnan(d[1])


def test_mahalanobis_nominal_vs_shifted():
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    z = rng.multivariate_normal([0, 0], cov, size=5000)
    model = fit_nominal(z)
    # Along the anti-correlated direction a small step is far in Mahalanobis terms.
    d_anti = mahalanobis(model, np.array([[1.0, -1.0]]))[0]
    d_corr = mahalanobis(model, np.array([[1.0, 1.0]]))[0]
    assert d_anti > 2 * d_corr
    assert np.mean(mahalanobis(model, z) ** 2) == pytest.approx(2.0, rel=0.1)


def test_mahalanobis_keeps_leading_axes():
    z = np.random.default_rng(0).normal(size=(500, 3))
    model = fit_nominal(z)
    assert mahalanobis(model, z.reshape(50, 10, 3)).shape == (50, 10)


def test_cluster_bootstrap_ci_contains_point_estimate():
    m, y = _streams(n_ok=300, n_fail=300, shift=1.0)
    s = prefix_score(ewma(m, 0.3), 25)
    clusters = np.repeat([0, 1, 2], 200)
    point, lo, hi = cluster_bootstrap_ci(auroc, s, y, clusters, n_boot=300, seed=0)
    assert lo <= point <= hi
    assert hi - lo < 0.2
