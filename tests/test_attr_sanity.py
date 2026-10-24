import numpy as np
import pytest
import torch

from projectv.control.analytic_controller import RELEVANT_GROUPS, PDController
from projectv.xai.attributions import (
    METHODS,
    attribute,
    gaussian_kl,
    grad_x_delta,
    group_shares,
    integrated_gradients,
)
from projectv.xai.groups import flat_dim, make_groups
from projectv.xai.policy_fn import LinearGaussian, controller_policy

RNG = np.random.default_rng(0)


def _pool(d, n=4000, scale=None):
    scale = np.ones(d) if scale is None else scale
    return RNG.normal(0, 1, (n, d)) * scale


# --- hand-built linear policy a = W s ----------------------------------------------------------
@pytest.fixture
def linear():
    d = 31
    W = RNG.normal(size=(4, d))
    pol = LinearGaussian(torch.tensor(W, dtype=torch.float64), log_std=-1.0)
    s = RNG.normal(size=d)
    pool = _pool(d)
    base = pool.mean(axis=0)
    u = W @ s / np.linalg.norm(W @ s)
    analytic = (W.T @ u) * (s - base)  # contribution of feature i to f(s) - f(base)
    return pol, s, pool, base, analytic


def test_grad_x_delta_equals_analytic_on_linear(linear):
    pol, s, _, base, analytic = linear
    np.testing.assert_allclose(grad_x_delta(pol, s, base), analytic, atol=1e-5)


def test_ig_equals_analytic_on_linear(linear):
    pol, s, _, base, analytic = linear
    np.testing.assert_allclose(integrated_gradients(pol, s, base, steps=16), analytic, atol=1e-5)


def test_ig_completeness_on_nonlinear_mlp():
    torch.manual_seed(0)
    d = 12
    net = torch.nn.Sequential(
        torch.nn.Linear(d, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
    ).double()

    class P:
        def dist(self, x):
            return net(x), torch.full((x.shape[0], 4), -1.0, dtype=x.dtype)

    pol = P()
    s, base = RNG.normal(size=d), RNG.normal(size=d)
    attr = integrated_gradients(pol, s, base, steps=256)
    with torch.no_grad():
        m_s = net(torch.tensor(s)[None])[0]
        u = m_s / m_s.norm()
        f = lambda x: float((net(torch.tensor(x)[None])[0] @ u).detach())  # noqa: E731
    assert attr.sum() == pytest.approx(f(s) - f(base), rel=1e-3)


def test_kl_zero_iff_equal_and_matches_closed_form():
    m = torch.tensor([[0.3, -1.0]], dtype=torch.float64)
    ls = torch.tensor([[-1.0, 0.5]], dtype=torch.float64)
    assert float(gaussian_kl(m, ls, m, ls)) == pytest.approx(0.0, abs=1e-12)
    m2, ls2 = m + 0.5, ls - 0.2
    s1, s2 = ls.exp(), ls2.exp()
    ref = (ls2 - ls + (s1**2 + (m - m2) ** 2) / (2 * s2**2) - 0.5).sum()
    assert float(gaussian_kl(m, ls, m2, ls2)) == pytest.approx(float(ref))


# --- analytic controller: object groups get zero attribution from every method ---------------
@pytest.mark.parametrize("method", METHODS)
def test_controller_zero_attribution_on_object_groups(method):
    task = "push"
    groups = make_groups(task)
    ctrl = PDController(task)
    pol = controller_policy(ctrl, task)
    pool = _pool(flat_dim(task), n=2000, scale=0.05)
    s = pool[7] + 0.3  # away from the pool mean so goal/ee terms are non-zero
    scores = attribute(method, pol, s, groups, pool, rng=np.random.default_rng(0))
    irrelevant = [g for g in groups if g not in RELEVANT_GROUPS]
    total = sum(abs(v) for v in scores.values())
    assert total > 0
    for g in irrelevant:
        assert abs(scores[g]) <= 1e-6 * total, (method, g, scores[g])


def test_controller_grad_x_delta_matches_analytic_contributions():
    task = "reach"
    groups = make_groups(task)
    ctrl = PDController(task)
    pol = controller_policy(ctrl, task)
    pool = _pool(flat_dim(task), n=500, scale=0.05)
    s = pool[3] + 0.2
    base = pool.mean(axis=0)
    attr = grad_x_delta(pol, s, base)
    # f = <mu(s'), u>; analytic: sum_k u_k * contribution_k for each group's features
    mu = ctrl.mean(_as_dict(task, s))
    u = mu / np.linalg.norm(mu)
    contrib = ctrl.contributions(_as_dict(task, s), _as_dict(task, base))
    for g in RELEVANT_GROUPS:
        ref = contrib[g] @ u[:3]
        np.testing.assert_allclose(attr[groups[g][:3]], ref, atol=1e-5)


def _as_dict(task, x):
    from projectv.envs.obs_layout import obs_dim

    n = obs_dim(task)
    return {"observation": x[:n], "achieved_goal": x[n : n + 3], "desired_goal": x[n + 3 :]}


def test_group_shares_sum_to_one():
    shares = group_shares({"a": -2.0, "b": 1.0, "c": 1.0})
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["a"] == pytest.approx(0.5)
