import numpy as np
import pytest
import torch

from projectv.control.analytic_controller import RELEVANT_GROUPS, PDController
from projectv.xai.groups import flat_dim, make_groups
from projectv.xai.gwie_c import gwie_c, gwie_orig_port
from projectv.xai.policy_fn import LinearGaussian, controller_policy

TASK = "push"
GROUPS = make_groups(TASK)
D = flat_dim(TASK)


def _only_group_policy(group: str, seed: int = 0) -> LinearGaussian:
    rng = np.random.default_rng(seed)
    W = np.zeros((4, D))
    idx = GROUPS[group]
    W[:, idx] = rng.normal(size=(4, len(idx)))
    return LinearGaussian(torch.tensor(W), log_std=-1.0)


def _pool(n=2000, seed=1):
    return np.random.default_rng(seed).normal(size=(n, D))


def _state(group: str, offset: float = 3.0) -> np.ndarray:
    s = np.random.default_rng(2).normal(size=D)
    s[GROUPS[group]] += offset  # far from the pool, so erasing the group changes the action
    return s


@pytest.mark.parametrize("group", ["obj_rel", "goal", "ee_vel"])
def test_gwie_c_erases_only_relevant_group_first_with_095_share(group):
    pol = _only_group_policy(group)
    res = gwie_c(pol, _state(group), GROUPS, _pool(), np.random.default_rng(0))
    assert res.order[0] == group
    assert res.shares[group] >= 0.95
    assert set(res.order) == set(GROUPS)


def test_gwie_c_zero_share_on_controller_object_groups():
    pol = controller_policy(PDController(TASK), TASK)
    pool = _pool() * 0.05
    s = pool[5] + 0.3
    res = gwie_c(pol, s, GROUPS, pool, np.random.default_rng(0))
    for g in GROUPS:
        if g not in RELEVANT_GROUPS:
            assert res.shares[g] <= 1e-6
    assert set(res.order[: len(RELEVANT_GROUPS)]) == set(RELEVANT_GROUPS)


def test_gwie_c_gains_sum_to_total_divergence():
    pol = _only_group_policy("obj_pos")
    s = _state("obj_pos")
    res = gwie_c(pol, s, GROUPS, _pool(), np.random.default_rng(0))
    assert sum(res.gains.values()) == pytest.approx(res.total_divergence, rel=1e-9)


def test_gwie_c_deterministic_given_rng():
    pol = _only_group_policy("goal")
    a = gwie_c(pol, _state("goal"), GROUPS, _pool(), np.random.default_rng(7))
    b = gwie_c(pol, _state("goal"), GROUPS, _pool(), np.random.default_rng(7))
    assert a.order == b.order and a.gains == b.gains


def test_orig_port_finds_single_group_counterfactual():
    pol = _only_group_policy("obj_rel")
    res = gwie_orig_port(pol, _state("obj_rel"), GROUPS, _pool(), np.random.default_rng(0))
    assert res.converged
    assert res.counterfactual_set == ["obj_rel"]
    assert res.ranking[0] == "obj_rel" and len(res.ranking) == len(GROUPS)


@pytest.mark.parametrize("seed", range(5))
def test_orig_port_pruned_set_is_valid_subset(seed):
    # The original's one-pass pruning is not guaranteed 1-minimal (32/33 in the DePlag
    # verification), so the faithful port is only required to return a valid subset.
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(4, D)) * 0.3
    pol = LinearGaussian(torch.tensor(W), log_std=-1.0)
    s = rng.normal(size=D) * 2.0
    res = gwie_orig_port(pol, s, GROUPS, _pool(), np.random.default_rng(0), alpha=0.75)
    assert set(res.counterfactual_set) <= set(res.erasure_order)
    if res.converged:
        assert res.similarity_of_set(res.counterfactual_set) < 0.75
    assert res.sims[0] == pytest.approx(1.0)


def test_c1_scores_for_port_follow_its_ranking():
    from projectv.xai.attributions import attribute

    pol = _only_group_policy("obj_rel")
    s, pool = _state("obj_rel"), _pool()
    res = gwie_orig_port(pol, s, GROUPS, pool, np.random.default_rng(0))
    scores = attribute("gwie_orig_port", pol, s, GROUPS, pool, np.random.default_rng(0))
    nonzero = [g for g in res.ranking if scores[g] > 0]
    assert sorted(nonzero, key=lambda g: -scores[g]) == nonzero  # follows the ranking
    assert nonzero[0] == "obj_rel"
    # groups the policy ignores have zero IG at the last iteration -> score 0, not a rank
    assert all(scores[g] == 0.0 for g in GROUPS if g not in nonzero)
