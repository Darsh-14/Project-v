"""GWIE adapted to continuous Gaussian policies over feature groups (C2).

Two variants; every deviation from DePlag's original is listed in docs/gwie_original.md.

- `gwie_orig_port`: direct port of DePlag GWIE. Target S(s') = cos(mu(s'), mu(s)). The pivot is
  the group with the largest sum |IG| of S on the current erased state (30 right-endpoint steps,
  pool-mean baseline). Erase the pivot (marginal, M chains). Stop once S < alpha or after
  max_iter. Then one-pass pruning.
- `gwie_c`: same guided-erasure loop on the CLAUDE.md scalar target
  f(s') = <mu(s'), mu(s)/||mu(s)||> (midpoint IG). Runs over all groups and assigns each the
  marginal KL gain D(pi(s) || pi(s_after)) - D(pi(s) || pi(s_before)).

Chains: the current state is a batch of M copies of s. Chain m always takes erased values from
pool row m, so jointly erased groups keep their joint (pool) distribution. Every quantity is
averaged over chains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from projectv.xai.attributions import DT, gaussian_kl, group_shares, pool_rows


def _ig_groups(
    fn: Callable[[torch.Tensor], torch.Tensor],
    X: np.ndarray,
    base: np.ndarray,
    groups: dict[str, np.ndarray],
    candidates: list[str],
    steps: int,
    rule: str,
) -> dict[str, float]:
    """Sum over the group's features of |IG|, averaged over the rows (chains) of X."""
    k = np.arange(1, steps + 1) if rule == "right" else np.arange(steps) + 0.5
    alphas = k / steps
    path = base + alphas[:, None, None] * (X[None] - base)  # (steps, M, d)
    x = torch.as_tensor(path.reshape(-1, X.shape[1]), dtype=DT).requires_grad_(True)
    fn(x).sum().backward()
    g = x.grad.numpy().reshape(steps, *X.shape).mean(0)
    ig = np.abs(g * (X - base)).mean(0)  # (d,)
    return {c: float(ig[groups[c]].sum()) for c in candidates}


def _erase(X: np.ndarray, idx: np.ndarray, rows: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, idx] = rows[:, idx]
    return X


@dataclass
class GWIECResult:
    order: list[str]
    gains: dict[str, float]
    shares: dict[str, float]
    pivot_scores: list[float]
    divergence_trace: list[float]

    @property
    def total_divergence(self) -> float:
        return self.divergence_trace[-1]


def gwie_c(
    pol,
    s: np.ndarray,
    groups: dict[str, np.ndarray],
    pool: np.ndarray,
    rng: np.random.Generator,
    M: int = 8,
    ig_steps: int = 30,
) -> GWIECResult:
    s = np.asarray(s, dtype=np.float64)
    base = pool.mean(axis=0)
    rows = pool_rows(pool, M, rng)
    s_t = torch.as_tensor(s[None], dtype=DT)
    with torch.no_grad():
        m0, l0 = pol.dist(s_t)
    u = m0 / m0.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def f(x):
        return (pol.dist(x)[0] * u).sum(-1)

    def div(X):
        with torch.no_grad():
            m, ls = pol.dist(torch.as_tensor(X, dtype=DT))
        return float(gaussian_kl(m0, l0, m, ls).mean())

    cur = np.repeat(s[None], M, axis=0)
    remaining = list(groups)
    order, gains, pivots, trace = [], {}, [], [0.0]
    while remaining:
        scores = _ig_groups(f, cur, base, groups, remaining, ig_steps, "midpoint")
        g = max(remaining, key=lambda c: scores[c])
        nxt = _erase(cur, groups[g], rows)
        d_next = div(nxt)
        gains[g] = d_next - trace[-1]
        order.append(g)
        pivots.append(scores[g])
        trace.append(d_next)
        cur = nxt
        remaining.remove(g)
    return GWIECResult(order, gains, group_shares(gains), pivots, trace)


@dataclass
class GWIEPortResult:
    counterfactual_set: list[str]
    erasure_order: list[str]
    ranking: list[str]
    scores: dict[str, float]
    sims: list[float]
    converged: bool
    similarity_of_set: Callable[[list[str]], float] = field(repr=False)
    last_ig: dict[str, float] = field(default_factory=dict)  # pivot scores at the last iteration


def gwie_orig_port(
    pol,
    s: np.ndarray,
    groups: dict[str, np.ndarray],
    pool: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.75,
    max_iter: int = 20,
    M: int = 8,
    ig_steps: int = 30,
) -> GWIEPortResult:
    s = np.asarray(s, dtype=np.float64)
    base = pool.mean(axis=0)
    rows = pool_rows(pool, M, rng)
    with torch.no_grad():
        ref = pol.dist(torch.as_tensor(s[None], dtype=DT))[0]
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def S(x):
        m = pol.dist(x)[0]
        return (m * ref).sum(-1) / m.norm(dim=-1).clamp_min(1e-12)

    def sim(X):
        with torch.no_grad():
            return float(S(torch.as_tensor(X, dtype=DT)).mean())

    def similarity_of_set(gs: list[str]) -> float:
        X = np.repeat(s[None], M, axis=0)
        for g in gs:
            X = _erase(X, groups[g], rows)
        return sim(X)

    cur = np.repeat(s[None], M, axis=0)
    sims = [sim(cur)]
    chosen: list[str] = []
    drops: dict[str, float] = {}
    last_scores: dict[str, float] = {}
    converged = False
    for _ in range(max_iter):
        if sims[-1] < alpha:
            converged = True
            break
        remaining = [g for g in groups if g not in chosen]
        if not remaining:
            break
        last_scores = _ig_groups(S, cur, base, groups, remaining, ig_steps, "right")
        g = max(remaining, key=lambda c: last_scores[c])
        cur = _erase(cur, groups[g], rows)
        sims.append(sim(cur))
        drops[g] = sims[-2] - sims[-1]
        chosen.append(g)
    if sims[-1] < alpha:
        converged = True

    retained = list(chosen)
    if chosen:
        for g in chosen:
            candidate = [x for x in retained if x != g]
            if similarity_of_set(candidate) < alpha:
                retained = candidate
        converged = similarity_of_set(retained) < alpha

    rest = [g for g in groups if g not in chosen]
    rest.sort(key=lambda c: last_scores.get(c, 0.0), reverse=True)
    scores = {g: drops.get(g, 0.0) for g in groups}
    return GWIEPortResult(
        retained, chosen, chosen + rest, scores, sims, converged, similarity_of_set, last_scores
    )
