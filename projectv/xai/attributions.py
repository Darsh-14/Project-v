"""Attribution methods over semantic groups (CLAUDE.md §8). Pure functions.

Conventions:
- Policy: any object with `dist(x) -> (mean_pre_tanh, log_std)` over flat x (policy_fn).
- Scalar target for gradient methods: f(s') = <mu(s'), mu(s)/||mu(s)||>, mu pre-tanh.
- Divergence for erasure effects: KL(N(mu(s), sigma(s)) || N(mu(s'), sigma(s'))), diagonal.
- Erasure: `marginal` replaces a group's values with M pool rows (averaged over M);
  `mean` replaces them with the pool mean. The M rows are drawn once per call and shared by
  every erasure in that call (common random numbers), so differences between erasures are not
  sampling noise.
- Baseline for grad_x_delta / IG: pool mean ("marginal-mean"). Never zero.
- Group score = sum over the group's features of |per-feature attribution|.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import torch

METHODS = ("grad", "grad_x_delta", "ig", "smoothgrad", "occlusion", "kernelshap_grouped")
GWIE_METHODS = ("gwie_c", "gwie_orig_port")
ALL_METHODS = METHODS + GWIE_METHODS
DT = torch.float64


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=DT)


def _2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[None] if x.ndim == 1 else x


# --- core quantities ----------------------------------------------------------------------------
def target_direction(pol, S: np.ndarray) -> torch.Tensor:
    """u = mu(s) / ||mu(s)|| per row, (n, a)."""
    with torch.no_grad():
        mu, _ = pol.dist(_t(_2d(S)))
    return mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def f_target(pol, X: np.ndarray | torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    mu, _ = pol.dist(X if torch.is_tensor(X) else _t(X))
    return (mu * u).sum(-1)


def gradients(pol, X: np.ndarray, u: torch.Tensor) -> np.ndarray:
    """d f / d x at each row of X, with direction u (broadcast over rows)."""
    x = _t(_2d(X)).requires_grad_(True)
    f_target(pol, x, u).sum().backward()
    return x.grad.numpy()


def gaussian_kl(m1, ls1, m2, ls2) -> torch.Tensor:
    """KL(N(m1, e^ls1) || N(m2, e^ls2)), diagonal, summed over the last axis."""
    v1, v2 = torch.exp(2 * ls1), torch.exp(2 * ls2)
    return (ls2 - ls1 + (v1 + (m1 - m2) ** 2) / (2 * v2) - 0.5).sum(-1)


def kl_from(pol, s: np.ndarray, X: np.ndarray) -> np.ndarray:
    """KL(pi(.|s) || pi(.|x)) for each row x of X."""
    with torch.no_grad():
        m1, l1 = pol.dist(_t(_2d(s)))
        m2, l2 = pol.dist(_t(_2d(X)))
    return gaussian_kl(m1, l1, m2, l2).numpy()


def pool_rows(pool: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    return pool[rng.integers(len(pool), size=M)]


def erase(
    X: np.ndarray, idx: np.ndarray, rows: np.ndarray | None, pool_mean: np.ndarray
) -> np.ndarray:
    """Erase features `idx` in every row of X. rows=None -> mean erasure (n, d); otherwise
    marginal erasure with the given pool rows -> (n * M, d), M copies per input row."""
    X = _2d(X)
    if rows is None:
        out = X.copy()
        out[:, idx] = pool_mean[idx]
        return out
    out = np.repeat(X, len(rows), axis=0)
    out[:, idx] = np.tile(rows[:, idx], (len(X), 1))
    return out


# --- feature-level methods (return (d,) for one state) ----------------------------------------
def grad(pol, s: np.ndarray) -> np.ndarray:
    return gradients(pol, s, target_direction(pol, s))[0]


def grad_x_delta(pol, s: np.ndarray, base: np.ndarray) -> np.ndarray:
    return grad(pol, s) * (np.asarray(s) - base)


def integrated_gradients_batch(pol, S: np.ndarray, base: np.ndarray, steps: int = 16) -> np.ndarray:
    """IG with midpoint Riemann sum, baseline `base`, target direction from each state. (n, d)"""
    S = _2d(S)
    u = target_direction(pol, S)  # (n, a)
    alphas = (np.arange(steps) + 0.5) / steps
    path = base + alphas[:, None, None] * (S[None] - base)  # (steps, n, d)
    g = gradients(pol, path.reshape(-1, S.shape[1]), u.repeat(steps, 1))
    return g.reshape(steps, *S.shape).mean(0) * (S - base)


def integrated_gradients(pol, s: np.ndarray, base: np.ndarray, steps: int = 16) -> np.ndarray:
    return integrated_gradients_batch(pol, s, base, steps)[0]


def smoothgrad(
    pol, s: np.ndarray, std: np.ndarray, rng: np.random.Generator, n: int = 16, frac: float = 0.1
) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    noisy = s + rng.normal(size=(n, s.size)) * (frac * std)
    return gradients(pol, noisy, target_direction(pol, s)).mean(0)


# --- group-level methods (return {group: score}) ----------------------------------------------
def occlusion(pol, s, groups, pool, rng, M: int = 8, mode: str = "marginal") -> dict[str, float]:
    rows = pool_rows(pool, M, rng) if mode == "marginal" else None
    mean = pool.mean(axis=0)
    return {
        g: float(kl_from(pol, s, erase(s, idx, rows, mean)).mean()) for g, idx in groups.items()
    }


def kernelshap_grouped(
    pol, s, groups, pool, rng, M: int = 8, max_coalitions: int = 512, mode: str = "marginal"
) -> dict[str, float]:
    """Kernel SHAP over groups for v(S) = E_m f(s_S, pool_m,-S) (efficiency-constrained WLS).
    Enumerates all coalitions when 2^p - 2 <= max_coalitions, else samples by the Shapley
    kernel."""
    names = list(groups)
    p = len(names)
    rows = pool_rows(pool, M, rng) if mode == "marginal" else None
    mean = pool.mean(axis=0)
    u = target_direction(pol, s)

    def v(z: np.ndarray) -> np.ndarray:
        out = []
        for zi in z:
            off = [groups[names[j]] for j in range(p) if not zi[j]]
            idx = np.concatenate(off) if off else np.array([], dtype=int)
            X = erase(s, idx, rows, mean) if len(idx) else _2d(s)
            with torch.no_grad():
                out.append(float(f_target(pol, X, u).mean()))
        return np.array(out)

    if 2**p - 2 <= max_coalitions:
        Z = np.array(
            [[j in c for j in range(p)] for k in range(1, p) for c in combinations(range(p), k)]
        )
        k = Z.sum(1)
        w = (p - 1) / (np.array([comb(p, int(ki)) for ki in k]) * k * (p - k))
    else:
        ks = np.arange(1, p)
        pk = (p - 1) / (ks * (p - ks))
        sizes = rng.choice(ks, size=max_coalitions, p=pk / pk.sum())
        Z = np.zeros((max_coalitions, p), dtype=bool)
        for i, k in enumerate(sizes):
            Z[i, rng.choice(p, size=k, replace=False)] = True
        w = np.ones(len(Z))
    v0, vN = v(np.zeros((1, p), bool))[0], v(np.ones((1, p), bool))[0]
    y = v(Z) - v0
    Zf = Z.astype(np.float64)
    A = Zf[:, :-1] - Zf[:, -1:]
    b = y - Zf[:, -1] * (vN - v0)
    sw = np.sqrt(w)
    phi, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    phi = np.append(phi, (vN - v0) - phi.sum())
    return dict(zip(names, phi.tolist(), strict=True))


# --- dispatch ---------------------------------------------------------------------------------
def to_groups(attr: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, float]:
    return {g: float(np.abs(attr[idx]).sum()) for g, idx in groups.items()}


def group_shares(scores: dict[str, float]) -> dict[str, float]:
    tot = sum(abs(v) for v in scores.values())
    return {g: (abs(v) / tot if tot > 0 else 0.0) for g, v in scores.items()}


def attribute(
    method: str,
    pol,
    s: np.ndarray,
    groups: dict[str, np.ndarray],
    pool: np.ndarray,
    rng: np.random.Generator,
    **kw,
) -> dict[str, float]:
    """Unsigned group scores for one state."""
    s = np.asarray(s, dtype=np.float64)
    base = pool.mean(axis=0)
    if method == "grad":
        return to_groups(grad(pol, s), groups)
    if method == "grad_x_delta":
        return to_groups(grad_x_delta(pol, s, base), groups)
    if method == "ig":
        return to_groups(integrated_gradients(pol, s, base, kw.get("steps", 16)), groups)
    if method == "smoothgrad":
        return to_groups(smoothgrad(pol, s, pool.std(axis=0), rng, kw.get("n", 16)), groups)
    if method == "occlusion":
        return occlusion(pol, s, groups, pool, rng, kw.get("M", 8))
    if method == "kernelshap_grouped":
        phi = kernelshap_grouped(pol, s, groups, pool, rng, kw.get("M", 8))
        return {g: abs(v) for g, v in phi.items()}
    if method in GWIE_METHODS:
        from projectv.xai import gwie_c as gw

        if method == "gwie_c":
            return {
                g: abs(v)
                for g, v in gw.gwie_c(pol, s, groups, pool, rng, kw.get("M", 8)).gains.items()
            }
        # The port outputs a counterfactual set, not scores. C1 uses its ranking adapter
        # (erasure order, then not-erased groups by last IG score) as Borda points (n - rank)/n;
        # a never-erased group whose last IG score is exactly 0 gets 0 (a true tie, no evidence).
        res = gw.gwie_orig_port(pol, s, groups, pool, rng, M=kw.get("M", 8))
        n = len(res.ranking)
        erased = set(res.erasure_order)
        return {
            g: (n - i) / n if g in erased or res.last_ig.get(g, 0.0) > 0 else 0.0
            for i, g in enumerate(res.ranking)
        }
    raise ValueError(f"unknown method {method}")
