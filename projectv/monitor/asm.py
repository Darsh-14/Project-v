"""Attribution-Shift Monitor (mvp.md §7): Mahalanobis distance of the group-attribution share
vector z_t to its nominal distribution (Ledoit-Wolf shrinkage), fit on the calib split only.

Signal: z_t = IG group shares (16 steps, baseline = mean of the policy's calib-split nominal
states), over the semantic groups of `monitor_groups`:
- Push: the 10 groups of `xai.groups.make_groups("push")` (9 observation groups + goal;
  achieved_goal is merged into obj_pos).
- History policies (push_drhist): the per-slot groups ("obj_pos@t-2") are summed into their base
  group ("obj_pos"), and the 3 past actions form one extra group "act" (11 groups). The share
  vector therefore has the same meaning as for push_dr, plus the reliance on past actions.

CLI (the full C3 pipeline lives in `projectv.monitor.c3`):

    python -m projectv.monitor.asm --exp push_dr [--seeds 0 1 2] [--root results]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

from projectv.xai.attributions import integrated_gradients_batch
from projectv.xai.groups import base_name, make_groups


@dataclass(frozen=True)
class NominalModel:
    mean: np.ndarray
    precision: np.ndarray


def fit_nominal(z: np.ndarray, standardize: bool = False) -> NominalModel:
    """z: (n_samples, d) from calib-split nominal-dynamics episodes.

    standardize=True z-scores every feature (calib mean/std) before the Ledoit-Wolf fit and maps
    the result back to raw units. Ledoit-Wolf shrinks towards a multiple of the identity, which is
    not scale invariant: without standardization, features with small variance (velocities in
    metres per step) would be shrunk away in a raw-unit observation vector. Features that are
    (numerically) constant, std <= 1e-6 * max std, keep unit scale instead of being blown up.
    """
    z = np.asarray(z, dtype=float).reshape(-1, np.shape(z)[-1])
    if not standardize:
        lw = LedoitWolf().fit(z)
        return NominalModel(mean=lw.location_, precision=lw.precision_)
    mu, sd = z.mean(axis=0), z.std(axis=0)
    sd = np.where(sd > 1e-6 * max(sd.max(), 1e-12), sd, 1.0)
    lw = LedoitWolf().fit((z - mu) / sd)
    return NominalModel(mean=mu + sd * lw.location_, precision=lw.precision_ / np.outer(sd, sd))


def mahalanobis(model: NominalModel, z: np.ndarray) -> np.ndarray:
    """Distance for every vector along the last axis; leading axes are kept."""
    d = np.asarray(z, dtype=float) - model.mean
    sq = np.einsum("...i,ij,...j->...", d, model.precision, d)
    return np.sqrt(np.maximum(sq, 0.0))


def monitor_groups(
    task: str, history: bool = False, shortcut: bool = False
) -> dict[str, np.ndarray]:
    """Groups for z_t: `make_groups` with history slots merged into base groups and the past
    actions merged into one "act" group. Order follows first appearance in `make_groups`."""
    merged: dict[str, list[np.ndarray]] = {}
    for g, idx in make_groups(task, history, shortcut).items():
        merged.setdefault(base_name(g), []).append(idx)
    return {g: np.sort(np.concatenate(v)) for g, v in merged.items()}


def attribution_shares(
    pol,
    X: np.ndarray,
    base: np.ndarray,
    groups: dict[str, np.ndarray],
    steps: int = 16,
    batch: int = 512,
) -> np.ndarray:
    """IG group-share vectors z for every row of X (n, d) -> (n, len(groups)). A state whose total
    attribution is zero gets an all-zero share vector."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, np.shape(X)[-1])
    base = np.asarray(base, dtype=np.float64)
    out = np.empty((len(X), len(groups)))
    for i in range(0, len(X), batch):
        a = np.abs(integrated_gradients_batch(pol, X[i : i + batch], base, steps))
        g = np.stack([a[:, idx].sum(axis=1) for idx in groups.values()], axis=1)
        tot = g.sum(axis=1, keepdims=True)
        out[i : i + batch] = np.divide(g, tot, out=np.zeros_like(g), where=tot > 0)
    return out


def asm_scores(z: np.ndarray, model: NominalModel) -> np.ndarray:
    """m_t = Mahalanobis(z_t) for z of shape (..., G)."""
    return mahalanobis(model, z)


def main(argv: list[str] | None = None) -> None:
    from projectv.monitor import c3

    c3.main(argv)


if __name__ == "__main__":
    main()
