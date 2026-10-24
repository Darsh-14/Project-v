"""C3 baseline monitors (mvp.md §7). Each returns a raw per-step score m of shape (E, T) from a
per-step log (docs/interfaces.md); higher = more anomalous. All monitors, ASM included, then get
the same EWMA + prefix aggregation (`metrics.ewma`, `metrics.prefix_score`).

- obs_ood:  Mahalanobis distance of `obs_flat` to the calib-split nominal states.
- act_std:  mean over action dims of the actor std exp(log_std).
- q_drop:   -q_min (min-critic Q of the issued action).
- ensemble: std across the seed actors' pre-tanh means on the same states, averaged over action
            dims (the evaluated policy is one of the members).
- act_ood:  Mahalanobis distance of the actor penultimate activations `penult`.

Nominal models are fit with `asm.fit_nominal(standardize=True)` (Ledoit-Wolf on z-scored
features), on calib-split nominal (cell 0) episodes only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from projectv.monitor.asm import NominalModel, fit_nominal, mahalanobis

SIGNALS = ("asm", "obs_ood", "act_std", "q_drop", "ensemble", "act_ood")
BASELINES = SIGNALS[1:]


def fit_feature_model(X: np.ndarray, calib: np.ndarray, standardize: bool = True) -> NominalModel:
    """X: (E, T, d) per-step features of nominal episodes; calib: (E,) bool. Only calib rows are
    used."""
    X = np.asarray(X)[np.asarray(calib, bool)]
    assert len(X), "no calib episodes to fit the nominal model"
    return fit_nominal(X.reshape(-1, X.shape[-1]), standardize=standardize)


def obs_ood(log: dict[str, np.ndarray], model: NominalModel) -> np.ndarray:
    return mahalanobis(model, log["obs_flat"])


def act_ood(log: dict[str, np.ndarray], model: NominalModel) -> np.ndarray:
    return mahalanobis(model, log["penult"])


def act_std(log: dict[str, np.ndarray]) -> np.ndarray:
    return np.exp(np.asarray(log["log_std"], dtype=np.float64)).mean(axis=-1)


def q_drop(log: dict[str, np.ndarray]) -> np.ndarray:
    return -np.asarray(log["q_min"], dtype=np.float64)


def ensemble_std(pols: Sequence, obs_flat: np.ndarray, batch: int = 4096) -> np.ndarray:
    """pols: objects with `dist(x) -> (mean_pre_tanh, log_std)`; obs_flat (E, T, d) -> (E, T)."""
    assert len(pols) >= 2, "ensemble needs at least two policies"
    lead = np.shape(obs_flat)[:-1]
    X = np.asarray(obs_flat, dtype=np.float64).reshape(-1, np.shape(obs_flat)[-1])
    out = np.empty(len(X))
    for i in range(0, len(X), batch):
        xt = torch.as_tensor(X[i : i + batch], dtype=torch.float64)
        with torch.no_grad():
            means = torch.stack([p.dist(xt)[0].to(torch.float64) for p in pols])
        out[i : i + batch] = means.std(dim=0, unbiased=False).mean(dim=-1).numpy()
    return out.reshape(lead)
