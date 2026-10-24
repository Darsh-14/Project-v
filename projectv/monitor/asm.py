"""Attribution-Shift Monitor (mvp.md §7): Mahalanobis distance of the group-attribution share
vector z_t to its nominal distribution (Ledoit-Wolf shrinkage), fit on the calib split only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class NominalModel:
    mean: np.ndarray
    precision: np.ndarray


def fit_nominal(z: np.ndarray) -> NominalModel:
    """z: (n_samples, d) from calib-split nominal-dynamics episodes."""
    z = np.asarray(z, dtype=float).reshape(-1, np.shape(z)[-1])
    lw = LedoitWolf().fit(z)
    return NominalModel(mean=lw.location_, precision=lw.precision_)


def mahalanobis(model: NominalModel, z: np.ndarray) -> np.ndarray:
    """Distance for every vector along the last axis; leading axes are kept."""
    d = np.asarray(z, dtype=float) - model.mean
    sq = np.einsum("...i,ij,...j->...", d, model.precision, d)
    return np.sqrt(np.maximum(sq, 0.0))
