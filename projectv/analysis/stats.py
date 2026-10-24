"""Bootstrap helpers (CLAUDE.md §7)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap over the first axis. Returns (point, lower, upper)."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(values), size=(n_boot, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(values)), float(lo), float(hi)


def two_level_bootstrap_ci(
    stat: Callable[[list[dict[str, np.ndarray]]], float | np.ndarray],
    clusters: Sequence[dict[str, np.ndarray]],
    axes: dict[str, int] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Two-level percentile bootstrap (CLAUDE.md §7): resample clusters (training seeds) with
    replacement, then, inside each sampled cluster, resample the units of every array.

    `clusters[c]` maps names to arrays. Each array is resampled along `axes[name]` (default 0);
    the arrays of one cluster are resampled independently of each other, so put quantities that
    must stay paired (e.g. success of the same episodes under different erasures) into one array.
    `stat(list_of_clusters)` may return a scalar or an array; the CI is elementwise and ignores
    NaN replicates. Returns (point, lower, upper) with the shape of `stat`'s output.
    """
    axes = axes or {}
    clusters = list(clusters)
    rng = np.random.default_rng(seed)
    point = np.asarray(stat(clusters), dtype=np.float64)

    def resample(c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for k, v in c.items():
            ax = axes.get(k, 0)
            out[k] = np.take(v, rng.integers(v.shape[ax], size=v.shape[ax]), axis=ax)
        return out

    boots = []
    for _ in range(n_boot):
        picked = rng.integers(len(clusters), size=len(clusters))
        boots.append(np.asarray(stat([resample(clusters[i]) for i in picked]), dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(np.array(boots), [alpha / 2, 1 - alpha / 2], axis=0)
    if point.ndim == 0:
        return float(point), float(lo), float(hi)
    return point, lo, hi
