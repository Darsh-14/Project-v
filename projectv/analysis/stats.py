"""Bootstrap helpers (CLAUDE.md §7)."""

from __future__ import annotations

from collections.abc import Callable

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
