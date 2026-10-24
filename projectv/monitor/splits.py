"""Deterministic calib/test split for C3 (CLAUDE.md §7, docs/interfaces.md).

Each episode is keyed by a sha256 hash of its reset seed `ep_seed`. Within a cell the episodes are
ordered by that key and the first half (rounded down) is `calib`, the rest `test`. The assignment
therefore depends only on the set of seeds in the cell, never on the episode order or on any
result, and is exactly 50/50 (+-1) per cell. Nothing may be fit on `test`.
"""

from __future__ import annotations

import hashlib

import numpy as np

CALIB, TEST = "calib", "test"
_SALT = b"projectv-c3-split-v1:"


def split_key(ep_seed: int) -> int:
    return int.from_bytes(hashlib.sha256(_SALT + str(int(ep_seed)).encode()).digest()[:8], "big")


def calib_mask(ep_seed: np.ndarray, cell_id: np.ndarray | None = None) -> np.ndarray:
    """Boolean (E,) mask, True = calib. `cell_id` (E,) splits each cell separately; None treats
    all episodes as one cell."""
    ep_seed = np.asarray(ep_seed).ravel()
    cell_id = np.zeros(len(ep_seed), int) if cell_id is None else np.asarray(cell_id).ravel()
    assert len(cell_id) == len(ep_seed)
    mask = np.zeros(len(ep_seed), bool)
    for c in np.unique(cell_id):
        idx = np.flatnonzero(cell_id == c)
        seeds = ep_seed[idx]
        assert len(np.unique(seeds)) == len(seeds), f"duplicate ep_seed in cell {c}"
        order = sorted(range(len(idx)), key=lambda i: (split_key(seeds[i]), int(seeds[i])))
        mask[idx[order[: len(idx) // 2]]] = True
    return mask


def assign_splits(ep_seed: np.ndarray, cell_id: np.ndarray | None = None) -> np.ndarray:
    return np.where(calib_mask(ep_seed, cell_id), CALIB, TEST)
