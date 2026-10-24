"""Seeding helpers. Eval episode seeds follow CLAUDE.md §7."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def eval_episode_seed(train_seed: int, cell_id: int, ep: int) -> int:
    """base_seed = 100_000 * train_seed + cell_id * 1_000 + ep (requires ep < 1000, cell < 100).

    CLAUDE.md §7 specifies a 10_000 stride per training seed, which repeats reset seeds across
    (training seed, cell) pairs once cell_id >= 10 (the held-out grid has ~38 cells), making
    "independent" seed clusters share initial states and noise streams. The stride is 100_000
    so every (train_seed, cell, ep) maps to a unique seed (CHANGELOG 2026-10-25)."""
    assert 0 <= ep < 1_000, ep
    assert 0 <= cell_id < 100, cell_id
    return 100_000 * train_seed + cell_id * 1_000 + ep
