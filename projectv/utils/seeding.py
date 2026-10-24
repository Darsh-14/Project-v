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
    """base_seed = 10_000 * train_seed + cell_id * 1_000 + ep (CLAUDE.md §7; requires ep < 1000)."""
    assert 0 <= ep < 1_000, ep
    return 10_000 * train_seed + cell_id * 1_000 + ep
