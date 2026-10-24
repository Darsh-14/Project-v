"""Semantic feature groups over the flat input vector used by policy_fn:

    x_flat = [observation (D), achieved_goal (3), desired_goal (3)]

`achieved_goal` is grouped with the observation slice it duplicates (CLAUDE.md §6). With
history (k=4 obs slots + 3 past actions) every group is split per time slot ("obj_pos@t-1"),
the current slot keeps the plain name, and past actions form groups "act@t-1".."act@t-3".
With the shortcut wrapper the appended hint is its own group.
"""

from __future__ import annotations

import numpy as np

from projectv.envs.obs_layout import ACHIEVED_GOAL_GROUP, layout, obs_dim

# Erased jointly as an extra unit in C1 (obj_rel = obj_pos - ee_pos is redundant).
REDUNDANT_CLUSTER = ("ee_pos", "obj_pos", "obj_rel")

HISTORY_K, HISTORY_ACTIONS, ACT_DIM = 4, 3, 4


def observation_dim(task: str, history: bool = False, shortcut: bool = False) -> int:
    d = obs_dim(task) + (3 if shortcut else 0)
    return HISTORY_K * d + HISTORY_ACTIONS * ACT_DIM if history else d


def flat_dim(task: str, history: bool = False, shortcut: bool = False) -> int:
    return observation_dim(task, history, shortcut) + 6


def make_groups(task: str, history: bool = False, shortcut: bool = False) -> dict[str, np.ndarray]:
    lay = layout(task)
    slot = obs_dim(task) + (3 if shortcut else 0)
    D = observation_dim(task, history, shortcut)
    n_slots = HISTORY_K if history else 1
    groups: dict[str, list[int]] = {}
    for k in range(n_slots):
        suffix = "" if k == 0 else f"@t-{k}"
        for name, sl in lay.items():
            groups[name + suffix] = list(range(k * slot + sl.start, k * slot + sl.stop))
        if shortcut:
            groups["hint" + suffix] = list(range(k * slot + slot - 3, k * slot + slot))
    if history:
        a0 = HISTORY_K * slot
        for j in range(HISTORY_ACTIONS):
            groups[f"act@t-{j + 1}"] = list(range(a0 + j * ACT_DIM, a0 + (j + 1) * ACT_DIM))
    groups[ACHIEVED_GOAL_GROUP[task]] += [D, D + 1, D + 2]
    groups["goal"] = [D + 3, D + 4, D + 5]
    out = {g: np.array(v) for g, v in groups.items()}
    _check_partition(out, D + 6)
    return out


def _check_partition(groups: dict[str, np.ndarray], d: int) -> None:
    allidx = np.sort(np.concatenate(list(groups.values())))
    assert np.array_equal(allidx, np.arange(d)), "groups must partition the flat vector"


def base_name(group: str) -> str:
    """'obj_pos@t-2' -> 'obj_pos'; 'act@t-1' -> 'act'."""
    return group.split("@")[0]


def slot_of(group: str) -> int:
    return int(group.split("@t-")[1]) if "@t-" in group else 0
