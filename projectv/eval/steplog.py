"""Per-step log files (docs/interfaces.md): one compressed npz per (policy, cell)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

STEP_KEYS = ("obs_flat", "act_mean", "log_std", "action", "q_min", "penult", "is_success")


def steplog_path(root: str | Path, exp: str, seed: int, cell_id: int) -> Path:
    return Path(root) / "steplogs" / f"{exp}_s{seed}" / f"cell{cell_id:02d}.npz"


def save_steplog(path: str | Path, rec: dict[str, np.ndarray], cell_id: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [
        k for k in (*STEP_KEYS, "success", "dr_values", "dr_names", "ep_seed") if k not in rec
    ]
    assert not missing, f"steplog missing {missing}"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, cell_id=np.array(cell_id), **rec)
    tmp.replace(path)


def load_steplog(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
