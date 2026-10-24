"""Output helpers: run_meta.json and incremental JSON writes."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from projectv.utils.config import REPO_ROOT, config_hash

_TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "mujoco",
    "gymnasium",
    "gymnasium-robotics",
    "stable-baselines3",
    "sb3-contrib",
    "scikit-learn",
    "rliable",
)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    return versions


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def write_run_meta(
    out_dir: str | Path,
    cfg: dict[str, Any],
    argv: list[str],
    t_start: float,
    filename: str = "run_meta.json",
    **extra: Any,
) -> dict[str, Any]:
    meta = {
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "argv": argv,
        "versions": package_versions(),
        "wall_clock_s": round(time.time() - t_start, 2),
        **extra,
    }
    write_json(Path(out_dir) / filename, meta)
    return meta
