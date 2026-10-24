"""YAML config loading and hashing.

An experiment config (`configs/exp/*.yaml`) references an algo config and a DR config by path:

    task: push
    algo: configs/algo/sac_her_push.yaml
    dr: configs/dr/train.yaml
    history: false
    shortcut: false

`load_exp` resolves those references into one dict so a single hash covers everything that
determines a run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(_resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_exp(path: str | Path) -> dict[str, Any]:
    exp = load_yaml(path)
    exp["name"] = exp.get("name", Path(path).stem)
    exp["algo_cfg"] = load_yaml(exp["algo"])
    exp["dr_cfg"] = load_yaml(exp["dr"])
    return exp


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p
