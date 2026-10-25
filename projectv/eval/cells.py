"""Held-out cells (docs/interfaces.md). `heldout.yaml` is written only by
scripts/calibrate_heldout.py and is frozen afterwards (CLAUDE.md §2.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from projectv.envs.dr_wrapper import fixed_spec
from projectv.utils.config import load_yaml, text_hash

HELDOUT_PATH = "configs/dr/heldout.yaml"
HASH_PATH = "results/heldout_hash.txt"


@dataclass(frozen=True)
class Cell:
    id: int
    name: str
    factor: str
    params: dict = field(default_factory=dict)

    def dr_spec(self) -> dict[str, dict]:
        nominal = {k: v["value"] for k, v in load_yaml("configs/dr/none.yaml").items()}
        return fixed_spec({**nominal, **self.params})


NOMINAL = Cell(0, "nominal", "nominal", {})


def load_cells(path: str | Path = HELDOUT_PATH, check_hash: bool = True) -> list[Cell]:
    if check_hash:
        assert_frozen(path)
    raw = load_yaml(path)
    cells = [
        Cell(int(c["id"]), c["name"], c["factor"], dict(c.get("params", {}))) for c in raw["cells"]
    ]
    assert cells[0].id == 0 and cells[0].factor == "nominal", "cell 0 must be nominal"
    assert len({c.id for c in cells}) == len(cells), "duplicate cell ids"
    return cells


def write_cells(cells: list[Cell], path: str | Path, **header) -> None:
    doc = {
        **header,
        "cells": [dict(id=c.id, name=c.name, factor=c.factor, params=c.params) for c in cells],
    }
    Path(path).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8", newline="\n")


def assert_frozen(path: str | Path = HELDOUT_PATH, hash_path: str | Path = HASH_PATH) -> None:
    recorded = Path(hash_path).read_text().strip()
    actual = text_hash(path)
    assert recorded == actual, f"{path} changed after freeze ({actual} != {recorded})"
