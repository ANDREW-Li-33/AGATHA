"""Scenario taxonomy loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

GT_SOURCES = ("human", "derived", "stats")


@dataclass(frozen=True)
class Label:
    name: str
    gt: str
    definition: str


def load_taxonomy(path: str | Path | None = None) -> list[Label]:
    if path is None:
        text = resources.files("scenario_miner").joinpath("taxonomy.toml").read_text()
    else:
        text = Path(path).read_text()
    labels = [Label(**entry) for entry in tomllib.loads(text)["label"]]
    for label in labels:
        if label.gt not in GT_SOURCES:
            raise ValueError(f"{label.name}: gt must be one of {GT_SOURCES}, got {label.gt!r}")
    return labels


def label_names(taxonomy: list[Label], gt: str | None = None) -> list[str]:
    return [label.name for label in taxonomy if gt is None or label.gt == gt]
