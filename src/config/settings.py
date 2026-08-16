from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    @property
    def max_ast_nodes(self) -> int:
        return int(self.raw["grammar"]["max_ast_nodes"])

    @property
    def candidates_per_round(self) -> int:
        return int(self.raw["mining"]["candidates_per_round"])

    @property
    def max_refinement_attempts(self) -> int:
        return int(self.raw["mining"]["max_refinement_attempts"])


def load_settings(path: str | Path = "config/alpha_miner.yaml") -> Settings:
    with Path(path).open("r", encoding="utf-8") as fh:
        return Settings(yaml.safe_load(fh))
