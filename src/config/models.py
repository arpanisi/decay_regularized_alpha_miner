from __future__ import annotations

from pathlib import Path

import yaml


def load_model_defaults(path: str | Path = "config/cheap_models.yaml") -> tuple[str, str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    openrouter = payload["openrouter"]
    return openrouter["default_generation_model"], openrouter["default_alignment_model"]
