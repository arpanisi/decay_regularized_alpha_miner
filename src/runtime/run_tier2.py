from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def main() -> None:
    full_panel = PROJECT / "outputs/panel/full_panel.parquet"
    tier_panel = PROJECT / "outputs/panel/tier2_panel.parquet"
    if not full_panel.exists():
        raise SystemExit("missing outputs/panel/full_panel.parquet; run the data build first")
    _run(
        [
            PYTHON,
            "src/runtime/slice_panel.py",
            "--panel",
            str(full_panel),
            "--output",
            str(tier_panel),
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2021-03-31",
        ]
    )
    _run(
        [
            PYTHON,
            "src/runtime/run_mining.py",
            "--panel",
            str(tier_panel),
            "--zoo",
            "outputs/zoo/tier2_zoo.json",
            "--values-dir",
            "outputs/zoo/tier2_values",
            "--log",
            "outputs/reports/tier2_mining_log.json",
            "--rounds",
            "5",
            "--train-start",
            "2020-01-01",
            "--train-end",
            "2020-12-31",
            "--validation-start",
            "2021-01-01",
            "--validation-end",
            "2021-03-31",
        ]
    )


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
