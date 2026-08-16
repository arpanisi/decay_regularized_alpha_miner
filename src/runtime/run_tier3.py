from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def main() -> None:
    full_panel = PROJECT / "outputs/panel/full_panel.parquet"
    if not full_panel.exists():
        raise SystemExit("missing outputs/panel/full_panel.parquet; run the data build first")
    _run(
        [
            PYTHON,
            "src/runtime/run_mining.py",
            "--panel",
            str(full_panel),
            "--zoo",
            "outputs/zoo/zoo.json",
            "--values-dir",
            "outputs/zoo/values",
            "--log",
            "outputs/reports/full_mining_log.json",
            "--rounds",
            "50",
        ]
    )
    _run(
        [
            PYTHON,
            "src/runtime/run_backtest_validation.py",
            "--zoo",
            "outputs/zoo/zoo.json",
            "--mining-log",
            "outputs/reports/full_mining_log.json",
            "--report",
            "outputs/reports/backtest_validation.json",
        ]
    )


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
