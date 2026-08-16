from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def main() -> None:
    user_agent = os.environ.get("SEC_USER_AGENT", "decay-regularized-alpha-miner contact@example.com")
    _run(
        [
            PYTHON,
            "src/runtime/fetch_crsp.py",
            "--start-date",
            "2015-01-01",
            "--end-date",
            "2023-12-31",
            "--universe-size",
            "2750",
            "--output-crsp",
            "data/raw/crsp_dsf_v2.parquet",
            "--output-universe",
            "data/raw/universe.txt",
        ]
    )
    _run(
        [
            PYTHON,
            "src/runtime/fetch_sec_facts.py",
            "--universe",
            "data/raw/universe.txt",
            "--user-agent",
            user_agent,
            "--output",
            "data/raw/sec_facts.csv",
        ]
    )
    _run(
        [
            PYTHON,
            "src/runtime/build_panel.py",
            "--crsp-parquet",
            "data/raw/crsp_dsf_v2.parquet",
            "--facts-csv",
            "data/raw/sec_facts.csv",
            "--output",
            "outputs/panel/full_panel.parquet",
        ]
    )


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
