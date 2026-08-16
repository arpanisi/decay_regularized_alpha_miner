from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data.panel import build_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crsp-csv")
    parser.add_argument("--crsp-parquet")
    parser.add_argument("--facts-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.crsp_csv) == bool(args.crsp_parquet):
        raise SystemExit("provide exactly one of --crsp-csv or --crsp-parquet")
    crsp = pd.read_csv(args.crsp_csv) if args.crsp_csv else pd.read_parquet(args.crsp_parquet)
    panel = build_panel(crsp, pd.read_csv(args.facts_csv))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)


if __name__ == "__main__":
    main()
