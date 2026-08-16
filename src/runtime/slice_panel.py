from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-instruments", type=int)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    dates = panel.index.get_level_values("datetime")
    sliced = panel.loc[(dates >= pd.Timestamp(args.start_date)) & (dates <= pd.Timestamp(args.end_date))]
    if args.max_instruments is not None:
        instruments = list(sliced.index.get_level_values("instrument").unique())[: args.max_instruments]
        sliced = sliced[sliced.index.get_level_values("instrument").isin(instruments)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_parquet(output)


if __name__ == "__main__":
    main()
