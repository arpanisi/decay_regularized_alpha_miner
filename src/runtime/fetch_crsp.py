from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import load_project_env
from data.connectors import fetch_crsp_dsf_v2
from data.panel import select_static_universe


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--universe-size", type=int, default=2750)
    parser.add_argument("--output-crsp", required=True)
    parser.add_argument("--output-universe", required=True)
    args = parser.parse_args()

    crsp = fetch_crsp_dsf_v2(args.start_date, args.end_date)
    universe = select_static_universe(crsp, args.start_date, args.end_date, args.universe_size)
    crsp = crsp[crsp["ticker"].astype(str).isin(universe)]

    output_crsp = Path(args.output_crsp)
    output_crsp.parent.mkdir(parents=True, exist_ok=True)
    crsp.to_parquet(output_crsp)
    Path(args.output_universe).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_universe).write_text("\n".join(universe) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
