from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.env import load_project_env
from data.connectors import fetch_companyfacts, fetch_sec_company_tickers
from data.sec_facts import parse_companyfacts


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True)
    parser.add_argument("--user-agent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tickers = [line.strip() for line in Path(args.universe).read_text(encoding="utf-8").splitlines() if line.strip()]
    cik_by_ticker = _cik_map(fetch_sec_company_tickers(args.user_agent))
    frames = []
    missing = []
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker.upper())
        if cik is None:
            missing.append(ticker)
            continue
        frames.append(parse_companyfacts(ticker, fetch_companyfacts(cik, args.user_agent)))
    facts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["instrument", "filed", "field", "value"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    facts.to_csv(output, index=False)
    if missing:
        output.with_suffix(".missing_tickers.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")


def _cik_map(payload: dict) -> dict[str, int]:
    return {str(item["ticker"]).upper(): int(item["cik_str"]) for item in payload.values()}


if __name__ == "__main__":
    main()
