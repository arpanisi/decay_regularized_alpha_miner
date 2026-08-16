from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.backtest_validation import compute_backtest_validation_report
from zoo.store import FactorZoo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoo", required=True)
    parser.add_argument("--mining-log", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    zoo = FactorZoo.load(args.zoo)
    records = json.loads(Path(args.mining_log).read_text(encoding="utf-8"))
    report = compute_backtest_validation_report(zoo, records)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
