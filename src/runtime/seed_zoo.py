from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.env import load_project_env
from config.models import load_model_defaults
from dsl.evaluator import evaluate
from dsl.parser import parse_expression
from evaluation.metrics import compute_daily_metric_series, compute_window_metrics
from evaluation.run import evaluate_candidate
from evaluation.seeds import SEED_FACTORS
from mining.alignment import score_alignment
from mining.llm import OpenRouterClient
from zoo.store import FactorZoo, ZooMember


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--zoo", required=True)
    parser.add_argument("--values-dir", required=True)
    parser.add_argument("--daily-ic-dir")
    parser.add_argument("--report", required=True)
    parser.add_argument("--alignment-model")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1/chat/completions")
    args = parser.parse_args()
    _, default_alignment_model = load_model_defaults()
    alignment_model = args.alignment_model or default_alignment_model

    panel = pd.read_parquet(args.panel)
    train_panel = _slice(panel, "2015-01-01", "2020-12-31")
    zoo = FactorZoo(args.zoo)
    member_values: dict[str, pd.Series] = {}
    report = []
    validation_panel = _slice(panel, "2021-01-01", "2023-12-31")
    values_dir = Path(args.values_dir)
    values_dir.mkdir(parents=True, exist_ok=True)
    daily_ic_dir = Path(args.daily_ic_dir) if args.daily_ic_dir else values_dir.parent / "daily_ic"
    daily_ic_dir.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient(args.base_url)

    for seed in SEED_FACTORS:
        parsed = parse_expression(seed.expression)
        values = evaluate(parsed, panel)
        train_metrics = compute_window_metrics(values.loc[train_panel.index], train_panel)
        alignment = score_alignment(
            client,
            alignment_model,
            seed.rationale,
            seed.expression,
            parsed,
            train_metrics.ic,
        )
        decision, values, node_count = evaluate_candidate(
            seed.expression,
            panel,
            "2015-01-01",
            "2020-12-31",
            "2021-01-01",
            "2023-12-31",
            alignment,
        )
        validation_values = values.loc[validation_panel.index]
        train_daily_path, validation_daily_path = _write_daily_ic_series(
            values,
            train_panel,
            validation_panel,
            daily_ic_dir,
            _slug(seed.name),
        )
        nearest = zoo.nearest(validation_values, member_values) if zoo.members else []
        max_corr = max((abs(corr) for _, corr in nearest), default=0.0)
        accepted_to_zoo = decision.accepted and max_corr <= 0.8
        value_path = values_dir / f"{_slug(seed.name)}.parquet"
        if accepted_to_zoo:
            validation_values.rename("value").to_frame().to_parquet(value_path)
            member = ZooMember(
                seed.name,
                seed.rationale,
                seed.expression,
                node_count,
                decision.to_dict(),
                str(value_path),
                str(train_daily_path),
                str(validation_daily_path),
            )
            breadth = zoo.add(member, validation_values, member_values)
            member_values[seed.name] = validation_values
        else:
            breadth = zoo.breadth()
        report.append(
            {
                "seed": seed.name,
                "decision": decision.to_dict(),
                "node_count": node_count,
                "nearest": nearest,
                "accepted_to_zoo": accepted_to_zoo,
                "breadth_after_candidate": breadth,
                "train_daily_ic_path": str(train_daily_path),
                "validation_daily_ic_path": str(validation_daily_path),
            }
        )

    zoo.save()
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _slice(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = panel.index.get_level_values("datetime")
    return panel.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _write_daily_ic_series(
    values: pd.Series,
    train_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    train_daily = compute_daily_metric_series(values.loc[train_panel.index], train_panel).daily_ic
    validation_daily = compute_daily_metric_series(values.loc[validation_panel.index], validation_panel).daily_ic
    train_path = output_dir / f"{stem}_train_daily_ic.parquet"
    validation_path = output_dir / f"{stem}_validation_daily_ic.parquet"
    train_daily.rename("daily_ic").to_frame().to_parquet(train_path)
    validation_daily.rename("daily_ic").to_frame().to_parquet(validation_path)
    return train_path, validation_path


if __name__ == "__main__":
    main()
