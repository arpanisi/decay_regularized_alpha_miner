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
from mining.alignment import score_alignment
from mining.llm import OpenRouterClient
from mining.loop import MiningLoop
from zoo.store import FactorZoo, ZooMember


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--zoo", required=True)
    parser.add_argument("--values-dir", required=True)
    parser.add_argument("--daily-ic-dir")
    parser.add_argument("--log", required=True)
    parser.add_argument("--generation-model")
    parser.add_argument("--alignment-model")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--candidates-per-round", type=int, default=8)
    parser.add_argument("--max-refinement-attempts", type=int, default=5)
    parser.add_argument("--train-start", default="2015-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-start", default="2021-01-01")
    parser.add_argument("--validation-end", default="2023-12-31")
    args = parser.parse_args()
    default_generation_model, default_alignment_model = load_model_defaults()
    generation_model = args.generation_model or default_generation_model
    alignment_model = args.alignment_model or default_alignment_model

    panel = pd.read_parquet(args.panel)
    train_panel = _slice(panel, args.train_start, args.train_end)
    validation_panel = _slice(panel, args.validation_start, args.validation_end)
    client = OpenRouterClient(args.base_url)
    loop = MiningLoop(
        client,
        generation_model,
        candidates_per_round=args.candidates_per_round,
        max_refinement_attempts=args.max_refinement_attempts,
    )
    zoo = FactorZoo.load(args.zoo)
    member_values = zoo.load_member_values()
    values_dir = Path(args.values_dir)
    values_dir.mkdir(parents=True, exist_ok=True)
    daily_ic_dir = Path(args.daily_ic_dir) if args.daily_ic_dir else values_dir.parent / "daily_ic"
    daily_ic_dir.mkdir(parents=True, exist_ok=True)
    feedback: list[str] = []
    records = []

    for round_idx in range(args.rounds):
        round_results = loop.propose_round(train_panel, feedback)
        for candidate_idx, result in enumerate(round_results):
            record = result.to_dict() | {"round": round_idx + 1, "candidate": candidate_idx + 1}
            if not result.proposal or not result.evaluated:
                records.append(record)
                continue
            try:
                parsed = parse_expression(result.proposal.expression)
                values = evaluate(parsed, panel)
                train_metrics = compute_window_metrics(values.loc[train_panel.index], train_panel)
                alignment = score_alignment(
                    client,
                    alignment_model,
                    result.proposal.rationale,
                    result.proposal.expression,
                    parsed,
                    train_metrics.ic,
                )
                decision, values, node_count = evaluate_candidate(
                    result.proposal.expression,
                    panel,
                    args.train_start,
                    args.train_end,
                    args.validation_start,
                    args.validation_end,
                    alignment,
                )
                validation_values = values.loc[validation_panel.index]
                stem = f"round_{round_idx + 1:03d}_candidate_{candidate_idx + 1:02d}_{_slug(result.proposal.rationale)[:40]}"
                train_daily_path, validation_daily_path = _write_daily_ic_series(
                    values,
                    train_panel,
                    validation_panel,
                    daily_ic_dir,
                    stem,
                )
                nearest = zoo.nearest(validation_values, member_values) if zoo.members else []
                max_corr = max((abs(corr) for _, corr in nearest), default=0.0)
                novelty_accepted = max_corr <= 0.8
                accepted_to_zoo = decision.accepted and novelty_accepted
                value_path = values_dir / f"{stem}.parquet"
                if accepted_to_zoo:
                    validation_values.rename("value").to_frame().to_parquet(value_path)
                    member = ZooMember(
                        f"round_{round_idx + 1}_candidate_{candidate_idx + 1}",
                        result.proposal.rationale,
                        result.proposal.expression,
                        node_count,
                        decision.to_dict(),
                        str(value_path),
                        str(train_daily_path),
                        str(validation_daily_path),
                    )
                    breadth = zoo.add(member, validation_values, member_values)
                    member_values[member.name] = validation_values
                    feedback = []
                else:
                    breadth = zoo.breadth()
                    if not novelty_accepted:
                        feedback = [f"Rejected as redundant with nearest zoo members: {nearest}"]
                    elif "alignment_score" in decision.failed_criteria:
                        feedback = [decision.alignment_justification]
                record.update(
                    {
                        "decision": decision.to_dict(),
                        "nearest": nearest,
                        "accepted_to_zoo": accepted_to_zoo,
                        "breadth_after_candidate": breadth,
                        "node_count": node_count,
                        "train_daily_ic_path": str(train_daily_path),
                        "validation_daily_ic_path": str(validation_daily_path),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "evaluated": False,
                        "error": str(exc),
                    }
                )
            records.append(record)
            zoo.save()
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            Path(args.log).write_text(json.dumps(records, indent=2), encoding="utf-8")


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
