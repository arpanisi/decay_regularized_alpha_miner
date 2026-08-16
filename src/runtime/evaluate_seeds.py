from __future__ import annotations

import argparse
import json
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
from evaluation.run import evaluate_candidate
from evaluation.metrics import compute_window_metrics
from evaluation.seeds import SEED_FACTORS
from mining.alignment import score_alignment
from mining.llm import OpenRouterClient


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alignment-model")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1/chat/completions")
    args = parser.parse_args()
    _, default_alignment_model = load_model_defaults()
    alignment_model = args.alignment_model or default_alignment_model
    panel = pd.read_parquet(args.panel)
    train_panel = _slice(panel, "2015-01-01", "2020-12-31")
    client = OpenRouterClient(args.base_url)
    results = []
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
        decision, _, node_count = evaluate_candidate(
            seed.expression,
            panel,
            "2015-01-01",
            "2020-12-31",
            "2021-01-01",
            "2023-12-31",
            alignment,
        )
        results.append({"seed": seed.name, "node_count": node_count, "decision": decision.to_dict()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _slice(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = panel.index.get_level_values("datetime")
    return panel.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


if __name__ == "__main__":
    main()
