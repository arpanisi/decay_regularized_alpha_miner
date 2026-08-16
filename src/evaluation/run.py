from __future__ import annotations

import pandas as pd

from dsl.evaluator import evaluate
from dsl.parser import parse_expression
from evaluation.gate import AlignmentResult, GateDecision, decide_gate
from evaluation.metrics import compute_window_metrics


def evaluate_candidate(
    expression: str,
    panel: pd.DataFrame,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    alignment: AlignmentResult,
    max_nodes: int = 50,
) -> tuple[GateDecision, pd.Series, int]:
    parsed = parse_expression(expression, max_nodes=max_nodes)
    values = evaluate(parsed, panel)
    train_mask = _date_mask(panel, train_start, train_end)
    validation_mask = _date_mask(panel, validation_start, validation_end)
    train_metrics = compute_window_metrics(values.loc[train_mask], panel.loc[train_mask])
    validation_metrics = compute_window_metrics(values.loc[validation_mask], panel.loc[validation_mask])
    decision = decide_gate(train_metrics, validation_metrics, alignment)
    return decision, values, parsed.node_count


def _date_mask(panel: pd.DataFrame, start: str, end: str) -> pd.Series:
    dates = panel.index.get_level_values("datetime")
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
