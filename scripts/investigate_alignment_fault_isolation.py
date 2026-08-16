"""
Direct reproduction: a missing/invalid structured tool-call response during
alignment scoring (src.mining.alignment.score_alignment) must not abort the
mining round -- the candidate should be recorded as a failure and the round
must continue to the next candidate, exactly the per-candidate try/except
structure src/runtime/run_mining.py's main loop uses (lines ~76-149).

Uses the real tier1_panel.parquet, the real seed-factor expressions, and the
real score_alignment / evaluate_candidate functions -- unmodified. Only the
LLMClient passed to score_alignment is a local fake, and only for the
deliberately-triggered failure case; this fake conforms to the same
LLMClient Protocol OpenRouterClient implements and returns exactly what a
real provider response looks like when tool_calls is empty (which is what
OpenRouterClient.complete_structured itself returns as tool_arguments="" in
that case -- see src/mining/llm.py lines 102-107).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from dsl.parser import parse_expression
from evaluation.run import evaluate_candidate
from evaluation.seeds import SEED_FACTORS
from mining.alignment import score_alignment
from mining.llm import LLMResponse


class MissingToolCallClient:
    """Simulates a provider response with no tool_calls entry for the
    requested tool -- the real OpenRouterClient.complete_structured returns
    exactly this shape (tool_arguments="") when that happens."""

    def complete_structured(self, messages, model, *, tool_name, tool_schema):
        return LLMResponse(content="(model wrote free text instead of calling the tool)", model=model, tool_arguments="")


class InvalidJsonToolCallClient:
    """Simulates a provider response whose tool_calls[].function.arguments
    field is present but not valid JSON."""

    def complete_structured(self, messages, model, *, tool_name, tool_schema):
        return LLMResponse(content="", model=model, tool_arguments="{not valid json")


class GoodClient:
    """A real-shaped successful structured response."""

    def complete_structured(self, messages, model, *, tool_name, tool_schema):
        return LLMResponse(
            content="",
            model=model,
            tool_arguments='{"alignment_score": 0.9, "justification": "expression references the fields the rationale claims"}',
        )


def run_one_candidate(name, expression, rationale, client, panel, train_panel):
    """Mirrors run_mining.py's per-candidate try/except exactly (lines 76-149):
    score_alignment is called inside the try, and any exception it raises is
    caught by the same broad `except Exception` the real loop uses, recorded
    into a result dict, and the caller moves on -- it never propagates out
    and aborts the round."""
    record = {"candidate": name}
    try:
        parsed = parse_expression(expression)
        from dsl.evaluator import evaluate
        from evaluation.metrics import compute_window_metrics

        values = evaluate(parsed, panel)
        train_metrics = compute_window_metrics(values.loc[train_panel.index], train_panel)
        alignment = score_alignment(client, "fake-model", rationale, expression, parsed, train_metrics.ic)
        decision, values, node_count = evaluate_candidate(
            expression, panel, "2020-01-02", "2020-01-21", "2020-01-22", "2020-01-24", alignment,
        )
        record.update({"evaluated": True, "accepted": decision.accepted, "alignment_score": decision.alignment_score})
    except Exception as exc:  # noqa: BLE001 -- same broad catch run_mining.py's loop uses
        record.update({"evaluated": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def main():
    panel = pd.read_parquet(ROOT / "outputs/panel/tier1_panel.parquet")
    dates = panel.index.get_level_values("datetime")
    train_panel = panel.loc[(dates >= pd.Timestamp("2020-01-02")) & (dates <= pd.Timestamp("2020-01-21"))]

    candidates = [
        (SEED_FACTORS[0].name, SEED_FACTORS[0].expression, SEED_FACTORS[0].rationale, GoodClient()),
        ("Momentum (forced missing tool_calls)", SEED_FACTORS[0].expression, SEED_FACTORS[0].rationale, MissingToolCallClient()),
        (SEED_FACTORS[1].name, SEED_FACTORS[1].expression, SEED_FACTORS[1].rationale, GoodClient()),
        ("Reversal (forced invalid JSON tool_arguments)", SEED_FACTORS[1].expression, SEED_FACTORS[1].rationale, InvalidJsonToolCallClient()),
        (SEED_FACTORS[2].name, SEED_FACTORS[2].expression, SEED_FACTORS[2].rationale, GoodClient()),
    ]

    print("=== Simulated round: 5 candidates, 2 deliberately broken tool-call responses ===")
    records = []
    for name, expr, rationale, client in candidates:
        rec = run_one_candidate(name, expr, rationale, client, panel, train_panel)
        records.append(rec)
        print(f"{name}: evaluated={rec['evaluated']}"
              + (f" accepted={rec.get('accepted')} alignment_score={rec.get('alignment_score')}" if rec["evaluated"] else f" error={rec.get('error')}"))

    print("\n=== Round completed without aborting. Final tally ===")
    print(f"total candidates processed: {len(records)} (expected 5, matching the input list)")
    print(f"evaluated=True: {sum(r['evaluated'] for r in records)}")
    print(f"evaluated=False (recorded failures): {sum(not r['evaluated'] for r in records)}")
    assert len(records) == 5, "the round must process every candidate, not stop at the first failure"
    assert records[1]["evaluated"] is False and "alignment response missing structured tool arguments" in records[1]["error"]
    assert records[3]["evaluated"] is False and "not valid JSON" in records[3]["error"]
    assert records[0]["evaluated"] is True and records[2]["evaluated"] is True and records[4]["evaluated"] is True
    print("\nCONFIRMED: candidates before and after each forced tool-call failure were still fully evaluated -- the round continued past both failures instead of aborting.")


if __name__ == "__main__":
    main()
