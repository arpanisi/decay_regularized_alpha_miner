from __future__ import annotations

from evaluation.gate import AlignmentResult
from evaluation.run import evaluate_candidate
from evaluation.seeds import SEED_FACTORS


def test_all_seed_factors_parse_evaluate_and_score(make_panel) -> None:
    for seed in SEED_FACTORS:
        decision, values, node_count = evaluate_candidate(
            seed.expression,
            make_panel,
            "2020-01-01",
            "2020-04-30",
            "2020-05-01",
            "2020-07-31",
            AlignmentResult(1.0, "synthetic seed alignment"),
        )
        assert node_count <= 50
        assert values.notna().sum() > 0
        assert "ic" in decision.train
        assert "coverage" in decision.validation
