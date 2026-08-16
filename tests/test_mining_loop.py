from __future__ import annotations

import json

from dsl.parser import ALLOWED_FIELDS, OPERATOR_ARITY
from evaluation.seeds import SEED_FACTORS
from mining.llm import LLMResponse
from mining.loop import MiningLoop
from mining.prompt import build_grammar_prompt


class FakeClient:
    def __init__(self, responses: list[dict[str, str]]):
        self.responses = responses
        self.calls = 0
        self.system_prompt: str | None = None

    def complete(self, messages, model):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        self.system_prompt = messages[0]["content"]
        return LLMResponse(json.dumps(response), model)


def test_round_returns_exactly_configured_candidate_results(make_panel) -> None:
    client = FakeClient(
        [{"rationale": "recent price changes rank future returns", "expression": "CS_RANK(DELTA($close, 5))"}]
    )
    loop = MiningLoop(client, "model", candidates_per_round=8, max_refinement_attempts=5)
    results = loop.propose_round(make_panel)
    assert len(results) == 8
    assert all(result.evaluated for result in results)
    assert client.calls == 8


def test_refinement_stops_after_five_failures_and_preserves_error(make_panel) -> None:
    client = FakeClient([{"rationale": "bad field", "expression": "CS_RANK($label_fwd_10d)"}])
    loop = MiningLoop(client, "model", candidates_per_round=1, max_refinement_attempts=5)
    result = loop.propose_round(make_panel)[0]
    assert not result.evaluated
    assert result.attempts == 5
    assert "unknown field" in result.final_error
    assert client.calls == 5


def test_system_prompt_contains_full_grammar() -> None:
    """The mining prompt must state the actual grammar: every one of the 27 operators and at least
    the core price/volume $field names, generated from the parser's own tables (not hand-written
    prose). Otherwise the LLM guesses operator/field names and every candidate fails to parse."""
    prompt = build_grammar_prompt()

    assert len(OPERATOR_ARITY) == 27
    for op in OPERATOR_ARITY:
        assert op in prompt, f"operator {op} missing from prompt"
        assert f"{op}(" in prompt, f"operator {op} must appear with its signature"

    core_fields = ("open", "high", "low", "close", "volume", "market_cap", "dollar_volume", "ret")
    for field in core_fields:
        assert f"${field}" in prompt, f"$field {field} missing from prompt"

    # Every allowed field is documented, not just the core ones.
    assert all(f"${field}" in prompt for field in ALLOWED_FIELDS)


def test_system_prompt_sent_to_model_matches_constructed_grammar(make_panel) -> None:
    """The system message actually sent to the model is the full grammar, not a vague stub."""
    client = FakeClient(
        [{"rationale": "r", "expression": "CS_RANK(DELTA($close, 126))"}]
    )
    loop = MiningLoop(client, "model", candidates_per_round=1, max_refinement_attempts=1)
    loop.propose_round(make_panel)
    assert client.system_prompt == build_grammar_prompt()
    assert "TS_MEAN" in client.system_prompt
    assert "$funda_days_since_disclosure" in client.system_prompt


def test_candidate_proposed_with_real_grammar_parses_and_evaluates(make_panel) -> None:
    """A candidate written in the real grammar (the seed worked example that the prompt now shows
    the model) parses and evaluates successfully through the mining loop — confirming the prompt
    surfaces the actual grammar rather than leaving the model to guess syntax."""
    seed = SEED_FACTORS[0]  # Momentum: CS_RANK(DELTA($close, 126))
    client = FakeClient([{"rationale": seed.rationale, "expression": seed.expression}])
    loop = MiningLoop(client, "model", candidates_per_round=1, max_refinement_attempts=1)
    result = loop.propose_round(make_panel)[0]

    assert result.parsed
    assert result.evaluated
    assert result.final_error is None
    assert result.proposal.expression == seed.expression
    # The model could only have known this grammar because the prompt showed it the example.
    assert seed.expression in client.system_prompt
