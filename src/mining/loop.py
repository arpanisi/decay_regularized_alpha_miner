from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import pandas as pd

from dsl.evaluator import evaluate
from dsl.parser import ParseError, parse_expression
from mining.llm import LLMClient, parse_json_object
from mining.prompt import build_grammar_prompt


@dataclass(frozen=True)
class CandidateProposal:
    rationale: str
    expression: str


@dataclass(frozen=True)
class CandidateResult:
    proposal: CandidateProposal | None
    parsed: bool
    evaluated: bool
    final_error: str | None
    attempts: int

    def to_dict(self) -> dict:
        return asdict(self)


class MiningLoop:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        candidates_per_round: int = 8,
        max_refinement_attempts: int = 5,
        max_nodes: int = 50,
    ):
        self.client = client
        self.model = model
        self.candidates_per_round = candidates_per_round
        self.max_refinement_attempts = max_refinement_attempts
        self.max_nodes = max_nodes

    def propose_round(self, panel_train: pd.DataFrame, feedback: list[str] | None = None) -> list[CandidateResult]:
        return [self._propose_one(panel_train, feedback or []) for _ in range(self.candidates_per_round)]

    def _propose_one(self, panel_train: pd.DataFrame, feedback: list[str]) -> CandidateResult:
        proposal: CandidateProposal | None = None
        error: str | None = None
        for attempt in range(1, self.max_refinement_attempts + 1):
            try:
                proposal = self._call_model(feedback, error)
                parsed = parse_expression(proposal.expression, max_nodes=self.max_nodes)
                values = evaluate(parsed, panel_train)
                if values.notna().sum() == 0:
                    error = "candidate evaluated to all-NaN over the training window"
                    continue
                return CandidateResult(proposal, parsed=True, evaluated=True, final_error=None, attempts=attempt)
            except (ParseError, KeyError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
        return CandidateResult(proposal, parsed=False, evaluated=False, final_error=error, attempts=self.max_refinement_attempts)

    def _call_model(self, feedback: list[str], error: str | None) -> CandidateProposal:
        messages = [
            {
                "role": "system",
                "content": build_grammar_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps({"prior_feedback": feedback, "last_error": error}),
            },
        ]
        response = self.client.complete(messages, self.model)
        payload = parse_json_object(response.content)
        if "rationale" not in payload or "expression" not in payload:
            raise ValueError("LLM response missing rationale or expression")
        return CandidateProposal(str(payload["rationale"]), str(payload["expression"]))
