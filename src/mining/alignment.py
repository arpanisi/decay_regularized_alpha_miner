from __future__ import annotations

import json

from dsl.ast import ParsedExpression
from evaluation.gate import AlignmentResult
from mining.llm import LLMClient

ALIGNMENT_TOOL_NAME = "score_expression_alignment"
ALIGNMENT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "alignment_score": {
            "type": "number",
            "description": "0 to 1; how well the expression mechanism implements its stated rationale",
        },
        "justification": {
            "type": "string",
            "description": "short reason grounded in the expression's mechanism",
        },
    },
    "required": ["alignment_score", "justification"],
}


def score_alignment(
    client: LLMClient,
    model: str,
    rationale: str,
    expression: str,
    parsed: ParsedExpression,
    train_ic: float,
) -> AlignmentResult:
    messages = [
        {
            "role": "system",
            "content": (
                "Score whether the expression mechanism implements its stated rationale, "
                "not whether the hypothesis is statistically correct."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "rationale": rationale,
                    "expression": expression,
                    "fields": sorted(parsed.fields),
                    "operators": sorted(parsed.operators),
                    "training_ic_sign_and_magnitude": train_ic,
                }
            ),
        },
    ]
    # Read the answer from the provider's native structured tool-calling field, never from
    # free-text content: a reasoning model writes draft JSON in its chain-of-thought before its
    # real final answer, and parsing that content would silently substitute a placeholder score.
    response = client.complete_structured(
        messages,
        model,
        tool_name=ALIGNMENT_TOOL_NAME,
        tool_schema=ALIGNMENT_TOOL_SCHEMA,
    )
    if not response.tool_arguments:
        raise ValueError("alignment response missing structured tool arguments")
    try:
        payload = json.loads(response.tool_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("alignment tool arguments were not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("alignment tool arguments JSON must be an object")
    return AlignmentResult(float(payload["alignment_score"]), str(payload["justification"]))
