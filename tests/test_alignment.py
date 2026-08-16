from __future__ import annotations

import json

import pytest

from evaluation.gate import AlignmentResult
from mining.alignment import ALIGNMENT_TOOL_NAME, ALIGNMENT_TOOL_SCHEMA, score_alignment
from mining.llm import LLMResponse


class StructuredFakeClient:
    """Captures the tool-calling request and returns a response whose free-text `content`
    deliberately contains reasoning-model draft JSON, while the real answer lives only in the
    provider's native `tool_arguments` field — exactly the bug scenario the fix targets."""

    def __init__(self, *, content: str, tool_arguments: str):
        self.content = content
        self.tool_arguments = tool_arguments
        self.calls = 0
        self.sent_messages: list[dict[str, str]] | None = None
        self.sent_tool_name: str | None = None
        self.sent_tool_schema: dict | None = None

    def complete(self, messages, model):
        raise AssertionError(
            "alignment must go through the native structured tool-calling path, "
            "never free-text completion"
        )

    def complete_structured(self, messages, model, *, tool_name, tool_schema):
        self.calls += 1
        self.sent_messages = messages
        self.sent_tool_name = tool_name
        self.sent_tool_schema = tool_schema
        return LLMResponse(content=self.content, model=model, tool_arguments=self.tool_arguments)


def _parsed():
    from dsl.parser import parse_expression

    return parse_expression("CS_RANK(DELTA($close, 126))")


def test_score_alignment_reads_structured_field_not_draft_content() -> None:
    """Regression for the reasoning-model draft-JSON bug: a synthetic response whose content
    contains a `<think>...` draft `{"alignment_score": 0.9, "justification": "..."}` block is
    unaffected, because the score comes from the provider's tool_arguments field, never from
    free-text content."""
    draft_content = (
        "I need to think about how to respond. I will return JSON shaped like "
        '{"alignment_score": 0.9, "justification": "..."} to start. '
        "</think>\n"
        "The real final answer follows."
    )
    client = StructuredFakeClient(
        content=draft_content,
        tool_arguments=json.dumps(
            {"alignment_score": 0.6, "justification": "the genuine final justification"}
        ),
    )

    result = score_alignment(
        client,
        "qwen/qwen3.5-flash-02-23",
        rationale="six-month price momentum, cross-sectionally ranked",
        expression="CS_RANK(DELTA($close, 126))",
        parsed=_parsed(),
        train_ic=0.02,
    )

    assert isinstance(result, AlignmentResult)
    # The draft score (0.9) in content must not leak through; the structured field wins.
    assert result.score == 0.6
    assert result.justification == "the genuine final justification"
    assert client.calls == 1


def test_score_alignment_registers_json_schema_tool() -> None:
    client = StructuredFakeClient(
        content="",
        tool_arguments=json.dumps({"alignment_score": 0.7, "justification": "j"}),
    )
    score_alignment(client, "m", "r", "CS_RANK(DELTA($close, 5))", _parsed(), 0.01)

    assert client.sent_tool_name == ALIGNMENT_TOOL_NAME
    assert client.sent_tool_schema == ALIGNMENT_TOOL_SCHEMA
    assert client.sent_tool_schema["type"] == "object"
    assert set(client.sent_tool_schema["properties"]) == {"alignment_score", "justification"}
    assert client.sent_tool_schema["required"] == ["alignment_score", "justification"]
    assert client.sent_tool_schema["properties"]["alignment_score"]["type"] == "number"
    # The messages passed through still carry the candidate's rationale/expression context.
    assert "rationale" in client.sent_messages[1]["content"]


def test_score_alignment_fails_loudly_when_tool_arguments_missing() -> None:
    """A response with only free-text draft JSON (no tool_arguments) is a loud failure, never a
    silent parse of the draft."""
    client = StructuredFakeClient(content='draft {"alignment_score": 0.9, "justification": "..."} only', tool_arguments="")
    with pytest.raises(ValueError):
        score_alignment(client, "m", "r", "CS_RANK(DELTA($close, 5))", _parsed(), 0.01)
