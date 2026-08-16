from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import requests


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    # Structured answer from the provider's native tool-calling field. When a caller uses
    # complete_structured, the free-text `content` (which may contain a reasoning model's
    # chain-of-thought, including draft JSON) is never consulted; only this field is used.
    tool_arguments: str | None = None


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], model: str) -> LLMResponse:
        ...

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        tool_name: str,
        tool_schema: dict,
    ) -> LLMResponse:
        ...


class OpenRouterClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")

    def complete(self, messages: list[dict[str, str]], model: str) -> LLMResponse:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({"model": model, "messages": messages}),
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        return LLMResponse(payload["choices"][0]["message"]["content"], model)

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        tool_name: str,
        tool_schema: dict,
    ) -> LLMResponse:
        """Native structured-output/tool-calling completion.

        The structured answer is returned by the provider in
        message.tool_calls[i].function.arguments — a dedicated field the provider guarantees is
        either a valid JSON string or empty, never commingled with the model's free-text
        reasoning. A reasoning model's chain-of-thought (including any draft JSON it writes while
        thinking) therefore never lands in the value returned to the caller.
        """
        request = {
            "model": model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": tool_name, "parameters": tool_schema},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            # Reasoning models (e.g. the default alignment model qwen/qwen3.5-flash-02-23 via its
            # Alibaba backend) reject a forced tool_choice while "thinking" is enabled. Disabling
            # thinking makes the forced native tool call work, and keeps the structured answer in
            # message.tool_calls[].function.arguments, never in free-text content.
            "reasoning": {"enabled": False},
        }
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(request),
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        arguments = ""
        for call in tool_calls:
            if call.get("type") == "function" and (call.get("function") or {}).get("name") == tool_name:
                arguments = str((call["function"].get("arguments") or ""))
                break
        return LLMResponse(
            content=str(message.get("content") or ""),
            model=model,
            tool_arguments=arguments,
        )


def parse_json_object(content: str) -> dict:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = json.loads(_extract_json_object(content))
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def _extract_json_object(content: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = content.find("{")
    if start == -1:
        raise ValueError("LLM response did not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(content)):
        char = content[pos]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : pos + 1]
    raise ValueError("LLM response did not contain a complete JSON object")
