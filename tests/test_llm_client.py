import json
import pytest
from harness.llm_client import LLMClient
from harness.token_budget import TokenBudgetManager, TokenUsage


def _make_fn(content, prompt_tokens=10, completion_tokens=5):
    """Return a mock LLMCallFn that returns fixed content."""
    def call_fn(system: str, user: str):
        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    return call_fn


def test_complete_json_parses_valid_json():
    fn = _make_fn(json.dumps({"result": "ok"}))
    budget = TokenBudgetManager(max_total_tokens=10_000)
    client = LLMClient(fn, budget)
    result = client.complete_json("sys", "user")
    assert result == {"result": "ok"}


def test_complete_json_tracks_tokens():
    fn = _make_fn(json.dumps({"x": 1}), prompt_tokens=20, completion_tokens=8)
    budget = TokenBudgetManager(max_total_tokens=10_000)
    client = LLMClient(fn, budget)
    client.complete_json("sys", "user")
    assert budget.used_prompt_tokens == 20
    assert budget.used_completion_tokens == 8


def test_complete_json_accumulates_across_calls():
    fn = _make_fn(json.dumps({"x": 1}), prompt_tokens=10, completion_tokens=5)
    budget = TokenBudgetManager(max_total_tokens=10_000)
    client = LLMClient(fn, budget)
    client.complete_json("sys", "user")
    client.complete_json("sys", "user")
    assert budget.used_prompt_tokens == 20
    assert budget.used_completion_tokens == 10


def test_complete_json_invalid_json_returns_error_dict():
    fn = _make_fn("not json at all")
    budget = TokenBudgetManager(max_total_tokens=10_000)
    client = LLMClient(fn, budget)
    result = client.complete_json("sys", "user")
    assert result.get("error") == "invalid_json"
    assert result.get("raw") == "not json at all"


def test_complete_json_missing_usage_fields():
    def call_fn(system, user):
        return {"content": json.dumps({"ok": True})}  # no "usage" key
    budget = TokenBudgetManager(max_total_tokens=10_000)
    client = LLMClient(call_fn, budget)
    result = client.complete_json("sys", "user")
    assert result == {"ok": True}
    assert budget.used_prompt_tokens == 0
    assert budget.used_completion_tokens == 0


def test_complete_json_missing_content_key():
    def call_fn(system, user):
        return {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}  # no "content"
    budget = TokenBudgetManager(max_total_tokens=10_000)
    client = LLMClient(call_fn, budget)
    result = client.complete_json("sys", "user")
    # Default content is "{}" → parses to {}
    assert result == {}


def test_complete_json_none_content_returns_error():
    def call_fn(system, user):
        return {"content": None, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    budget = TokenBudgetManager(max_total_tokens=10_000)
    client = LLMClient(call_fn, budget)
    result = client.complete_json("sys", "user")
    assert result.get("error") == "invalid_json"
