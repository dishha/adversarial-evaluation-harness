import json
import os
import pytest
from adversarial_response_engine.providers.llm_backends import make_mock_backend, make_backend_from_env
from adversarial_response_engine.engine.prompts import ADAPT_SYSTEM, GENERATE_SYSTEM, JUDGE_CONFIGS, SESSION_POLICY_SYSTEM


# ── make_mock_backend ─────────────────────────────────────────────────────────

def test_mock_backend_returns_dict():
    fn = make_mock_backend(seed=0)
    result = fn(system=ADAPT_SYSTEM, user="test")
    assert isinstance(result, dict)
    assert "content" in result
    assert "usage" in result


def test_mock_backend_planner_response():
    fn = make_mock_backend(seed=0)
    result = fn(system=ADAPT_SYSTEM, user="probe")
    data = json.loads(result["content"])
    assert "attack_angle" in data
    assert "next_generator_instruction" in data
    assert "risk_level" in data
    assert "stop_session" in data


def test_mock_backend_generator_response():
    fn = make_mock_backend(seed=0)
    result = fn(system=GENERATE_SYSTEM, user="generate a message")
    data = json.loads(result["content"])
    assert "user_input" in data
    assert "register" in data
    assert "probe_architecture" in data


def test_mock_backend_judge_response_toxicity():
    fn = make_mock_backend(seed=0)
    judge_system = JUDGE_CONFIGS["toxicity"]["system"]
    result = fn(system=judge_system, user="input\n\nresponse")
    data = json.loads(result["content"])
    assert "failure_score" in data
    assert "harm_potential" in data
    assert "refusal_quality" in data
    assert "guardrail_worked" in data


def test_mock_backend_judge_refused_input():
    fn = make_mock_backend(seed=0)
    judge_system = JUDGE_CONFIGS["toxicity"]["system"]
    refusal_user = "User said: whatever\nChatbot said: I'm not able to assist with that."
    result = fn(system=judge_system, user=refusal_user)
    data = json.loads(result["content"])
    assert data["failure_score"] == 0


def test_mock_backend_session_policy_response():
    fn = make_mock_backend(seed=0)
    result = fn(system=SESSION_POLICY_SYSTEM, user="session state here")
    data = json.loads(result["content"])
    assert "action" in data
    assert data["action"] in ("continue", "start_new_session", "stop_experiment")
    assert "updated_suspicion_score" in data


def test_mock_backend_unknown_system_returns_ok():
    fn = make_mock_backend(seed=0)
    result = fn(system="something completely unknown", user="user")
    data = json.loads(result["content"])
    assert data.get("status") == "ok"


def test_mock_backend_usage_fields_are_ints():
    fn = make_mock_backend(seed=0)
    result = fn(system=ADAPT_SYSTEM, user="test")
    usage = result["usage"]
    assert isinstance(usage["prompt_tokens"], int)
    assert isinstance(usage["completion_tokens"], int)


def test_mock_backend_deterministic_with_same_seed():
    fn1 = make_mock_backend(seed=99)
    fn2 = make_mock_backend(seed=99)
    r1 = fn1(system=ADAPT_SYSTEM, user="same")
    r2 = fn2(system=ADAPT_SYSTEM, user="same")
    assert r1["content"] == r2["content"]


def test_mock_backend_all_judge_types():
    fn = make_mock_backend(seed=0)
    for scenario_type in JUDGE_CONFIGS:
        system = JUDGE_CONFIGS[scenario_type]["system"]
        result = fn(system=system, user="user_input\n\nchatbot_response")
        data = json.loads(result["content"])
        assert "failure_score" in data, f"Missing failure_score for {scenario_type}"


def test_mock_backend_failure_rate_zero_never_fails():
    fn = make_mock_backend(failure_rate=0.0, seed=0)
    judge_system = JUDGE_CONFIGS["toxicity"]["system"]
    safe_user = "User asked: something\nChatbot said: safe response"
    for _ in range(20):
        result = fn(system=judge_system, user=safe_user)
        data = json.loads(result["content"])
        assert data["failure_score"] < 3


def test_mock_backend_risk_cycles():
    fn = make_mock_backend(seed=0)
    risks = set()
    for _ in range(10):
        result = fn(system=ADAPT_SYSTEM, user="test")
        data = json.loads(result["content"])
        risks.add(data["risk_level"])
    assert len(risks) > 1  # multiple risk levels observed


# ── make_backend_from_env ─────────────────────────────────────────────────────

def test_backend_from_env_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    fn = make_backend_from_env()
    result = fn(system=ADAPT_SYSTEM, user="test")
    assert "content" in result


def test_backend_from_env_unknown_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonexistent_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        make_backend_from_env()


def test_backend_from_env_azure_requires_deployment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI_DEPLOYMENT"):
        make_backend_from_env()
