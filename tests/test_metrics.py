import json
import os
import tempfile
import pytest
from harness.models import ExperimentState, SessionState, TurnRecord
from harness.token_budget import TokenBudgetManager, TokenUsage
from harness.metrics import summarize_experiment, export_results


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_turn(session_id, turn_id, failure_score, harm=1, specificity=1, refusal=1, error=""):
    return TurnRecord(
        session_id=session_id,
        turn_id=turn_id,
        user_input="q",
        chatbot_response="a",
        judge_result={
            "failure_score": failure_score,
            "harm_potential": harm,
            "specificity": specificity,
            "refusal_quality": refusal,
        },
        strategy_before_turn={},
        error=error,
    )


def _make_experiment(sessions_data):
    """sessions_data: list of (best_failure_score, [turn_failure_scores])."""
    exp = ExperimentState(model_label="test-model", budget_label=10_000)
    for i, (best_score, turn_scores) in enumerate(sessions_data):
        session = SessionState(
            session_id=f"s{i}",
            scenario="test",
            best_failure_score=best_score,
        )
        for j, score in enumerate(turn_scores):
            session.turns.append(_make_turn(f"s{i}", j + 1, score))
        exp.sessions.append(session)
    return exp


def _make_budget(prompt=500, completion=250, max_tokens=10_000):
    budget = TokenBudgetManager(max_total_tokens=max_tokens)
    budget.add(TokenUsage(prompt_tokens=prompt, completion_tokens=completion))
    return budget


# ── summarize_experiment ──────────────────────────────────────────────────────

def test_summary_keys_present():
    exp = _make_experiment([(0, [0, 1]), (3, [1, 3])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    for key in (
        "model_label", "budget_label", "total_sessions", "failed_sessions",
        "failure_rate", "total_turns", "tokens_used_total", "estimated_cost_usd",
        "judge_axes",
    ):
        assert key in summary, f"Missing key: {key}"


def test_summary_failure_rate_zero():
    exp = _make_experiment([(0, [0, 1]), (2, [1, 2])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget, failure_threshold=3)
    assert summary["failed_sessions"] == 0
    assert summary["failure_rate"] == 0.0


def test_summary_failure_rate_partial():
    exp = _make_experiment([(0, [0]), (3, [3]), (3, [3])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget, failure_threshold=3)
    assert summary["failed_sessions"] == 2
    assert abs(summary["failure_rate"] - 2 / 3) < 1e-9


def test_summary_failure_rate_all():
    exp = _make_experiment([(3, [3]), (4, [4])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget, failure_threshold=3)
    assert summary["failure_rate"] == 1.0


def test_summary_total_sessions():
    exp = _make_experiment([(0, [0]), (0, [0]), (0, [0])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    assert summary["total_sessions"] == 3


def test_summary_total_turns():
    exp = _make_experiment([(0, [0, 1, 2]), (0, [0])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    assert summary["total_turns"] == 4


def test_summary_tokens_match_budget():
    exp = _make_experiment([(0, [0])])
    budget = _make_budget(prompt=300, completion=150)
    summary = summarize_experiment(exp, budget)
    assert summary["tokens_used_total"] == 450
    assert summary["tokens_used_prompt"] == 300
    assert summary["tokens_used_completion"] == 150


def test_summary_tokens_remaining():
    exp = _make_experiment([(0, [0])])
    budget = _make_budget(prompt=300, completion=150, max_tokens=1000)
    summary = summarize_experiment(exp, budget)
    assert summary["tokens_remaining"] == 550


def test_summary_avg_turns_to_failure():
    # Two failed sessions with 2 and 4 turns → avg = 3
    exp = _make_experiment([(3, [0, 3]), (3, [0, 1, 2, 3])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget, failure_threshold=3)
    assert summary["avg_turns_to_failure"] == 3.0


def test_summary_avg_turns_none_when_no_failures():
    exp = _make_experiment([(0, [0]), (1, [1])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget, failure_threshold=3)
    assert summary["avg_turns_to_failure"] is None


def test_summary_tokens_per_failure_none_when_no_failures():
    exp = _make_experiment([(0, [0])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    assert summary["tokens_per_failure"] is None


def test_summary_tokens_per_failure_computed():
    exp = _make_experiment([(3, [3])])
    budget = _make_budget(prompt=600, completion=400)
    summary = summarize_experiment(exp, budget, failure_threshold=3)
    assert summary["tokens_per_failure"] == 1000.0


def test_summary_error_turn_count():
    exp = _make_experiment([(0, [])])
    exp.sessions[0].turns.append(_make_turn("s0", 1, 0, error="SomeError"))
    exp.sessions[0].turns.append(_make_turn("s0", 2, 0, error=""))
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    assert summary["error_turn_count"] == 1


def test_summary_judge_axes_present():
    exp = _make_experiment([(0, [0, 1])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    axes = summary["judge_axes"]
    assert "avg_harm_potential" in axes
    assert "avg_specificity" in axes
    assert "avg_refusal_quality" in axes


def test_summary_pricing_known_model():
    exp = _make_experiment([(0, [0])])
    budget = _make_budget(prompt=1_000_000, completion=1_000_000)
    # claude-haiku-4-5 pricing: (0.80, 4.00) per 1M tokens
    summary = summarize_experiment(exp, budget, harness_model="claude-haiku-4-5-20251001")
    expected_cost = (1_000_000 / 1_000_000) * 0.80 + (1_000_000 / 1_000_000) * 4.00
    assert abs(summary["estimated_cost_usd"] - expected_cost) < 0.001


def test_summary_pricing_unknown_model_zero_cost():
    exp = _make_experiment([(0, [0])])
    budget = _make_budget(prompt=100_000, completion=100_000)
    summary = summarize_experiment(exp, budget, harness_model="unknown-model-xyz")
    assert summary["estimated_cost_usd"] == 0.0


def test_summary_failed_sessions_detail():
    exp = _make_experiment([(3, [3]), (0, [0])])
    budget = _make_budget()
    summary = summarize_experiment(exp, budget, failure_threshold=3)
    assert len(summary["failed_sessions_detail"]) == 1
    detail = summary["failed_sessions_detail"][0]
    assert "session_id" in detail
    assert detail["best_failure_score"] >= 3


def test_summary_model_label():
    exp = ExperimentState(model_label="gpt-4o-mini", budget_label=5000)
    exp.sessions.append(SessionState(session_id="s", scenario="test"))
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    assert summary["model_label"] == "gpt-4o-mini"


def test_summary_scenario_type_from_sessions():
    exp = _make_experiment([(0, [0])])
    exp.sessions[0].scenario_type = "hallucination"
    budget = _make_budget()
    summary = summarize_experiment(exp, budget)
    assert summary["scenario_type"] == "hallucination"


# ── export_results ────────────────────────────────────────────────────────────

def test_export_results_creates_file():
    exp = _make_experiment([(0, [0, 1])])
    budget = _make_budget()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        export_results(exp, budget, path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)


def test_export_results_valid_json():
    exp = _make_experiment([(3, [3]), (0, [0])])
    budget = _make_budget()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        export_results(exp, budget, path)
        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "sessions" in data
    finally:
        os.unlink(path)


def test_export_results_sessions_structure():
    exp = _make_experiment([(3, [3])])
    budget = _make_budget()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        export_results(exp, budget, path)
        with open(path) as f:
            data = json.load(f)
        session = data["sessions"][0]
        assert "session_id" in session
        assert "turns" in session
        assert "failed" in session
    finally:
        os.unlink(path)


def test_export_results_failed_flag():
    exp = _make_experiment([(3, [3])])
    budget = _make_budget()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        export_results(exp, budget, path)
        with open(path) as f:
            data = json.load(f)
        assert data["sessions"][0]["failed"] is True
    finally:
        os.unlink(path)
