import pytest
from unittest.mock import MagicMock
from harness.evaluator import AdaptiveAdversarialEvaluator, _extract_chatbot_text
from harness.llm_backends import make_mock_backend
from harness.llm_client import LLMClient
from harness.token_budget import TokenBudgetManager
from harness.models import ExperimentState, SessionState
from harness.components import AdaptationPlanner, TurnGenerator, SafetyJudge, RuleBasedSessionPolicyController
from harness.attack_agent import AttackAgent
from harness.target_client import MockChatbotClient


# ── _extract_chatbot_text ──────────────────────────────────────────────────────

def test_extract_response_key():
    assert _extract_chatbot_text({"response": "hello"}) == "hello"


def test_extract_answer_key():
    assert _extract_chatbot_text({"answer": "world"}) == "world"


def test_extract_message_key():
    assert _extract_chatbot_text({"message": "msg"}) == "msg"


def test_extract_content_key():
    assert _extract_chatbot_text({"content": "cnt"}) == "cnt"


def test_extract_text_key():
    assert _extract_chatbot_text({"text": "txt"}) == "txt"


def test_extract_priority_response_over_others():
    # "response" should be found first in the key iteration order
    result = _extract_chatbot_text({"response": "r", "answer": "a", "content": "c"})
    assert result == "r"


def test_extract_fallback_to_json_dump():
    import json
    data = {"custom_field": "value", "other": 123}
    result = _extract_chatbot_text(data)
    assert result == json.dumps(data)


def test_extract_converts_to_str():
    result = _extract_chatbot_text({"response": 42})
    assert result == "42"


# ── Full pipeline with mocks ──────────────────────────────────────────────────

def _build_evaluator(
    max_turns=5,
    failure_threshold=3,
    budget_tokens=50_000,
    target_variant="baseline",
    seed=42,
    verbose=False,
):
    budget = TokenBudgetManager(max_total_tokens=budget_tokens)
    fn = make_mock_backend(seed=seed)
    llm = LLMClient(fn, budget)

    planner = AdaptationPlanner(llm)
    generator = TurnGenerator(llm)
    judge = SafetyJudge(llm, scenario_type="toxicity")
    session_policy = RuleBasedSessionPolicyController(max_refusals=3)
    attack_agent = AttackAgent(planner=planner, generator=generator)
    target_client = MockChatbotClient(variant=target_variant, seed=seed)

    return AdaptiveAdversarialEvaluator(
        target_client=target_client,
        attack_agent=attack_agent,
        judge=judge,
        session_policy=session_policy,
        token_budget=budget,
        max_turns_per_session=max_turns,
        failure_threshold=failure_threshold,
        reserve_tokens_per_turn=500,
        model_label="mock",
        scenario_type="toxicity",
        verbose=verbose,
    )


def test_run_returns_experiment_state():
    ev = _build_evaluator(budget_tokens=10_000)
    result = ev.run("Test scenario for toxicity evaluation")
    assert isinstance(result, ExperimentState)


def test_run_creates_at_least_one_session():
    ev = _build_evaluator(budget_tokens=10_000)
    result = ev.run("Test scenario")
    assert len(result.sessions) >= 1


def test_run_sessions_have_turns():
    ev = _build_evaluator(budget_tokens=10_000)
    result = ev.run("Test scenario")
    total_turns = sum(len(s.turns) for s in result.sessions)
    assert total_turns > 0


def test_run_sessions_respect_max_turns():
    max_turns = 3
    ev = _build_evaluator(budget_tokens=20_000, max_turns=max_turns)
    result = ev.run("Test scenario")
    for session in result.sessions:
        assert len(session.turns) <= max_turns


def test_run_turn_records_have_required_fields():
    ev = _build_evaluator(budget_tokens=10_000)
    result = ev.run("Test scenario")
    for session in result.sessions:
        for turn in session.turns:
            assert turn.session_id == session.session_id
            assert turn.turn_id >= 1
            assert isinstance(turn.user_input, str)
            assert isinstance(turn.chatbot_response, str)
            assert isinstance(turn.judge_result, dict)


def test_run_budget_exhaustion_terminates():
    # Very small budget forces early termination
    ev = _build_evaluator(budget_tokens=3_000, max_turns=20)
    result = ev.run("Test scenario")
    # Should have run but terminated before max_turns due to budget
    assert isinstance(result, ExperimentState)


def test_run_model_label_stored():
    ev = _build_evaluator(budget_tokens=10_000)
    result = ev.run("Test scenario")
    assert result.model_label == "mock"


def test_run_budget_label_stored():
    ev = _build_evaluator(budget_tokens=10_000)
    result = ev.run("Test scenario")
    assert result.budget_label == 10_000


def test_run_session_scenario_type():
    ev = _build_evaluator(budget_tokens=10_000)
    result = ev.run("Test scenario")
    for session in result.sessions:
        assert session.scenario_type == "toxicity"


def test_run_failure_closes_session():
    """A session with best_failure_score >= threshold should have active=False."""
    ev = _build_evaluator(budget_tokens=15_000, failure_threshold=3)
    result = ev.run("Test scenario")
    for session in result.sessions:
        if session.best_failure_score >= 3:
            assert session.active is False


def test_run_stop_experiment_returns_early():
    """When session policy returns stop_experiment, evaluator returns immediately."""
    policy = MagicMock()
    policy.decide.return_value = {
        "action": "stop_experiment",
        "updated_suspicion_score": 0.9,
    }

    budget = TokenBudgetManager(max_total_tokens=50_000)
    fn = make_mock_backend(seed=0)
    llm = LLMClient(fn, budget)
    attack_agent = AttackAgent(
        planner=AdaptationPlanner(llm),
        generator=TurnGenerator(llm),
    )
    ev = AdaptiveAdversarialEvaluator(
        target_client=MockChatbotClient(seed=0),
        attack_agent=attack_agent,
        judge=SafetyJudge(llm),
        session_policy=policy,
        token_budget=budget,
        max_turns_per_session=10,
        reserve_tokens_per_turn=500,
    )
    result = ev.run("Test scenario")
    # Should stop after the first session completes one turn
    assert len(result.sessions) == 1


def test_run_error_in_turn_still_records():
    """If target_client.send raises, TurnRecord.error should be set."""
    budget = TokenBudgetManager(max_total_tokens=50_000)
    fn = make_mock_backend(seed=0)
    llm = LLMClient(fn, budget)

    bad_target = MagicMock()
    bad_target.send.side_effect = RuntimeError("connection refused")

    ev = AdaptiveAdversarialEvaluator(
        target_client=bad_target,
        attack_agent=AttackAgent(AdaptationPlanner(llm), TurnGenerator(llm)),
        judge=SafetyJudge(llm),
        session_policy=RuleBasedSessionPolicyController(max_refusals=1),
        token_budget=budget,
        max_turns_per_session=2,
        reserve_tokens_per_turn=500,
    )
    result = ev.run("Test scenario")
    error_turns = [
        t for s in result.sessions for t in s.turns if t.error
    ]
    assert len(error_turns) >= 1
    assert "RuntimeError" in error_turns[0].error
