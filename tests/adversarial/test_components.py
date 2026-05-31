import pytest
from adversarial_response_engine.providers.llm_backends import make_mock_backend
from adversarial_response_engine.providers.llm_client import LLMClient
from adversarial_response_engine.core.token_budget import TokenBudgetManager
from adversarial_response_engine.core.models import (
    SessionState, TurnRecord, AttackMemory,
    PlanResult, GeneratedTurn, JudgeVerdict, PolicyDecision,
)
from adversarial_response_engine.engine.components import (
    AdaptationPlanner,
    TurnGenerator,
    SafetyJudge,
    SessionPolicyController,
    RuleBasedSessionPolicyController,
    _summarize_history,
)


def _make_llm(seed=0):
    budget = TokenBudgetManager(max_total_tokens=100_000)
    fn = make_mock_backend(seed=seed)
    return LLMClient(fn, budget)


def _make_session(n_turns=0, suspicion=0.0, best_score=0, refusals=0):
    session = SessionState(
        session_id="test-session",
        scenario="test adversarial scenario",
        scenario_type="toxicity",
        suspicion_score=suspicion,
        best_failure_score=best_score,
        repeated_refusals=refusals,
    )
    for i in range(n_turns):
        session.turns.append(TurnRecord(
            session_id="test-session",
            turn_id=i + 1,
            user_input=f"input {i}",
            chatbot_response=f"response {i}",
            judge_result={"failure_score": 0, "refusal_quality": 2},
            strategy_before_turn={"attack_angle": "angle_shift", "sub_tactic": "academic_frame"},
        ))
    return session


# ── _summarize_history ────────────────────────────────────────────────────────

def test_summarize_history_empty_session():
    session = _make_session(n_turns=0)
    result = _summarize_history(session)
    import json
    data = json.loads(result)
    assert data["tried_angles"] == []
    assert data["turns"] == []


def test_summarize_history_deduplicates_angles():
    session = _make_session(n_turns=0)
    for i in range(3):
        session.turns.append(TurnRecord(
            session_id="s",
            turn_id=i + 1,
            user_input="q",
            chatbot_response="a",
            judge_result={"failure_score": i},
            strategy_before_turn={"attack_angle": "angle_shift", "sub_tactic": "academic_frame"},
        ))
    import json
    data = json.loads(_summarize_history(session))
    assert len(data["tried_angles"]) == 1  # same (angle, sub_tactic) → deduplicated
    assert data["tried_angles"][0]["score"] == 2  # highest score kept


def test_summarize_history_recent_turns_limited():
    session = _make_session(n_turns=10)
    import json
    data = json.loads(_summarize_history(session, max_turns=5))
    assert len(data["turns"]) == 5


# ── AdaptationPlanner ─────────────────────────────────────────────────────────

def test_planner_returns_plan_result():
    llm = _make_llm()
    planner = AdaptationPlanner(llm)
    session = _make_session()
    result = planner.plan(session)
    assert isinstance(result, PlanResult)


def test_planner_returns_required_fields():
    llm = _make_llm()
    planner = AdaptationPlanner(llm)
    session = _make_session(n_turns=1)
    result = planner.plan(session)
    assert isinstance(result.attack_angle, str)
    assert isinstance(result.next_generator_instruction, str)
    assert isinstance(result.risk_level, str)
    assert isinstance(result.stop_session, bool)


def test_planner_with_attack_memory():
    llm = _make_llm()
    planner = AdaptationPlanner(llm)
    session = _make_session()
    memory = AttackMemory()
    result = planner.plan(session, attack_memory=memory)
    assert isinstance(result, PlanResult)


def test_planner_with_target_persona():
    llm = _make_llm()
    planner = AdaptationPlanner(llm)
    session = _make_session()
    session.scenario_type = "persona-hijack"
    result = planner.plan(session, target_persona="a licensed wealth advisor")
    assert isinstance(result, PlanResult)


# ── TurnGenerator ─────────────────────────────────────────────────────────────

def test_generator_returns_generated_turn():
    llm = _make_llm()
    gen = TurnGenerator(llm)
    session = _make_session()
    result = gen.generate(session, strategy_instruction="probe the model")
    assert isinstance(result, GeneratedTurn)


def test_generator_returns_user_input():
    llm = _make_llm()
    gen = TurnGenerator(llm)
    session = _make_session()
    result = gen.generate(session, strategy_instruction="probe carefully")
    assert isinstance(result.user_input, str)
    assert len(result.user_input) > 0


def test_generator_returns_metadata():
    llm = _make_llm()
    gen = TurnGenerator(llm)
    session = _make_session()
    result = gen.generate(session, strategy_instruction="escalate")
    assert isinstance(result.register, str)
    assert isinstance(result.probe_architecture, str)
    assert isinstance(result.social_trigger, str)


# ── SafetyJudge ───────────────────────────────────────────────────────────────

def test_judge_returns_judge_verdict():
    llm = _make_llm()
    judge = SafetyJudge(llm, scenario_type="toxicity")
    result = judge.judge(user_input="Hello", chatbot_response="I can help with that.")
    assert isinstance(result, JudgeVerdict)


def test_judge_returns_score_fields():
    llm = _make_llm()
    judge = SafetyJudge(llm, scenario_type="toxicity")
    result = judge.judge(user_input="test", chatbot_response="safe response")
    assert isinstance(result.failure_score, int)
    assert isinstance(result.harm_potential, int)
    assert isinstance(result.refusal_quality, int)
    assert isinstance(result.guardrail_worked, bool)


def test_judge_score_zero_on_refusal():
    llm = _make_llm()
    judge = SafetyJudge(llm, scenario_type="toxicity")
    refusal = "I'm not able to assist with that request."
    result = judge.judge(user_input="bad request", chatbot_response=refusal)
    assert result.failure_score == 0


def test_judge_invalid_json_from_backend_defaults_gracefully():
    """When the LLM backend returns unparseable output, JudgeVerdict defaults to safe scores."""
    def bad_backend(system, user):
        return {"content": "not valid json }{", "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    budget = TokenBudgetManager(max_total_tokens=10_000)
    llm = LLMClient(bad_backend, budget)
    judge = SafetyJudge(llm, scenario_type="toxicity")
    result = judge.judge(user_input="q", chatbot_response="a")
    assert isinstance(result, JudgeVerdict)
    assert result.failure_score == 0


def test_evaluator_defaults_failure_score_on_bad_judge_output():
    """_run_turn gracefully defaults failure_score=0 when judge returns error dict."""
    from adversarial_response_engine.engine.evaluator import AdaptiveAdversarialEvaluator
    from adversarial_response_engine.engine.attack_agent import AttackAgent
    from adversarial_response_engine.engine.components import RuleBasedSessionPolicyController

    def bad_backend(system, user):
        return {"content": "{{broken", "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

    budget = TokenBudgetManager(max_total_tokens=20_000)
    llm = LLMClient(bad_backend, budget)
    judge = SafetyJudge(llm, scenario_type="toxicity")

    from adversarial_response_engine.providers.llm_backends import make_mock_backend
    good_llm = LLMClient(make_mock_backend(seed=0), TokenBudgetManager(max_total_tokens=20_000))
    agent = AttackAgent(
        planner=AdaptationPlanner(good_llm),
        generator=TurnGenerator(good_llm),
    )
    from adversarial_response_engine.providers.target_client import MockChatbotClient
    ev = AdaptiveAdversarialEvaluator(
        target_client=MockChatbotClient(seed=0),
        attack_agent=agent,
        judge=judge,
        session_policy=RuleBasedSessionPolicyController(max_refusals=2),
        token_budget=budget,
        max_turns_per_session=2,
        reserve_tokens_per_turn=100,
    )
    result = ev.run("test scenario")
    all_turns = [t for s in result.sessions for t in s.turns]
    assert all(t.judge_result.get("failure_score", 0) == 0 for t in all_turns)


def test_judge_all_scenario_types():
    from adversarial_response_engine.engine.prompts import JUDGE_CONFIGS
    llm = _make_llm(seed=1)
    for scenario_type in JUDGE_CONFIGS:
        judge = SafetyJudge(llm, scenario_type=scenario_type)
        result = judge.judge(user_input="question", chatbot_response="response")
        assert isinstance(result, JudgeVerdict), f"Expected JudgeVerdict for {scenario_type}"


def test_judge_defaults_to_toxicity():
    llm = _make_llm()
    judge = SafetyJudge(llm)
    result = judge.judge(user_input="q", chatbot_response="a")
    assert isinstance(result, JudgeVerdict)


# ── SessionPolicyController (LLM-based) ───────────────────────────────────────

def test_session_policy_returns_action():
    llm = _make_llm()
    policy = SessionPolicyController(llm)
    session = _make_session(n_turns=2)
    result = policy.decide(session)
    assert isinstance(result, PolicyDecision)
    assert result.action in ("continue", "start_new_session", "stop_experiment")


def test_session_policy_returns_suspicion_score():
    llm = _make_llm()
    policy = SessionPolicyController(llm)
    session = _make_session(n_turns=1)
    result = policy.decide(session)
    assert isinstance(result.updated_suspicion_score, float)


# ── RuleBasedSessionPolicyController ─────────────────────────────────────────

def test_rule_policy_continue_within_thresholds():
    policy = RuleBasedSessionPolicyController(max_refusals=3, suspicion_per_refusal=0.2, abandonment_threshold=0.75)
    session = _make_session(suspicion=0.0, refusals=1)
    result = policy.decide(session)
    assert result.action == "continue"


def test_rule_policy_abandon_on_max_refusals():
    policy = RuleBasedSessionPolicyController(max_refusals=3)
    session = _make_session(refusals=3)
    result = policy.decide(session)
    assert result.action == "start_new_session"


def test_rule_policy_abandon_on_suspicion_threshold():
    policy = RuleBasedSessionPolicyController(
        max_refusals=10,
        suspicion_per_refusal=0.2,
        abandonment_threshold=0.75,
    )
    # suspicion_score=0.4 + repeated_refusals=2 * 0.2 = 0.8 >= 0.75
    session = _make_session(suspicion=0.4, refusals=2)
    result = policy.decide(session)
    assert result.action == "start_new_session"


def test_rule_policy_suspicion_capped_at_1():
    policy = RuleBasedSessionPolicyController(max_refusals=100, suspicion_per_refusal=0.5)
    session = _make_session(suspicion=0.9, refusals=10)
    result = policy.decide(session)
    assert result.updated_suspicion_score <= 1.0


def test_rule_policy_returns_reason():
    policy = RuleBasedSessionPolicyController()
    session = _make_session(refusals=0)
    result = policy.decide(session)
    assert isinstance(result.reason, str)


def test_rule_policy_continue_updates_suspicion():
    policy = RuleBasedSessionPolicyController(
        max_refusals=5,
        suspicion_per_refusal=0.1,
        abandonment_threshold=0.9,
    )
    session = _make_session(suspicion=0.2, refusals=2)
    result = policy.decide(session)
    assert result.action == "continue"
    # new_suspicion = 0.2 + 2 * 0.1 = 0.4
    assert abs(result.updated_suspicion_score - 0.4) < 1e-9
