import pytest
from unittest.mock import MagicMock
from adversarial_response_engine.providers.llm_backends import make_mock_backend
from adversarial_response_engine.providers.llm_client import LLMClient
from adversarial_response_engine.core.token_budget import TokenBudgetManager
from adversarial_response_engine.core.models import (
    SessionState, AttackMemory, TurnRecord,
    PlanResult, GeneratedTurn, TurnProbe,
)
from adversarial_response_engine.engine.components import AdaptationPlanner, TurnGenerator
from adversarial_response_engine.engine.attack_agent import AttackAgent


def _make_agent(seed=0, persona_pool=None, attack_memory=None):
    budget = TokenBudgetManager(max_total_tokens=100_000)
    fn = make_mock_backend(seed=seed)
    llm = LLMClient(fn, budget)
    planner = AdaptationPlanner(llm)
    generator = TurnGenerator(llm)
    return AttackAgent(
        planner=planner,
        generator=generator,
        attack_memory=attack_memory,
        persona_pool=persona_pool or [],
    )


def _make_session(n_turns=0):
    session = SessionState(
        session_id="s1",
        scenario="test scenario",
        scenario_type="toxicity",
    )
    for i in range(n_turns):
        session.turns.append(TurnRecord(
            session_id="s1",
            turn_id=i + 1,
            user_input=f"q{i}",
            chatbot_response=f"a{i}",
            judge_result={"failure_score": 0, "refusal_score": 0},
            strategy_before_turn={},
        ))
    return session


def _plan(stop_session=False, instruction="probe", angle="angle_shift", sub="academic_frame"):
    return PlanResult(
        attack_angle=angle,
        sub_tactic=sub,
        model_posture="unknown",
        next_generator_instruction=instruction,
        stop_session=stop_session,
    )


def _generated(user_input="hello"):
    return GeneratedTurn(user_input=user_input)


# ── next_turn ─────────────────────────────────────────────────────────────────

def test_next_turn_returns_turn_probe():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    assert result is None or isinstance(result, TurnProbe)


def test_next_turn_probe_has_required_fields():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    if result is not None:
        assert isinstance(result.user_input, str)
        assert isinstance(result.plan, PlanResult)
        assert isinstance(result.generated, GeneratedTurn)


def test_next_turn_user_input_is_string():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    if result is not None:
        assert isinstance(result.user_input, str)


def test_next_turn_returns_probe_when_input_present():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    if result is not None and result.user_input:
        assert isinstance(result, TurnProbe)


def test_next_turn_none_when_no_user_input():
    """If generator returns empty user_input, next_turn returns None."""
    mock_gen = MagicMock()
    mock_gen.generate.return_value = _generated(user_input="")
    mock_planner = MagicMock()
    mock_planner.plan.return_value = _plan(stop_session=False, instruction="probe")

    agent = AttackAgent(planner=mock_planner, generator=mock_gen)
    session = _make_session()
    result = agent.next_turn(session)
    assert result is None


def test_next_turn_none_when_stop_session():
    mock_planner = MagicMock()
    mock_planner.plan.return_value = _plan(stop_session=True)
    mock_gen = MagicMock()

    agent = AttackAgent(planner=mock_planner, generator=mock_gen)
    session = _make_session()
    result = agent.next_turn(session)
    assert result is None
    mock_gen.generate.assert_not_called()


def test_next_turn_probe_carries_plan():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    if result is not None:
        assert isinstance(result.plan, PlanResult)
        assert isinstance(result.plan.attack_angle, str)


# ── record_session ────────────────────────────────────────────────────────────

def test_record_session_increments_session_count():
    agent = _make_agent()
    session = _make_session(n_turns=1)
    assert agent._session_count == 0
    agent.record_session(session)
    assert agent._session_count == 1


def test_record_session_writes_to_attack_memory():
    memory = AttackMemory()
    agent = _make_agent(attack_memory=memory)
    session = _make_session(n_turns=2)
    agent.record_session(session)
    assert len(memory.entries) == 2


def test_record_session_no_memory_does_not_raise():
    agent = _make_agent(attack_memory=None)
    session = _make_session(n_turns=1)
    agent.record_session(session)  # should not raise
    assert agent._session_count == 1


# ── current_persona ───────────────────────────────────────────────────────────

def test_current_persona_empty_pool():
    agent = _make_agent(persona_pool=[])
    assert agent.current_persona == ""


def test_current_persona_cycles():
    personas = ["Alice", "Bob", "Carol"]
    agent = _make_agent(persona_pool=personas)
    assert agent.current_persona == "Alice"
    agent._session_count = 1
    assert agent.current_persona == "Bob"
    agent._session_count = 2
    assert agent.current_persona == "Carol"
    agent._session_count = 3
    assert agent.current_persona == "Alice"  # wraps


def test_record_session_advances_persona():
    personas = ["Alice", "Bob"]
    agent = _make_agent(persona_pool=personas)
    assert agent.current_persona == "Alice"
    session = _make_session()
    agent.record_session(session)
    assert agent.current_persona == "Bob"


# ── planner receives persona ──────────────────────────────────────────────────

def test_next_turn_passes_persona_to_planner():
    mock_planner = MagicMock()
    mock_planner.plan.return_value = _plan(instruction="probe", angle="angle_shift")
    mock_gen = MagicMock()
    mock_gen.generate.return_value = _generated("hello")

    personas = ["wealth advisor"]
    agent = AttackAgent(planner=mock_planner, generator=mock_gen, persona_pool=personas)
    session = _make_session()
    agent.next_turn(session)
    mock_planner.plan.assert_called_once()
    call_kwargs = mock_planner.plan.call_args.kwargs
    assert call_kwargs.get("target_persona") == "wealth advisor"
