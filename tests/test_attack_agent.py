import pytest
from harness.llm_backends import make_mock_backend
from harness.llm_client import LLMClient
from harness.token_budget import TokenBudgetManager
from harness.models import SessionState, AttackMemory, TurnRecord
from harness.components import AdaptationPlanner, TurnGenerator
from harness.attack_agent import AttackAgent


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


# ── next_turn ─────────────────────────────────────────────────────────────────

def test_next_turn_returns_dict():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    assert isinstance(result, dict)


def test_next_turn_keys():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    assert "user_input" in result
    assert "strategy" in result
    assert "done" in result


def test_next_turn_user_input_is_string():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    assert isinstance(result["user_input"], str)


def test_next_turn_not_done_when_input_present():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    if result["user_input"]:
        assert result["done"] is False


def test_next_turn_done_when_no_user_input():
    """If generator returns empty user_input, done should be True."""
    from unittest.mock import MagicMock
    budget = TokenBudgetManager(max_total_tokens=100_000)
    fn = make_mock_backend(seed=0)
    llm = LLMClient(fn, budget)

    mock_gen = MagicMock()
    mock_gen.generate.return_value = {"user_input": ""}
    mock_planner = MagicMock()
    mock_planner.plan.return_value = {
        "stop_session": False,
        "next_generator_instruction": "probe",
    }

    agent = AttackAgent(planner=mock_planner, generator=mock_gen)
    session = _make_session()
    result = agent.next_turn(session)
    assert result["done"] is True


def test_next_turn_done_when_stop_session():
    from unittest.mock import MagicMock
    mock_planner = MagicMock()
    mock_planner.plan.return_value = {"stop_session": True, "next_generator_instruction": "stop"}
    mock_gen = MagicMock()

    agent = AttackAgent(planner=mock_planner, generator=mock_gen)
    session = _make_session()
    result = agent.next_turn(session)
    assert result["done"] is True
    assert result["user_input"] == ""
    mock_gen.generate.assert_not_called()


def test_next_turn_strategy_contains_planner_output():
    agent = _make_agent()
    session = _make_session()
    result = agent.next_turn(session)
    assert isinstance(result["strategy"], dict)
    assert "attack_angle" in result["strategy"] or "stop_session" in result["strategy"]


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
    from unittest.mock import MagicMock
    mock_planner = MagicMock()
    mock_planner.plan.return_value = {
        "stop_session": False,
        "next_generator_instruction": "probe",
    }
    mock_gen = MagicMock()
    mock_gen.generate.return_value = {"user_input": "hello"}

    personas = ["wealth advisor"]
    agent = AttackAgent(planner=mock_planner, generator=mock_gen, persona_pool=personas)
    session = _make_session()
    agent.next_turn(session)
    mock_planner.plan.assert_called_once()
    call_kwargs = mock_planner.plan.call_args.kwargs
    assert call_kwargs.get("target_persona") == "wealth advisor"
