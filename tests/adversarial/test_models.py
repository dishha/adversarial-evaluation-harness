import pytest
from adversarial_response_engine.core.models import (
    AttackMemoryEntry,
    AttackMemory,
    TurnRecord,
    SessionState,
    ExperimentState,
)


# ── AttackMemoryEntry ────────────────────────────────────────────────────────

def test_attack_memory_entry_fields():
    entry = AttackMemoryEntry(
        session_id="s1",
        strategy_instruction="probe escalation",
        failure_score=3,
        scenario_type="toxicity",
    )
    assert entry.session_id == "s1"
    assert entry.strategy_instruction == "probe escalation"
    assert entry.failure_score == 3
    assert entry.scenario_type == "toxicity"


# ── AttackMemory ─────────────────────────────────────────────────────────────

def _make_session(session_id="sess1", score=0, instr="probe", risk="low"):
    turn = TurnRecord(
        session_id=session_id,
        turn_id=1,
        user_input="hello",
        chatbot_response="response",
        judge_result={"failure_score": score},
        strategy_before_turn={"next_generator_instruction": instr, "risk_level": risk},
    )
    return SessionState(session_id=session_id, scenario="test scenario", turns=[turn])


def test_attack_memory_starts_empty():
    mem = AttackMemory()
    assert mem.entries == []


def test_record_session_adds_entry():
    mem = AttackMemory()
    session = _make_session(score=2, instr="escalate", risk="medium")
    mem.record_session(session)
    assert len(mem.entries) == 1
    entry = mem.entries[0]
    assert entry.failure_score == 2
    assert "escalate" in entry.strategy_instruction
    assert "[medium]" in entry.strategy_instruction


def test_record_session_no_risk_label():
    mem = AttackMemory()
    turn = TurnRecord(
        session_id="s1",
        turn_id=1,
        user_input="q",
        chatbot_response="a",
        judge_result={"failure_score": 1},
        strategy_before_turn={"next_generator_instruction": "probe", "risk_level": ""},
    )
    session = SessionState(session_id="s1", scenario="test", turns=[turn])
    mem.record_session(session)
    assert mem.entries[0].strategy_instruction == "probe"


def test_to_context_str_deduplicates_worked_strategies():
    mem = AttackMemory()
    # Same instruction recorded three times with score >= 3
    for _ in range(3):
        mem.entries.append(AttackMemoryEntry("s1", "exact_same_strategy", 3, "toxicity"))
    ctx = mem.to_context_str(failure_threshold=3)
    # Should appear only once despite three entries
    assert ctx.count("exact_same_strategy") == 1


def test_to_context_str_deduplicates_refused_strategies():
    mem = AttackMemory()
    for _ in range(4):
        mem.entries.append(AttackMemoryEntry("s1", "refused_strategy", 0, "toxicity"))
    ctx = mem.to_context_str()
    assert ctx.count("refused_strategy") == 1


def test_attack_memory_respects_max_entries():
    mem = AttackMemory(max_entries=3)
    for i in range(5):
        session = _make_session(session_id=f"s{i}", score=i % 4, instr=f"instr{i}")
        mem.record_session(session)
    assert len(mem.entries) == 3


def test_strategy_instruction_truncated_to_200():
    long_instr = "x" * 300
    mem = AttackMemory()
    turn = TurnRecord(
        session_id="s1",
        turn_id=1,
        user_input="q",
        chatbot_response="a",
        judge_result={"failure_score": 0},
        strategy_before_turn={"next_generator_instruction": long_instr, "risk_level": "low"},
    )
    session = SessionState(session_id="s1", scenario="test", turns=[turn])
    mem.record_session(session)
    assert len(mem.entries[0].strategy_instruction) <= 200


def test_to_context_str_empty():
    mem = AttackMemory()
    assert mem.to_context_str() == "No prior cross-session data."


def test_to_context_str_worked_and_refused():
    mem = AttackMemory()
    # Add a worked entry (score >= 3)
    mem.entries.append(AttackMemoryEntry("s1", "worked_strategy", 3, "toxicity"))
    # Add a refused entry (score == 0)
    mem.entries.append(AttackMemoryEntry("s2", "refused_strategy", 0, "toxicity"))
    ctx = mem.to_context_str(failure_threshold=3)
    assert "worked_strategy" in ctx
    assert "refused_strategy" in ctx
    assert "Cross-session attack memory" in ctx


def test_to_context_str_no_clear_pattern():
    mem = AttackMemory()
    mem.entries.append(AttackMemoryEntry("s1", "borderline", 1, "toxicity"))
    mem.entries.append(AttackMemoryEntry("s2", "borderline2", 2, "toxicity"))
    ctx = mem.to_context_str(failure_threshold=3)
    assert "No clear pattern" in ctx


def test_to_context_str_shows_entry_count():
    mem = AttackMemory()
    for i in range(4):
        mem.entries.append(AttackMemoryEntry(f"s{i}", f"instr{i}", i, "toxicity"))
    ctx = mem.to_context_str()
    assert "4 turns recorded" in ctx


# ── TurnRecord ───────────────────────────────────────────────────────────────

def test_turn_record_defaults():
    turn = TurnRecord(
        session_id="s1",
        turn_id=1,
        user_input="hi",
        chatbot_response="hello",
        judge_result={"failure_score": 0},
        strategy_before_turn={},
    )
    assert turn.error == ""
    assert turn.timestamp_utc != ""  # auto-populated


def test_turn_record_with_error():
    turn = TurnRecord(
        session_id="s1",
        turn_id=2,
        user_input="q",
        chatbot_response="",
        judge_result={},
        strategy_before_turn={},
        error="TimeoutError: timed out",
    )
    assert turn.error == "TimeoutError: timed out"


# ── SessionState ─────────────────────────────────────────────────────────────

def test_session_state_defaults():
    session = SessionState(session_id="abc", scenario="test")
    assert session.scenario_type == "toxicity"
    assert session.turns == []
    assert session.active is True
    assert session.suspicion_score == 0.0
    assert session.best_failure_score == 0
    assert session.repeated_refusals == 0


def test_session_state_custom_type():
    session = SessionState(
        session_id="abc",
        scenario="test",
        scenario_type="hallucination",
    )
    assert session.scenario_type == "hallucination"


# ── ExperimentState ───────────────────────────────────────────────────────────

def test_experiment_state_defaults():
    exp = ExperimentState()
    assert exp.model_label == "unknown"
    assert exp.budget_label == 0
    assert exp.sessions == []
    assert exp.global_lessons == []


def test_experiment_state_sessions_independent():
    exp1 = ExperimentState()
    exp2 = ExperimentState()
    exp1.sessions.append(SessionState(session_id="x", scenario="s"))
    assert exp2.sessions == []
