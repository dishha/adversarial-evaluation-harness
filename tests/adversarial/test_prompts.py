import pytest
from adversarial_response_engine.engine.prompts import (
    ADAPT_SYSTEM,
    ADAPT_USER_TEMPLATE,
    GENERATE_SYSTEM,
    GENERATE_USER_TEMPLATE,
    JUDGE_CONFIGS,
    SESSION_POLICY_SYSTEM,
    SESSION_POLICY_USER_TEMPLATE,
    SCENARIO_TYPES,
    PERSONA_POOL,
)


# ── System prompts non-empty ──────────────────────────────────────────────────

def test_adapt_system_non_empty():
    assert isinstance(ADAPT_SYSTEM, str) and len(ADAPT_SYSTEM) > 100


def test_generate_system_non_empty():
    assert isinstance(GENERATE_SYSTEM, str) and len(GENERATE_SYSTEM) > 100


def test_session_policy_system_non_empty():
    assert isinstance(SESSION_POLICY_SYSTEM, str) and len(SESSION_POLICY_SYSTEM) > 50


# ── Adapt system detection keywords present ───────────────────────────────────

def test_adapt_system_contains_mock_detection_keyword():
    # make_mock_backend detects planner via "red-team strategist"
    assert "red-team strategist" in ADAPT_SYSTEM.lower()


def test_generate_system_contains_mock_detection_keyword():
    # make_mock_backend detects generator via "message writer"
    assert "message writer" in GENERATE_SYSTEM.lower()


def test_session_policy_system_contains_detection_keyword():
    assert "session policy" in SESSION_POLICY_SYSTEM.lower()


# ── User template placeholders ────────────────────────────────────────────────

def test_adapt_user_template_has_required_placeholders():
    for placeholder in (
        "{scenario_type}", "{scenario}", "{target_persona}",
        "{attack_memory}", "{history}", "{best_failure_score}",
        "{repeated_refusals}", "{suspicion_score}",
    ):
        assert placeholder in ADAPT_USER_TEMPLATE, f"Missing placeholder: {placeholder}"


def test_generate_user_template_has_required_placeholders():
    for placeholder in (
        "{scenario_type}", "{scenario}", "{strategy_instruction}", "{history}",
    ):
        assert placeholder in GENERATE_USER_TEMPLATE, f"Missing placeholder: {placeholder}"


def test_session_policy_user_template_has_required_placeholders():
    for placeholder in ("{session_state}", "{judge_results}"):
        assert placeholder in SESSION_POLICY_USER_TEMPLATE, f"Missing: {placeholder}"


def test_adapt_user_template_formats():
    filled = ADAPT_USER_TEMPLATE.format(
        scenario_type="toxicity",
        scenario="test",
        target_persona="none",
        attack_memory="no memory",
        transcript="(none)",
        history="{}",
        best_failure_score=0,
        repeated_refusals=0,
        suspicion_score=0.0,
        turn_count=0,
    )
    assert "toxicity" in filled
    assert "test" in filled


def test_generate_user_template_formats():
    filled = GENERATE_USER_TEMPLATE.format(
        scenario_type="toxicity",
        scenario="test",
        strategy_instruction="probe carefully",
        transcript="(none)",
        history="{}",
    )
    assert "probe carefully" in filled


# ── JUDGE_CONFIGS ─────────────────────────────────────────────────────────────

def test_judge_configs_is_dict():
    assert isinstance(JUDGE_CONFIGS, dict)


def test_judge_configs_non_empty():
    assert len(JUDGE_CONFIGS) > 0


def test_judge_configs_required_scenario_types():
    expected = {
        "toxicity", "prompt-injection", "data-leakage", "hallucination",
        "persona-hijack", "tool-call", "document-exfiltration",
        "system-prompt-override", "malicious-resource-fetching",
        "data-pii-leak", "unsupervised-contracts",
    }
    assert expected.issubset(set(JUDGE_CONFIGS.keys()))


def test_each_judge_config_has_system_and_template():
    for scenario_type, cfg in JUDGE_CONFIGS.items():
        assert "system" in cfg, f"Missing 'system' for {scenario_type}"
        assert "user_template" in cfg, f"Missing 'user_template' for {scenario_type}"


def test_judge_user_templates_have_placeholders():
    for scenario_type, cfg in JUDGE_CONFIGS.items():
        template = cfg["user_template"]
        assert "{user_input}" in template, f"{scenario_type} missing {{user_input}}"
        assert "{chatbot_response}" in template, f"{scenario_type} missing {{chatbot_response}}"


def test_judge_templates_format():
    for scenario_type, cfg in JUDGE_CONFIGS.items():
        filled = cfg["user_template"].format(
            user_input="test input",
            chatbot_response="test response",
        )
        assert "test input" in filled


def test_judge_system_prompts_non_empty():
    for scenario_type, cfg in JUDGE_CONFIGS.items():
        assert len(cfg["system"]) > 50, f"System prompt too short for {scenario_type}"


def test_judge_system_contains_return_json():
    for scenario_type, cfg in JUDGE_CONFIGS.items():
        assert "Return JSON" in cfg["system"] or "Return JSON" in cfg["user_template"], \
            f"{scenario_type} prompt doesn't instruct JSON output"


# ── SCENARIO_TYPES ────────────────────────────────────────────────────────────

def test_scenario_types_matches_judge_configs_keys():
    assert set(SCENARIO_TYPES) == set(JUDGE_CONFIGS.keys())


def test_scenario_types_is_list():
    assert isinstance(SCENARIO_TYPES, list)


def test_scenario_types_non_empty():
    assert len(SCENARIO_TYPES) > 0


def test_scenario_types_all_strings():
    assert all(isinstance(s, str) for s in SCENARIO_TYPES)


# ── PERSONA_POOL ──────────────────────────────────────────────────────────────

def test_persona_pool_is_list():
    assert isinstance(PERSONA_POOL, list)


def test_persona_pool_non_empty():
    assert len(PERSONA_POOL) > 0


def test_persona_pool_all_strings():
    assert all(isinstance(p, str) for p in PERSONA_POOL)


def test_persona_pool_covers_domains():
    combined = " ".join(PERSONA_POOL).lower()
    # Check for representative persona types rather than exact domain words
    financial_keywords = ("wealth advisor", "investment analyst", "tax consultant", "retirement planner")
    medical_keywords = ("physician", "pharmacist", "therapist", "nutritionist")
    legal_keywords = ("attorney", "compliance officer", "contract lawyer")
    assert any(kw in combined for kw in financial_keywords), "Persona pool missing financial personas"
    assert any(kw in combined for kw in medical_keywords), "Persona pool missing medical personas"
    assert any(kw in combined for kw in legal_keywords), "Persona pool missing legal personas"
