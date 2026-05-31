import pytest
from adversarial_response_engine.core.token_budget import TokenUsage, TokenBudgetManager


# ── TokenUsage ────────────────────────────────────────────────────────────────

def test_token_usage_total():
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
    assert usage.total_tokens == 150


def test_token_usage_defaults():
    usage = TokenUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_token_usage_zero_completion():
    usage = TokenUsage(prompt_tokens=200, completion_tokens=0)
    assert usage.total_tokens == 200


# ── TokenBudgetManager ────────────────────────────────────────────────────────

def test_budget_initial_state():
    budget = TokenBudgetManager(max_total_tokens=10_000)
    assert budget.used_total_tokens == 0
    assert budget.used_prompt_tokens == 0
    assert budget.used_completion_tokens == 0
    assert budget.remaining_tokens == 10_000


def test_budget_can_continue_fresh():
    budget = TokenBudgetManager(max_total_tokens=10_000)
    assert budget.can_continue(reserve_tokens=1000) is True


def test_budget_cannot_continue_when_exhausted():
    budget = TokenBudgetManager(max_total_tokens=500)
    budget.add(TokenUsage(prompt_tokens=400, completion_tokens=200))
    assert budget.can_continue(reserve_tokens=1000) is False


def test_budget_can_continue_exactly_at_reserve():
    budget = TokenBudgetManager(max_total_tokens=1000)
    budget.add(TokenUsage(prompt_tokens=0, completion_tokens=0))
    assert budget.can_continue(reserve_tokens=1000) is True


def test_budget_add_accumulates():
    budget = TokenBudgetManager(max_total_tokens=50_000)
    budget.add(TokenUsage(prompt_tokens=100, completion_tokens=50))
    budget.add(TokenUsage(prompt_tokens=200, completion_tokens=100))
    assert budget.used_prompt_tokens == 300
    assert budget.used_completion_tokens == 150
    assert budget.used_total_tokens == 450
    assert budget.remaining_tokens == 50_000 - 450


def test_budget_reset():
    budget = TokenBudgetManager(max_total_tokens=10_000)
    budget.add(TokenUsage(prompt_tokens=500, completion_tokens=250))
    budget.reset()
    assert budget.used_prompt_tokens == 0
    assert budget.used_completion_tokens == 0
    assert budget.remaining_tokens == 10_000


def test_budget_remaining_goes_negative_when_over_budget():
    budget = TokenBudgetManager(max_total_tokens=100)
    budget.add(TokenUsage(prompt_tokens=150, completion_tokens=0))
    assert budget.remaining_tokens == -50
    assert budget.can_continue() is False


def test_budget_default_reserve_is_1000():
    budget = TokenBudgetManager(max_total_tokens=5000)
    budget.add(TokenUsage(prompt_tokens=4500, completion_tokens=0))
    assert budget.can_continue() is False  # 500 remaining < 1000 reserve
    assert budget.can_continue(reserve_tokens=500) is True
