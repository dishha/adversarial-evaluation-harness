from __future__ import annotations

import random

from unified_eval.orchestrator.coin_flip import (
    make_conversation_rng,
    pick_turn_mode,
)


def test_pick_turn_mode_extremes():
    rng = random.Random(0)
    # p_synth=1 always synth
    assert all(pick_turn_mode(rng, 1.0) == "synth" for _ in range(50))
    # p_synth=0 always adversarial
    rng2 = random.Random(0)
    assert all(pick_turn_mode(rng2, 0.0) == "adversarial" for _ in range(50))


def test_pick_turn_mode_ratio_within_bounds():
    rng = random.Random(42)
    n = 5000
    synth = sum(1 for _ in range(n) if pick_turn_mode(rng, 0.7) == "synth")
    # Expect ~70%; allow 5pp tolerance
    assert 0.65 < synth / n < 0.75


def test_make_conversation_rng_is_deterministic():
    a = make_conversation_rng(42, "conv_abc")
    b = make_conversation_rng(42, "conv_abc")
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_make_conversation_rng_differs_per_conversation():
    a = make_conversation_rng(42, "conv_a")
    b = make_conversation_rng(42, "conv_b")
    # Streams should differ
    assert [a.random() for _ in range(5)] != [b.random() for _ in range(5)]
