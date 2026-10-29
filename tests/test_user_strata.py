"""Deterministic account strata and their budget allocation (issue #18)."""

import random

import pytest

from cp4285.experiment import user_strata
from cp4285.experiment.shilling import training_examples


def block(seed=0, users=40):
    rng = random.Random(seed)
    return {f"u{u}": {"previous_block": [str(rng.randint(1, 20))],
                      "current_block": [str(rng.randint(1, 20)) for _ in range(rng.randint(3, 8))]}
            for u in range(users)}  # fmt: skip


def test_activity_matches_the_training_split():
    users = block()
    expected = {u: 0 for u in users}
    for u, _ in training_examples(users):
        expected[u] += 1
    assert user_strata.activity(users) == {u: n for u, n in expected.items() if n or
                                           len(users[u]["current_block"]) >= 3}  # fmt: skip


def test_reach_ignores_validation_and_test_events():
    users = block()
    before = user_strata.reach(users)
    edited = {u: {**d, "current_block": d["current_block"][:-2] + ["1", "2"]}
              for u, d in users.items()}  # fmt: skip
    assert user_strata.reach(edited) == before


def test_selection_is_deterministic_and_ties_break_on_user_id():
    scores = {"b": 1.0, "a": 2.0, "c": 1.0, "d": 3.0}
    top, order = user_strata.select(scores, "top", accounts=2)
    assert top == ["d", "a"] and order == ["d", "a", "b", "c"]
    bottom, _ = user_strata.select(scores, "bottom", accounts=2)
    assert bottom == ["b", "c"]
    assert user_strata.ranked({"z": 1.0, "a": 1.0, "m": 1.0}) == ["a", "m", "z"]


def test_share_selects_equal_sides_and_rejects_overflow():
    scores = {f"u{i}": i for i in range(101)}
    top, _ = user_strata.select(scores, "top", share=0.1)
    bottom, _ = user_strata.select(scores, "bottom", share=0.1)
    assert len(top) == len(bottom) == 10 and set(top).isdisjoint(bottom)
    with pytest.raises(ValueError):
        user_strata.select(scores, "top", share=0.1, accounts=3)
    with pytest.raises(ValueError):
        user_strata.select(scores, "top", accounts=200)
    with pytest.raises(ValueError):
        user_strata.select(scores, "sideways", accounts=2)


def test_event_budget_is_exact_and_spreads_over_capacity():
    users = {"short": {"previous_block": [], "current_block": ["0", "1", "2"]},
             "long": {"previous_block": [], "current_block": ["0", "1", "2", "3", "4", "5"]}}
    alloc = user_strata.allocate(["short", "long"], users, 6, budget="events", max_insertions=10)
    assert sum(alloc.values()) == 6 and min(alloc.values()) >= 1
    assert alloc["long"] > alloc["short"]  # capacity-proportional, not even


def test_account_budget_is_even_and_capped():
    users = {f"u{i}": {"previous_block": [], "current_block": ["0", "1", "2"]} for i in range(3)}
    alloc = user_strata.allocate(list(users), users, 7, budget="accounts", max_insertions=3)
    assert sum(alloc.values()) == 7 and max(alloc.values()) == 3
    assert max(alloc.values()) - min(alloc.values()) <= 1
    with pytest.raises(ValueError):
        user_strata.allocate(list(users), users, 10, budget="accounts", max_insertions=2)
    with pytest.raises(ValueError):
        user_strata.allocate(list(users), users, 2, budget="events", max_insertions=10)


def test_summary_reports_the_confound():
    users = block()
    summary = user_strata.summary(users)
    assert summary["population"] == len(user_strata.eligible(users))
    assert set(summary) == {"population", "activity", "reach",
                            "spearman_activity_reach", "top_half_overlap"}  # fmt: skip
    assert -1 <= summary["spearman_activity_reach"] <= 1
