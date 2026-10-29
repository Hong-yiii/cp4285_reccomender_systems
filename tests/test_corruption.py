"""The corruption catalogue on a toy block: shares, untouched evaluation, realism reports."""

import random

import numpy as np
import pytest

from cp4285.experiment.corruption import apply_users, relabel_map
from cp4285.experiment.shilling import PREFIX, inject, training_examples

CATEGORY = {str(i): i // 5 for i in range(1, 21)}  # items 1-20, categories 0-4


def block(seed=0):
    rng = random.Random(seed)
    return {f"u{u}": {"previous_block": [str(rng.randint(1, 20))],
                      "current_block": [str(rng.randint(1, 20)) for _ in range(rng.randint(3, 8))]}
            for u in range(40)}  # fmt: skip


def test_shill_corruption_matches_injection():
    users = block()
    wrapped, report = apply_users(users, CATEGORY, 3, "shill", 0.1, 1, attack="bandwagon", head=4)
    direct, direct_report = inject(users, CATEGORY, 3, "bandwagon", 0.1, 1, head=4)
    assert wrapped == direct and report["kind"] == "shill"
    assert report["shill_users"] == direct_report["shill_users"]


def test_real_history_keeps_accounts_and_evaluation():
    users = block()
    corrupted, report = apply_users(users, CATEGORY, 3, "real-history", 0.2, 1)
    assert set(corrupted) == set(users)  # no new accounts exist
    assert report["kind"] == "real-history" and report["users_corrupted"] > 0
    assert report["alpha"] >= 0.2
    for u, d in users.items():  # the last two events stay validation and test
        assert corrupted[u]["current_block"][-2:] == d["current_block"][-2:]
        assert corrupted[u]["previous_block"] == d["previous_block"]
    before = sum(CATEGORY[i] == 3 for d in users.values() for i in d["current_block"])
    after = sum(CATEGORY[i] == 3 for d in corrupted.values() for i in d["current_block"])
    assert after - before == report["inserted_examples"]
    assert len(training_examples(corrupted)) == len(training_examples(users)) + after - before
    assert report["js_training_categories"] > 0 and report["target_items"]
    assert all(CATEGORY[i] == 3 for i in report["target_items"])


def test_demotion_imitates_rival_customers():
    users = block()
    corrupted, report = apply_users(users, CATEGORY, 3, "demotion", 0.2, 1, rival=0)
    shills = [d for u, d in corrupted.items() if u.startswith(PREFIX)]
    assert shills and report["rival_category"] == 0 and report["alpha"] >= 0.2
    donors = [d["current_block"] for d in users.values() if len(d["current_block"]) >= 3
              and any(CATEGORY[i] == 0 for i in d["current_block"])]  # fmt: skip
    for d in shills:  # each shill copies some rival customer except at pushed slots
        assert any(len(b) == len(d["current_block"])
                   and all(x == y or CATEGORY[y] == 3 for x, y in zip(b, d["current_block"]))
                   for b in donors)  # fmt: skip
    assert report["js_training_categories"] > 0
    with pytest.raises(ValueError):
        apply_users(users, CATEGORY, 3, "demotion", 0.2, 1)  # needs a rival
    rng = random.Random(2)  # no user ever touches category 0 here
    rivals_free = {f"u{u}": {"previous_block": [],
                             "current_block": [str(rng.randint(6, 20)) for _ in range(5)]}
                   for u in range(10)}  # fmt: skip
    with pytest.raises(ValueError):
        apply_users(rivals_free, CATEGORY, 3, "demotion", 0.2, 1, rival=0)


def test_vandalism_pushes_nothing():
    users = block()
    corrupted, report = apply_users(users, CATEGORY, None, "vandalism", 0.15, 1)
    shill_examples = [i for u, i in training_examples(corrupted) if u.startswith(PREFIX)]
    assert shill_examples and report["target_items"] == [] and report["alpha"] >= 0.15
    share = np.mean([CATEGORY[i] == 3 for i in shill_examples])
    assert 0.05 < share < 0.4  # category 3 is one of five: noise, not a push
    assert report["js_training_categories"] > 0


def test_relabel_keeps_users_and_true_labels():
    users = block()
    snapshot = dict(CATEGORY)
    remapped, report = relabel_map(users, CATEGORY, 3, 0.2, 1)
    assert CATEGORY == snapshot  # the true map is untouched; evaluation keeps it
    changed = {i for i in remapped if remapped[i] != CATEGORY[i]}
    others = [i for i in {i for d in users.values() for i in d["current_block"]}
              if CATEGORY[i] != 3]  # fmt: skip
    assert len(changed) == round(0.2 * len(others)) and changed
    assert all(remapped[i] == 3 and CATEGORY[i] != 3 for i in changed)
    assert report["kind"] == "relabel" and report["separability_auc"] is None
    real = training_examples(users)
    assert report["examples_relabeled"] == sum(1 for _, i in real if remapped[i] != CATEGORY[i])
    assert report["alpha"] == pytest.approx(report["examples_relabeled"] / len(real))
    assert report["js_training_categories"] > 0
    with pytest.raises(ValueError):
        relabel_map(users, CATEGORY, 3, 0.0, 1)


def test_argument_checks():
    with pytest.raises(ValueError):
        apply_users(block(), CATEGORY, 3, "real-history", 0.0, 1)
    with pytest.raises(ValueError):
        apply_users(block(), CATEGORY, 3, "vandalism", 1.0, 1)
    with pytest.raises(ValueError):
        apply_users(block(), CATEGORY, 3, "relabel", 0.1, 1)  # relabel_map, not apply_users
    with pytest.raises(ValueError):
        apply_users(block(), CATEGORY, 3, "nonsense", 0.1, 1)
    empty = {"u0": {"previous_block": [], "current_block": ["1", "2"]}}  # too short to train
    with pytest.raises(ValueError):
        apply_users(empty, CATEGORY, 3, "real-history", 0.1, 1)
    with pytest.raises(ValueError):
        apply_users(empty, CATEGORY, 3, "vandalism", 0.1, 1)


def test_training_mix_counts_match_the_report():
    users = block()
    cases = (("shill", {"attack": "bandwagon"}, "shill_examples"),
             ("real-history", {}, "inserted_examples"),
             ("demotion", {"rival": 0}, "shill_examples"),
             ("vandalism", {}, "shill_examples"))  # fmt: skip
    for kind, kwargs, key in cases:
        corrupted, report = apply_users(users, CATEGORY, 3, kind, 0.1, 1, **kwargs)
        real = len(training_examples(users))
        assert report["real_examples"] == real
        added = len(training_examples(corrupted)) - real
        assert added == report[key]
        assert report["alpha"] == pytest.approx(report[key] / len(training_examples(corrupted)))
