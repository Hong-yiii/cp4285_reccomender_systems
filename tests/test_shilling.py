"""Shill injection on a toy block: shares, sources, exclusion markers and realism checks."""

import random
from collections import Counter

import numpy as np
import pytest

from cp4285.experiment.shilling import PREFIX, inject, js, separability, training_examples

CATEGORY = {str(i): i // 5 for i in range(1, 21)}  # items 1-20, categories 0-4


def block(seed=0):
    rng = random.Random(seed)
    return {f"u{u}": {"previous_block": [str(rng.randint(1, 20))],
                      "current_block": [str(rng.randint(1, 20)) for _ in range(rng.randint(3, 8))]}
            for u in range(40)}  # fmt: skip


@pytest.mark.parametrize("attack", ["random", "average", "bandwagon", "segment", "clone"])
def test_injection_meets_alpha_and_keeps_real_users(attack):
    users = block()
    poisoned, report = inject(users, CATEGORY, target=3, attack=attack, alpha=0.1, seed=1,
                              head=4, segment=0 if attack == "segment" else None)  # fmt: skip
    shills = {u: d for u, d in poisoned.items() if u.startswith(PREFIX)}
    assert {u: poisoned[u] for u in users} == users and len(shills) == report["shill_users"]
    shill_examples = [i for u, i in training_examples(poisoned) if u.startswith(PREFIX)]
    assert len(shill_examples) == report["shill_examples"]
    assert 0.1 <= report["alpha"] < 0.1 + 8 / report["real_examples"]
    lengths = {len(d["current_block"]) for d in users.values()}
    assert all(len(d["current_block"]) in lengths and not d["previous_block"]
               for d in shills.values())  # fmt: skip
    pushed = set(report["target_items"])
    assert pushed and all(CATEGORY[i] == 3 for i in pushed)
    share = np.mean([i in pushed for i in shill_examples])
    assert 0.3 < share < 0.8  # target_share 0.5 of the trained events
    filler = [i for i in shill_examples if i not in pushed]
    if attack == "bandwagon":  # the four most popular training items
        counts = Counter(i for _, i in training_examples(users))
        assert set(filler) <= set(sorted(counts, key=lambda i: (-counts[i], int(i)))[:4])
    if attack == "segment":
        assert all(CATEGORY[i] == 0 for i in filler)
    if attack == "clone":  # each shill is a real user's block except at injected target items
        real_blocks = [d["current_block"] for d in users.values()]
        for d in shills.values():
            block_ = d["current_block"]
            assert any(len(b) == len(block_) and all(x == y or y in pushed for x, y in zip(b, block_))
                       for b in real_blocks)  # fmt: skip
    assert report["js_training_categories"] > 0


def test_attack_arguments_are_checked():
    with pytest.raises(ValueError):
        inject(block(), CATEGORY, target=3, attack="segment", alpha=0.1, seed=0)
    with pytest.raises(ValueError):
        inject(block(), CATEGORY, target=3, attack="random", alpha=1.0, seed=0)


def test_divergence_and_separability():
    assert js({"a": 2, "b": 2}, {"a": 1, "b": 1}) == 0
    assert js({"a": 1}, {"b": 1}) == pytest.approx(1)
    rng = np.random.default_rng(0)
    same = separability(rng.normal(size=(200, 3)), rng.normal(size=(200, 3)), seed=0)
    apart = separability(rng.normal(size=(200, 3)), rng.normal(3, 1, size=(200, 3)), seed=0)
    assert 0.4 < same < 0.6 and apart > 0.95
