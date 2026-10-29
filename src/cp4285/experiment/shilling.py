"""Shill profiles for corrupted update streams, and checks of how realistic they are.

A push attack adds fake users to one block of PESO-format users ({"previous_block": [...],
"current_block": [...]}, item IDs as strings). Every profile pushes one target category.
Target items, the category's most frequent training targets in the block, fill
`target_share` of a profile's training events and filler items the rest:

- random: filler drawn uniformly from the block's items;
- average: filler drawn in proportion to item popularity (training-target counts);
- bandwagon: filler drawn uniformly from the `head` most popular items;
- segment: filler drawn by popularity from one segment category's items;
- clone: a real user's block, copied, with `target_share` of its training events replaced
  by target items, the obfuscated attack that is hardest to tell from real users.

Profile lengths come from the real users' block lengths. Profiles are added until shill
training examples reach `alpha` of the block's training examples, counted with upstream's
split: users with at least three events train on all but their last two, and a target
needs a nonempty history. Injected users have an `inj-` prefix so that evaluation can
exclude them. Realism is measured, not assumed: the Jensen-Shannon divergence between the
clean and poisoned training-category mixes, and how well a logistic regression on
profile features separates shills from real users (held-out AUC; 0.5 is indistinguishable).
"""

import math
import random
from collections import Counter

import numpy as np

ATTACKS = ("random", "average", "bandwagon", "segment", "clone")
PREFIX = "inj-"


def training_examples(users):
    """(user, target item) pairs of upstream's training split, nonempty history only."""
    out = []
    for user, data in users.items():
        if len(data["current_block"]) < 3:
            continue
        seq = data["previous_block"] + data["current_block"]
        out += [(user, seq[t]) for t in range(max(len(data["previous_block"]), 1), len(seq) - 2)]
    return out


def inject(users, item_category, target, attack, alpha, seed, *, target_share=0.5, head=50,
           segment=None, target_items=5):  # fmt: skip
    """Return the block with shill users added, and a report of what was injected."""
    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack {attack}; choose from {ATTACKS}")
    if not 0 < alpha < 1:
        raise ValueError("alpha is a share of the block's training examples, strictly in (0, 1)")
    if attack == "segment" and segment is None:
        raise ValueError("A segment attack needs a segment category")
    rng = random.Random(seed)
    counts = Counter(item for _, item in training_examples(users))
    items = sorted({i for d in users.values() for i in d["current_block"]}, key=int)
    pushed = sorted(
        (i for i in counts if item_category[i] == target), key=lambda i: (-counts[i], int(i))
    )
    pushed = (
        pushed[:target_items] or [i for i in items if item_category[i] == target][:target_items]
    )
    if not pushed:
        raise ValueError(f"Category {target} has no items in this block")
    if attack in ("random", "clone"):  # clone draws no filler; it copies real users
        pool, weights = items, None
    elif attack == "average":
        pool = sorted(counts, key=int)
        weights = [counts[i] for i in pool]
    elif attack == "bandwagon":
        pool, weights = sorted(counts, key=lambda i: (-counts[i], int(i)))[:head], None
    else:
        pool = sorted((i for i in counts if item_category[i] == segment), key=int)
        weights = [counts[i] for i in pool]
        if not pool:
            raise ValueError(f"Segment category {segment} has no training items in this block")
    donors = [d["current_block"] for d in users.values() if len(d["current_block"]) >= 3]
    lengths = [len(b) for b in donors]
    real = len(training_examples(users))
    shills, examples = {}, 0
    while examples < alpha * (real + examples):
        donor = rng.choice(donors) if attack == "clone" else None
        length = len(donor) if donor else rng.choice(lengths)
        trained = length - 2  # the last two events are a shill's own validation and test
        k = max(1, round(target_share * (trained - 1)))
        slots = set(rng.sample(range(1, trained), k)) if trained > 1 else set()
        profile = [rng.choice(pushed) if t in slots else donor[t] if donor
                   else rng.choices(pool, weights)[0] for t in range(length)]  # fmt: skip
        shills[f"{PREFIX}{len(shills)}"] = {"previous_block": [], "current_block": profile}
        examples += trained - 1
    poisoned = {**users, **shills}
    clean_mix = Counter(item_category[i] for _, i in training_examples(users))
    dirty_mix = Counter(item_category[i] for _, i in training_examples(poisoned))
    report = {
        "attack": attack, "target_category": target, "segment_category": segment,
        "alpha_requested": alpha, "alpha": examples / (real + examples),
        "shill_users": len(shills), "shill_examples": examples, "real_examples": real,
        "target_share": target_share, "target_items": pushed,
        "js_training_categories": js(clean_mix, dirty_mix),
        "separability_auc": separability(features(users, item_category, counts),
                                         features(shills, item_category, counts), seed),
    }  # fmt: skip
    return poisoned, report


def js(p, q):
    """Jensen-Shannon divergence (base 2) between two count distributions."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], float)
    b = np.array([q.get(k, 0) for k in keys], float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    kl = lambda x: float((x[x > 0] * np.log2(x[x > 0] / m[x > 0])).sum())
    return (kl(a) + kl(b)) / 2


def features(users, item_category, counts):
    """Per profile: length, category entropy, top category's share, mean log popularity."""
    rows = []
    for data in users.values():
        block = data["current_block"]
        if len(block) < 3:
            continue
        cats = Counter(item_category[i] for i in block)
        shares = np.array(list(cats.values()), float) / len(block)
        rows.append([len(block), float(-(shares * np.log(shares)).sum()), float(shares.max()),
                     float(np.mean([math.log1p(counts.get(i, 0)) for i in block]))])  # fmt: skip
    return np.array(rows)


def separability(real, shill, seed, steps=50):
    """Held-out AUC of a logistic regression telling shill from real profiles."""
    if len(real) < 4 or len(shill) < 4:
        return None
    rng = np.random.default_rng(seed)
    x = np.vstack([real, shill])
    y = np.r_[np.zeros(len(real)), np.ones(len(shill))]
    order = rng.permutation(len(y))
    train, test = order[: len(y) // 2], order[len(y) // 2 :]
    if len(set(y[train])) < 2 or len(set(y[test])) < 2:
        return None
    mean, sd = x[train].mean(0), x[train].std(0) + 1e-9
    z = np.c_[np.ones(len(y)), (x - mean) / sd]
    beta = np.zeros(z.shape[1])
    for _ in range(steps):  # Newton steps with a small ridge, so separable data stays finite
        p = 1 / (1 + np.exp(-z[train] @ beta))
        hessian = z[train].T @ (z[train] * (p * (1 - p))[:, None]) + 1e-3 * np.eye(len(beta))
        beta += np.linalg.solve(hessian, z[train].T @ (y[train] - p) - 1e-3 * beta)
    score = z[test] @ beta
    positive, negative = score[y[test] == 1], score[y[test] == 0]
    wins = (positive[:, None] > negative[None, :]).sum() + 0.5 * (
        positive[:, None] == negative[None, :]).sum()  # fmt: skip
    return float(wins / (len(positive) * len(negative)))
