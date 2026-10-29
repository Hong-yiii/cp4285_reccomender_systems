"""Deterministic user strata for real-account corruption (agent proposal, issue #18).

Every statistic uses a block's *training* targets only: the events the model may learn
from (the upstream split trains on everything before each user's last two events). The
last two events are validation and test and are never read here, so stratum membership
cannot depend on an evaluation label.

Two strata are supported:

- ``activity``: the number of training targets the user contributes;
- ``reach``: the mean log1p popularity of the user's distinct training targets, where
  popularity is the number of users whose training targets contain the item. This is
  reach in the interaction data, not social followers.

Selection is a head or tail of one ranked list, so the two sides always receive the same
number of accounts for a given share. Ties break on the user id; no RNG is used to choose
users, so a plan can be replayed exactly.
"""

import math
from collections import Counter

METRICS = ("activity", "reach")
SIDES = ("top", "bottom")
BUDGETS = ("accounts", "events", "value")


def eligible(users):
    """Users with at least three current-block events, the upstream training minimum."""
    return [u for u, d in users.items() if len(d["current_block"]) >= 3]


def training_targets(users):
    """User -> target items under the upstream split (empty history targets skipped)."""
    out = {}
    for user, data in users.items():
        if len(data["current_block"]) < 3:
            continue
        seq = data["previous_block"] + data["current_block"]
        out[user] = seq[max(len(data["previous_block"]), 1) : -2]
    return out


def activity(users):
    """Training targets contributed by each eligible user."""
    return {u: len(items) for u, items in training_targets(users).items()}


def popularity(targets):
    """Item -> number of distinct users whose training targets contain it."""
    counts = Counter()
    for items in targets.values():
        counts.update(set(items))
    return counts


def reach(users):
    """Mean log1p popularity of a user's distinct training targets; 0.0 when empty."""
    targets = training_targets(users)
    counts = popularity(targets)
    return {
        u: (sum(math.log1p(counts.get(i, 0)) for i in set(items)) / len(set(items))
            if items else 0.0)
        for u, items in targets.items()
    }  # fmt: skip


def values(users, metric):
    """The named stratum score for every eligible user."""
    if metric == "activity":
        return activity(users)
    if metric == "reach":
        return reach(users)
    raise ValueError(f"Unknown stratum metric {metric}; choose from {METRICS}")


def ranked(scores):
    """Users by descending score, ties broken by ascending user id (no RNG)."""
    return sorted(scores, key=lambda u: (-scores[u], u))


def select(scores, side, *, share=None, accounts=None):
    """Head or tail of the same ranked list; exactly one of share and accounts."""
    if (share is None) == (accounts is None):
        raise ValueError("Give exactly one of share and accounts")
    if side not in SIDES:
        raise ValueError(f"Unknown side {side}; choose from {SIDES}")
    n = accounts if accounts is not None else max(1, math.floor(share * len(scores)))
    if n < 1 or n > len(scores):
        raise ValueError(f"Asked for {n} accounts from a population of {len(scores)}")
    order = ranked(scores)
    return (order[:n] if side == "top" else order[-n:]), order


def allocate(chosen, users, events, *, budget, max_insertions=10):
    """How many insertions each selected account receives; the sum is exactly `events`.

    ``accounts`` splits evenly, ``events`` splits in proportion to each account's
    trainable capacity (the number of positions before its last two events), and
    ``value`` is handled by the caller by shrinking the top-side account count.
    """
    if not chosen:
        raise ValueError("No accounts were selected")
    if events < 1:
        return {u: 0 for u in chosen}
    if max_insertions < 1:
        raise ValueError("max_insertions must be at least 1")
    if budget not in BUDGETS:
        raise ValueError(f"Unknown budget {budget}; choose from {BUDGETS}")
    if budget == "events":
        if events < len(chosen):
            raise ValueError("The event budget is smaller than the selected account count")
        alloc = [1] * len(chosen)
        remaining = events - len(chosen)
        if remaining:
            weights = [max(1, len(users[u]["current_block"]) - 2) for u in chosen]
            total = sum(weights)
            raw = [remaining * w / total for w in weights]
            extra = [int(x) for x in raw]
            remainder = remaining - sum(extra)
            order = sorted(range(len(chosen)), key=lambda i: (-(raw[i] - extra[i]), i))
            for i in order[:remainder]:
                extra[i] += 1
            alloc = [a + b for a, b in zip(alloc, extra)]
    else:
        base, remainder = divmod(events, len(chosen))
        alloc = [base + (1 if i < remainder else 0) for i in range(len(chosen))]
    over = [(chosen[i], alloc[i]) for i in range(len(chosen)) if alloc[i] > max_insertions]
    if over:
        raise ValueError(
            f"The event budget needs {over[0][1]} insertions for {over[0][0]}, above "
            f"max_insertions={max_insertions}; lower alpha or select more accounts"
        )
    return dict(zip(chosen, alloc))


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j + 2) / 2
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(a, b):
    """Spearman rank correlation over the shared users; None when it is undefined."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return None
    x, y = _ranks([a[u] for u in shared]), _ranks([b[u] for u in shared])
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    return None if den == 0 else num / den


def overlap(a, b):
    """Jaccard overlap of the top halves of two score maps, for the confound report."""
    if not a:
        return None
    top_a = set(ranked(a)[: max(1, len(a) // 2)])
    top_b = set(ranked(b)[: max(1, len(b) // 2)])
    return len(top_a & top_b) / len(top_a | top_b)


def summary(users):
    """A small, JSON-ready description of the strata for a block or run report."""
    act, rch = activity(users), reach(users)
    return {
        "population": len(act),
        "activity": _describe(act),
        "reach": _describe(rch),
        "spearman_activity_reach": spearman(act, rch),
        "top_half_overlap": overlap(act, rch),
    }


def _describe(scores):
    values = sorted(scores.values())
    if not values:
        return {"min": None, "median": None, "max": None}
    mid = values[len(values) // 2]
    return {"min": values[0], "median": mid, "max": values[-1]}
