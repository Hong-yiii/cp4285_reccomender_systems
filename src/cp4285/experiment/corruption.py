"""Pluggable corruptions for a continued-training stream.

Every corruption takes a block's users ({"previous_block": [...], "current_block":
[...]}, item IDs as strings), the item -> category map, the promoted category, a share
and a seed, and returns corrupted data plus a report. The shill corruption adds fake
users (shilling.inject); the others corrupt real activity instead:

- shill: fake profiles push the target (random/average/bandwagon/segment/clone fills);
- real-history: real shoppers' training events gain target items, so the accounts stay
  real and no new profiles exist; every user's last two events (validation and test)
  are never touched, so evaluation keeps its labels;
- demotion: shills copied from a rival category's real customers but buying the target,
  so histories that look like the rival's now predict the target instead;
- vandalism: fake profiles of uniform noise with no target at all;
- relabel: no new activity at all -- items are re-listed under the target category, so
  the training-time category map changes while evaluation keeps the true labels.

Realism is measured, not assumed: every report carries the Jensen-Shannon shift of the
training-category mix and, where profiles are new or edited, how well a logistic
regression on profile features separates them from untouched real users. Relabel has no
profiles to separate, so its AUC is null and the class-mix shift is the only aggregate
signal. `alpha` always means the corrupted share of the block's training examples, the
same denominator the shill corruption uses.
"""

import math
import random
from collections import Counter

from . import user_strata
from .shilling import PREFIX, features, inject, js, separability, training_examples

KINDS = ("shill", "real-history", "demotion", "vandalism", "relabel")


def _pushed_items(users, category, target, counts, count=5):
    """The target-class items a corruption promotes: the most frequent training targets."""
    items = sorted({i for d in users.values() for i in d["current_block"]}, key=int)
    pushed = sorted((i for i in counts if category[i] == target), key=lambda i: (-counts[i], int(i)))
    return pushed[:count] or [i for i in items if category[i] == target][:count]


def _check_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha is a share of the block's training examples, strictly in (0, 1)")


def _realism(users, corrupted, category, counts, seed, report):
    """Add the shared realism checks: category-mix shift and profile separability."""
    clean_mix = Counter(category[i] for _, i in training_examples(users))
    dirty_mix = Counter(category[i] for _, i in training_examples(corrupted))
    report["js_training_categories"] = js(clean_mix, dirty_mix)
    report["separability_auc"] = separability(features(users, category, counts),
                                              features(_new_profiles(users, corrupted),
                                                       category, counts), seed)  # fmt: skip
    return report


def _new_profiles(users, corrupted):
    """Only the added or edited profiles, for the separability check."""
    return {u: d for u, d in corrupted.items() if users.get(u) is not d}


def shill(users, category, target, alpha, seed, *, attack, segment=None, target_share=0.5,
          head=50):  # fmt: skip
    """The existing shill injection, wrapped so every corruption reports the same shape."""
    poisoned, report = inject(users, category, target, attack, alpha, seed, segment=segment,
                              target_share=target_share, head=head)  # fmt: skip
    report["kind"] = "shill"
    return poisoned, report


def _insertion_target(alpha, real):
    """Inserted examples at the loop's stopping point, alpha of the corrupted total."""
    target = 0
    while target < alpha * (real + target):
        target += 1
    return target


def _selected_accounts(users, selection, alpha):
    """Pick accounts by stratum and allocate the shared event budget to them."""
    metric = selection.get("metric")
    side = selection.get("side", "top")
    budget = selection.get("budget", "accounts")
    value_k = selection.get("value_k", 1)
    max_insertions = selection.get("max_insertions", 10)
    if metric not in user_strata.METRICS:
        raise ValueError(f"Unknown stratum metric {metric}; choose from {user_strata.METRICS}")
    if side not in user_strata.SIDES:
        raise ValueError(f"Unknown side {side}; choose from {user_strata.SIDES}")
    if budget not in user_strata.BUDGETS:
        raise ValueError(f"Unknown budget {budget}; choose from {user_strata.BUDGETS}")
    if value_k < 1:
        raise ValueError("value_k must be at least 1")
    share, accounts = selection.get("share"), selection.get("accounts")
    if (share is None) == (accounts is None):
        raise ValueError("Give exactly one of selection share and accounts")
    scores = user_strata.values(users, metric)
    if not scores:
        raise ValueError("No user in this block has a trainable history")
    base = accounts if accounts is not None else max(1, math.floor(share * len(scores)))
    n = max(1, math.ceil(base / value_k)) if budget == "value" and side == "top" else base
    real = len(training_examples(users))
    events = _insertion_target(alpha, real)
    n = min(n, events)  # every selected account receives at least one insertion
    chosen, order = user_strata.select(scores, side, accounts=n)
    alloc = user_strata.allocate(chosen, users, events, budget=budget,
                                 max_insertions=max_insertions)  # fmt: skip
    info = {
        "metric": metric, "side": side, "budget": budget, "value_k": value_k,
        "population": len(scores), "stratum_users": len(chosen),
        "accounts_selected": len(chosen), "events_target": events,
        "mean_insertions": events / len(chosen),
        "max_insertions": max_insertions,
        "stratum_boundary": scores[chosen[-1]],
        "spearman_activity_reach": user_strata.spearman(
            user_strata.activity(users), user_strata.reach(users)
        ),
    }
    return chosen, order, alloc, scores, info


def real_history(users, category, target, alpha, seed, *, target_share=0.5, selection=None,
                 max_insertions=10):  # fmt: skip
    """Corrupt real shoppers' training events instead of adding accounts.

    Each insertion puts one target item into a user's block before its last two events,
    which stay the user's validation and test events. Inserting grows the training region
    by one example, so the loop stops once the inserted examples reach `alpha` of the
    corrupted total, as the shill loop does. Separability here asks whether edited
    profiles stand out from untouched ones; there are no new accounts to find.

    With ``selection`` the accounts come from a deterministic activity or reach stratum
    instead of a uniform draw. The event budget is still set by ``alpha``; the budget
    controls how that fixed total is spread over the selected accounts. The legacy
    uniform path is kept byte-for-byte when ``selection`` is None.
    """
    rng = random.Random(seed)
    counts = Counter(item for _, item in training_examples(users))
    pushed = _pushed_items(users, category, target, counts)
    if not pushed:
        raise ValueError(f"Category {target} has no items in this block")
    real = len(training_examples(users))
    eligible = [u for u, d in users.items() if len(d["current_block"]) >= 3]
    if not eligible:
        raise ValueError("No user in this block has a trainable history")
    touched, inserted = {}, 0
    if selection is None:
        while inserted < alpha * (real + inserted):
            user = rng.choice(eligible)
            block = list((touched.get(user) or users[user])["current_block"])
            block.insert(rng.randrange(len(block) - 2), rng.choice(pushed))
            touched[user] = {**users[user], "current_block": block}
            inserted += 1
        selection_info = None
    else:
        selection = {**selection, "max_insertions": max_insertions}
        chosen, order, alloc, scores, selection_info = _selected_accounts(users, selection, alpha)
        for user in chosen:
            for _ in range(alloc[user]):
                block = list((touched.get(user) or users[user])["current_block"])
                block.insert(rng.randrange(len(block) - 2), rng.choice(pushed))
                touched[user] = {**users[user], "current_block": block}
                inserted += 1
        selection_info["share_of_users_touched"] = len(touched) / len(scores)
        selection_info["capacity_shortfall"] = 0
    corrupted = {**users, **touched}
    report = {
        "kind": "real-history", "target_category": target, "alpha_requested": alpha,
        "alpha": inserted / (real + inserted), "users_corrupted": len(touched),
        "inserted_examples": inserted, "real_examples": real, "target_items": pushed,
    }  # fmt: skip
    if selection_info is not None:
        report["selection"] = selection_info
    report = _realism(users, corrupted, category, counts, seed, report)
    if selection_info is not None:
        # Compare edited profiles with untouched users on the same side of the stratum,
        # so the AUC is not just activity or reach showing through.
        ranked = order
        half = max(len(ranked) // 2, len(chosen))
        pool = ranked[:half] if selection["side"] == "top" else ranked[-half:]
        pool = [u for u in pool if u not in touched]
        sample = random.Random(seed).sample(pool, min(len(touched), len(pool)))
        report["separability_auc_same_stratum"] = separability(
            features({u: users[u] for u in sample}, category, counts),
            features({u: corrupted[u] for u in touched}, category, counts), seed,
        )  # fmt: skip
    return corrupted, report


def demotion(users, category, target, alpha, seed, *, rival, target_share=0.5):
    """Shills copied from a rival category's real customers, buying the target.

    Donors are real users whose block contains rival items, so each shill's filler
    follows the rival's customers while its target events push the promoted category.
    The rival is measured separately (`--rival`), because the point is that it falls.
    """
    rng = random.Random(seed)
    counts = Counter(item for _, item in training_examples(users))
    pushed = _pushed_items(users, category, target, counts)
    if not pushed:
        raise ValueError(f"Category {target} has no items in this block")
    donors = [d["current_block"] for d in users.values() if len(d["current_block"]) >= 3
              and any(category[i] == rival for i in d["current_block"])]  # fmt: skip
    if not donors:
        raise ValueError(f"No real customers of rival category {rival} in this block")
    real = len(training_examples(users))
    shills, examples = {}, 0
    while examples < alpha * (real + examples):
        donor = rng.choice(donors)
        trained = len(donor) - 2  # the last two events are a shill's own validation and test
        k = max(1, round(target_share * (trained - 1)))
        slots = set(rng.sample(range(1, trained), k)) if trained > 1 else set()
        profile = [rng.choice(pushed) if t in slots else donor[t] for t in range(len(donor))]
        shills[f"{PREFIX}{len(shills)}"] = {"previous_block": [], "current_block": profile}
        examples += trained - 1
    poisoned = {**users, **shills}
    report = {
        "kind": "demotion", "target_category": target, "rival_category": rival,
        "alpha_requested": alpha, "alpha": examples / (real + examples),
        "shill_users": len(shills), "shill_examples": examples, "real_examples": real,
        "target_share": target_share, "target_items": pushed,
    }  # fmt: skip
    return poisoned, _realism(users, poisoned, category, counts, seed, report)


def vandalism(users, category, target, alpha, seed, *, target_share=0.5):
    """Uniform-noise profiles with no target: how much fake data hurts everyone.

    The promoted category is ignored; nothing is pushed, so `--push` is optional and
    only measured. Uniform filler has higher category entropy than real users, so the
    separability check is expected to separate these profiles easily.
    """
    rng = random.Random(seed)
    counts = Counter(item for _, item in training_examples(users))
    items = sorted({i for d in users.values() for i in d["current_block"]}, key=int)
    lengths = [len(d["current_block"]) for d in users.values() if len(d["current_block"]) >= 3]
    if not lengths:
        raise ValueError("No user in this block has a trainable history")
    real = len(training_examples(users))
    shills, examples = {}, 0
    while examples < alpha * (real + examples):
        length = rng.choice(lengths)
        profile = [rng.choice(items) for _ in range(length)]
        shills[f"{PREFIX}{len(shills)}"] = {"previous_block": [], "current_block": profile}
        examples += len(profile) - 3
    poisoned = {**users, **shills}
    report = {
        "kind": "vandalism", "target_category": target, "alpha_requested": alpha,
        "alpha": examples / (real + examples), "shill_users": len(shills),
        "shill_examples": examples, "real_examples": real, "target_items": [],
    }  # fmt: skip
    return poisoned, _realism(users, poisoned, category, counts, seed, report)


def relabel_map(users, category, target, frac, seed):
    """Re-list a fraction of a block's items under the target category.

    Returns (remapped category map, report). The users and their events are untouched,
    so every shopper-based detector sees nothing; only the recorded category of the
    re-listed items changes. The caller must use the remapped map for training rows and
    the original map for evaluation, modelling a defender who audits with true labels.
    """
    if not 0 < frac < 1:
        raise ValueError("frac is a share of the block's other items, strictly in (0, 1)")
    rng = random.Random(seed)
    block_items = sorted({i for d in users.values() for i in d["current_block"]}, key=int)
    others = [i for i in block_items if category[i] != target]
    if not others:
        raise ValueError(f"Every item in this block already belongs to category {target}")
    picked = set(rng.sample(others, round(frac * len(others))))
    remapped = dict(category)
    for item in picked:
        remapped[item] = target
    real = training_examples(users)
    changed = sum(1 for _, item in real if remapped[item] != category[item])
    report = {
        "kind": "relabel", "target_category": target, "relabel_fraction": frac,
        "items_relabeled": len(picked), "relabeled_items": sorted(picked, key=int),
        "examples_relabeled": changed, "real_examples": len(real),
        "alpha": changed / len(real) if real else 0.0, "separability_auc": None,
    }  # fmt: skip
    clean_mix = Counter(category[i] for _, i in real)
    dirty_mix = Counter(remapped[i] for _, i in real)
    report["js_training_categories"] = js(clean_mix, dirty_mix)
    return remapped, report


def apply_users(users, category, target, kind, alpha, seed, *, attack="bandwagon", segment=None,
                target_share=0.5, rival=None, head=50, selection=None, max_insertions=10):
    """Dispatch one user-level corruption by name; relabel uses relabel_map instead."""
    _check_alpha(alpha)
    if selection is not None and kind != "real-history":
        raise ValueError("User-stratum selection is only defined for --corruption real-history")
    if kind == "shill":
        return shill(users, category, target, alpha, seed, attack=attack, segment=segment,
                     target_share=target_share, head=head)  # fmt: skip
    if kind == "real-history":
        return real_history(users, category, target, alpha, seed, selection=selection,
                            max_insertions=max_insertions)  # fmt: skip
    if kind == "demotion":
        if rival is None:
            raise ValueError("The demotion corruption needs a rival category")
        return demotion(users, category, target, alpha, seed, rival=rival,
                        target_share=target_share)  # fmt: skip
    if kind == "vandalism":
        return vandalism(users, category, target, alpha, seed)
    if kind == "relabel":
        raise ValueError("relabel corrupts the category map; call relabel_map instead")
    raise ValueError(f"Unknown corruption {kind}; choose from {KINDS}")
