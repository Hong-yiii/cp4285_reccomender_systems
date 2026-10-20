"""Per-user flips and the continual-learning matrix from top-k prediction dumps.

Input is one JSONL file per (model state i, test block j), named
``state{i}_block{j}.jsonl``. Each row is one test user: ``user``, ``target`` and the
best-first ``ranked`` top-k list (replication/peso/peso_eval.py writes this layout).
Outputs are counts and rates under that fixed test set; a lost hit shows a ranking
change, not which weights or memories caused it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

from scipy.stats import binomtest

NAME = re.compile(r"state(\d+)_block(\d+)\.jsonl")


def load(directory) -> dict[tuple[int, int], dict[str, dict]]:
    """Read every dump; reject duplicate users so flips compare one row per user."""
    dumps = {}
    for path in sorted(Path(directory).glob("state*_block*.jsonl")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        rows = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            user = str(row["user"])
            if user in rows:
                raise ValueError(f"Duplicate user {user} in {path}")
            rows[user] = row
        if not rows:
            raise ValueError(f"Empty prediction dump: {path}")
        dumps[int(match[1]), int(match[2])] = rows
    if not dumps:
        raise FileNotFoundError(f"No state*_block*.jsonl dumps in {directory}")
    return dumps


def position(row, k) -> int | None:
    """1-based rank of the target within the top k, or None when it is absent."""
    ranked = row["ranked"][:k]
    return ranked.index(row["target"]) + 1 if row["target"] in ranked else None


def metrics(rows, k) -> dict[str, float]:
    """Hit@k and NDCG@k with one relevant target, as upstream evaluate.py defines them."""
    ranks = [position(row, k) for row in rows.values()]
    return {
        f"hit@{k}": sum(r is not None for r in ranks) / len(ranks),
        f"ndcg@{k}": sum(1 / math.log2(r + 1) for r in ranks if r is not None) / len(ranks),
        "users": len(ranks),
    }


def matrix(dumps, k=10, metric="ndcg") -> dict:
    """R[i][j]: quality on block j's test users after training through block i."""
    states = sorted({i for i, _ in dumps})
    blocks = sorted({j for _, j in dumps})
    key = f"{metric}@{k}"
    values = [
        [metrics(dumps[i, j], k)[key] if (i, j) in dumps else None for j in blocks] for i in states
    ]
    return {"metric": key, "states": states, "blocks": blocks, "values": values}


def backward_transfer(table) -> float | None:
    """Mean of R[last][j] - R[j][j] over earlier blocks; negative means quality fell."""
    cell = {
        (i, j): value
        for i, row in zip(table["states"], table["values"])
        for j, value in zip(table["blocks"], row)
        if value is not None
    }
    last = table["states"][-1]
    diffs = [
        cell[last, j] - cell[j, j]
        for j in table["blocks"]
        if j < last and (last, j) in cell and (j, j) in cell
    ]
    return sum(diffs) / len(diffs) if diffs else None


def flips(before, after, k=10, groups=None) -> dict:
    """Count how each shared user's top-k outcome changed between two model states.

    kept/lost/gained/never partition users by hit before and after. Among kept users,
    the target moved up, down or stayed within the top k. The exact McNemar test
    asks whether lost and gained differ more than a fair coin would.
    """
    if before.keys() != after.keys():
        raise ValueError("Flip counts need the same test users in both states")
    if any(before[u]["target"] != after[u]["target"] for u in before):
        raise ValueError("A user's target changed between states; the test set is not fixed")
    labels = {"all": list(before)}
    if groups is not None:
        for user in before:
            labels.setdefault(str(groups.get(user, "unlabelled")), []).append(user)
    out = {}
    for label, users in labels.items():
        counts = Counter()
        for user in users:
            a, b = position(before[user], k), position(after[user], k)
            if a and b:
                counts["kept"] += 1
                counts["moved_up" if b < a else "moved_down" if b > a else "same_position"] += 1
            else:
                counts["lost" if a else "gained" if b else "never"] += 1
        lost, gained, kept = counts["lost"], counts["gained"], counts["kept"]
        out[label] = {
            "users": len(users),
            **{key: counts[key] for key in ("kept", "lost", "gained", "never")},
            **{key: counts[key] for key in ("moved_up", "moved_down", "same_position")},
            "net": gained - lost,
            "lost_hit_rate": lost / (kept + lost) if kept + lost else None,
            "mcnemar_p": binomtest(lost, lost + gained).pvalue if lost + gained else None,
        }
    return out


def forgetting_events(dumps, block, k=10) -> dict:
    """Per-user hit -> miss transitions across consecutive states on one test block."""
    states = sorted(i for i, j in dumps if j == block)
    users = dumps[states[0], block].keys()
    if any(dumps[i, block].keys() != users for i in states):
        raise ValueError(f"Block {block} test users differ across states")
    events = Counter()
    for user in users:
        hits = [position(dumps[i, block][user], k) is not None for i in states]
        events[sum(a and not b for a, b in pairwise(hits))] += 1
    return {"states": states, "users_by_event_count": dict(sorted(events.items()))}


def report(directory, k=10, groups=None) -> dict:
    """Matrix, backward transfer, consecutive-state flips and forgetting events.

    groups optionally maps a test block (as a string) to {user: label} for slices.
    """
    dumps = load(directory)
    groups = groups or {}
    table = matrix(dumps, k)
    consecutive = {}
    for block in sorted({j for _, j in dumps}):
        states = sorted(i for i, j in dumps if j == block)
        for i, nxt in pairwise(states):
            counts = flips(dumps[i, block], dumps[nxt, block], k, groups.get(str(block)))
            consecutive[f"block{block}:state{i}->state{nxt}"] = counts
    return {
        "source": str(Path(directory).resolve()),
        "k": k,
        "matrix": table,
        "hit_matrix": matrix(dumps, k, metric="hit"),
        "backward_transfer": backward_transfer(table),
        "consecutive_flips": consecutive,
        "forgetting_events": {
            f"block{block}": forgetting_events(dumps, block, k)
            for block in sorted({j for _, j in dumps})
            if sum(j == block for _, j in dumps) > 1
        },
        "scope": "ranking changes on fixed test users; not proof of erased weights or memories",
    }
