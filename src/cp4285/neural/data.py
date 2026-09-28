"""Bounded familiar-item A-only pilot split for the neural cutoff protocol."""

from __future__ import annotations

from pathlib import Path

from ..common.data import connection, load
from ..common.utils import millis, sha256


def prepare(path, cfg):
    cutoff, end = millis(cfg["initial_cutoff"]), millis(cfg["update_end"])
    if end <= cutoff or cfg["max_users"] < 1 or cfg["max_length"] < 1:
        raise ValueError("Require update_end > initial_cutoff and positive size limits")
    with connection() as con:
        load(con, path, "a")
        con.execute(
            "CREATE TABLE filtered AS SELECT DISTINCT * FROM a WHERE rating >= ? AND timestamp < ?",
            [cfg["min_rating"], end],
        )
        tied = con.execute("""SELECT coalesce(sum(n),0) FROM (SELECT count(*) n FROM filtered
                            GROUP BY user_id,timestamp HAVING count(*)>1)""").fetchall()[0][0]
        con.execute("""CREATE TABLE ordered AS SELECT * FROM filtered
                       QUALIFY count(*) OVER (PARTITION BY user_id,timestamp)=1""")
        # Cohort eligibility depends on the initial period only, never future activity.
        rows = con.execute(
            """WITH eligible AS (
            SELECT user_id FROM ordered WHERE timestamp < ? GROUP BY user_id HAVING count(*) >= 5
            ORDER BY md5(user_id || ?) LIMIT ?)
            SELECT user_id,parent_asin,timestamp FROM ordered JOIN eligible USING(user_id)
            ORDER BY user_id,timestamp""",
            [cutoff, str(cfg["seed"]), cfg["max_users"]],
        ).fetchall()
    histories = {}
    for user, item, stamp in rows:
        histories.setdefault(user, []).append((item, stamp))
    if not histories:
        raise ValueError(
            "No eligible histories before cutoff; inspect audit or change the pilot window"
        )
    initial = {u: [(i, t) for i, t in h if t < cutoff] for u, h in histories.items()}
    # Last two pre-cutoff events are held out globally from every training prefix and target.
    items = sorted({i for h in initial.values() for i, _ in h[:-2]})
    item_map = {item: idx + 1 for idx, item in enumerate(items)}
    if len(items) < 2:
        raise ValueError("Need at least two training items")
    sets = {k: [] for k in ("train", "validation", "retention", "continuation")}
    skipped = {"validation_oov": 0, "retention_oov": 0, "continuation_oov": 0}

    def example(user, prefix, target, stamp):
        return {
            "user": user,
            "history": prefix[-cfg["max_length"] :],
            "target": target,
            "timestamp": stamp,
        }

    for user, h in histories.items():
        before = initial[user]
        prefix = []
        for item, stamp in before[:-2]:
            target = item_map[item]
            if prefix:
                sets["train"].append(example(user, prefix, target, stamp))
            prefix.append(target)
        for split, (item, stamp) in zip(("validation", "retention"), before[-2:]):
            if item in item_map:
                sets[split].append(example(user, prefix, item_map[item], stamp))
            else:
                skipped[f"{split}_oov"] += 1
        for item, stamp in h:
            if stamp < cutoff:
                continue
            if item not in item_map:
                skipped["continuation_oov"] += 1
                continue
            target = item_map[item]
            sets["continuation"].append(example(user, prefix, target, stamp))
            prefix.append(target)
    for examples in sets.values():
        examples.sort(key=lambda x: (x["timestamp"], x["user"]))
    for split, examples in sets.items():
        if not examples:
            raise ValueError(f"Empty {split} partition; inspect the data/window before training")
    return {
        "metadata": {
            "domain": "Electronics",
            "source": str(Path(path).resolve()),
            "source_sha256": sha256(path),
            "config": cfg,
            "users": len(histories),
            "items": len(items),
            "dropped_tied_rows": tied,
            "skipped": skipped,
            "counts": {k: len(v) for k, v in sets.items()},
            "scope": "bounded familiar-item A-only pilot; unknown later items omitted",
            "test_rule": "last two pre-cutoff events excluded from all training histories",
            "global_5core_warning": "upstream 5-core filtering uses full-history activity",
        },
        "item_ids": items,
        **sets,
    }
