"""Bounded local Amazon review pilot, not a downloader or a finalized study protocol."""

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

from .data import DataSource, Evaluation, Event, Example, PreparedExperiment, interleave


class AmazonSource(DataSource):
    """Read official ID CSV or review JSONL (optionally gzip), preserving source IDs.

    Proposed offline ordinal protocol: each user's first event is context, next
    initial_targets events are initial A targets, next update_targets are fresh
    updates, and the immediately following event is held out. Hash-disjoint users
    provide development versus test targets, avoiding a validation-event test prefix.
    Domain B has no initial gradient updates. Catalogues use pre-holdout events only
    (including incoming training inventory); this is not online vocabulary discovery.
    """

    def __init__(
        self,
        paths: Mapping[str, str | Path],
        *,
        max_rows: int = 50000,
        max_users: int = 100,
        initial_targets: int = 2,
        update_targets: int = 2,
        block_size: int = 4,
        blocks: int = 2,
    ) -> None:
        if set(paths) != {"A", "B"}:
            raise ValueError("Supply explicitly labelled A and B local files")
        knobs = (max_rows, max_users, initial_targets, update_targets, block_size, blocks)
        if any(type(n) is not int or n <= 0 for n in knobs):
            raise ValueError("All bounds must be positive integers")
        self.paths = {domain: Path(path) for domain, path in paths.items()}
        self.max_rows, self.max_users = max_rows, max_users
        self.initial_targets, self.update_targets = initial_targets, update_targets
        self.block_size, self.blocks = block_size, blocks
        self.audit: dict = {}
        self.vocabularies: dict[str, tuple[str, ...]] = {}

    def _read(self, domain: str) -> tuple[Event, ...]:
        path = self.paths[domain]
        opener = gzip.open if path.suffix == ".gz" else open
        csv_format = path.name.removesuffix(".gz").endswith(".csv")
        digest = hashlib.sha256()
        events = []
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            rows = csv.DictReader(stream) if csv_format else (json.loads(line) for line in stream)
            for index, row in enumerate(islice(rows, self.max_rows), start=1):
                digest.update(json.dumps(row, sort_keys=True).encode() + b"\n")
                try:
                    user, item = row["user_id"], row["parent_asin"]
                    raw_timestamp = row["timestamp"]
                    timestamp = int(raw_timestamp)
                    if isinstance(raw_timestamp, bool) or str(timestamp) != str(raw_timestamp):
                        raise ValueError("timestamp must be an integer")
                    if (
                        not isinstance(user, str)
                        or not user
                        or not isinstance(item, str)
                        or not item
                    ):
                        raise ValueError("empty/non-string identity")
                    if timestamp < 0:
                        raise ValueError("negative timestamp")
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid Amazon schema at {path.name} row {index}") from error
                events.append(Event(domain, user, f"row:{index}", item, timestamp))
        self.audit[domain] = {
            "file": path.name,
            "file_bytes": path.stat().st_size,
            "rows_scanned": len(events),
            "row_limit": self.max_rows,
            "prefix_sha256": digest.hexdigest(),
            "timestamp_min": min((e.timestamp for e in events), default=None),
            "timestamp_max": max((e.timestamp for e in events), default=None),
            "users_scanned": len({e.user for e in events}),
            "items_scanned": len({e.item for e in events}),
        }
        return tuple(events)

    def load_data(self) -> PreparedExperiment:
        self.audit = {}
        initial, streams, development, tests = [], {}, {}, {}
        selected_users = {}
        prefix_size = 1 + self.initial_targets + self.update_targets
        for domain in ("A", "B"):
            users = defaultdict(list)
            for event in self._read(domain):
                users[event.user].append(event)
            cleaned = {}
            ties = 0
            for user, events in users.items():
                ordered = sorted(events, key=lambda e: (e.timestamp, int(e.event_id[4:])))
                unique = []
                for event in ordered:
                    if unique and unique[-1].timestamp == event.timestamp:
                        ties += 1
                    else:
                        unique.append(event)
                if len(unique) >= prefix_size:
                    cleaned[user] = unique
            selected = sorted(cleaned)[: self.max_users]
            selected_users[domain] = set(selected)
            catalogue = tuple(sorted({e.item for u in selected for e in cleaned[u][:prefix_size]}))
            self.vocabularies[domain] = catalogue
            updates, dev, test = [], [], []
            exclusions = Counter()
            for user in selected:
                events = cleaned[user]
                for index in range(1, prefix_size):
                    example = Example(tuple(events[:index]), events[index])
                    if index <= self.initial_targets:
                        if domain == "A":
                            initial.append(example)
                    else:
                        updates.append(example)
                if len(events) <= prefix_size:
                    exclusions["no_future_target"] += 1
                elif events[prefix_size].item not in catalogue:
                    exclusions["unknown_heldout_item"] += 1
                else:
                    example = Example(tuple(events[:prefix_size]), events[prefix_size])
                    bucket = int(hashlib.sha256(user.encode()).hexdigest(), 16) % 2
                    (dev if bucket == 0 else test).append(example)
            streams[domain] = tuple(
                sorted(updates, key=lambda e: (e.target.timestamp, e.target.key))
            )
            development[domain] = Evaluation(f"development-{domain}", domain, tuple(dev), catalogue)
            tests[domain] = Evaluation(
                "retention-A" if domain == "A" else "adaptation-B", domain, tuple(test), catalogue
            )
            self.audit[domain].update(
                timestamp_ties_dropped=ties,
                eligible_users=len(cleaned),
                selected_users=len(selected),
                catalogue_items=len(catalogue),
                fresh_update_targets=len(updates),
                development_targets=len(dev),
                test_targets=len(test),
                exclusions=dict(exclusions),
            )
        self.audit["selected_user_overlap"] = len(selected_users["A"] & selected_users["B"])
        self.audit["protocol"] = {
            "initial_targets_per_user": self.initial_targets,
            "update_targets_per_user": self.update_targets,
            "timestamp_units": "source units preserved; official Amazon 2023 uses milliseconds",
            "filter": "all ratings; next review is not a positive-preference label",
            "sampling": "bounded file prefix then lexical eligible users; not representative",
            "cohort": "offline minimum pre-holdout activity; not strict calendar-time simulation",
            "ties": "retain first source row per user/timestamp; repeated items otherwise retained",
        }
        data = PreparedExperiment(
            provenance="LOCAL AMAZON REVIEW PILOT; proposed ordinal protocol; "
            + json.dumps(self.audit, sort_keys=True),
            initial=tuple(sorted(initial, key=lambda e: (e.target.timestamp, e.target.key))),
            validation=development["A"],
            tests=(tests["A"], development["B"], tests["B"]),
            conditions=tuple(
                interleave(
                    streams["A"],
                    streams["B"],
                    b_shares=(share,) * self.blocks,
                    block_size=self.block_size,
                    name=name,
                )
                for name, share in (("A-only", 0.0), ("mixed", 0.5), ("B-only", 1.0))
            ),
        )
        data.validate()
        return data
