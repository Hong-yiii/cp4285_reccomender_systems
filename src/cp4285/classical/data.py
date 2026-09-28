"""Load two Amazon'23 ID files into integer arrays with a shared user vocabulary.

Interactions are implicit feedback (every rating counts). The split is the official Amazon'23
leave-last-out rule: per user, sorted by time, the last event is test, the second-to-last is
validation and the rest is training. This differs from the pilot's cutoff protocol (see
docs/CLASSICAL.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import scipy.sparse as sp


@dataclass
class Domain:
    """Interactions of one category, encoded against a shared user vocabulary."""

    name: str
    user: np.ndarray  # int64 index into the shared user vocabulary
    item: np.ndarray  # int64 index into this domain's own item vocabulary
    ts: np.ndarray  # int64 Unix milliseconds
    n_items: int


def _cache_path(a: Path, b: Path, cache: Path) -> Path:
    key = hashlib.sha256(f"{a.resolve()}|{b.resolve()}".encode()).hexdigest()[:12]
    return cache / f"classical-{a.name.split('.')[0]}-{b.name.split('.')[0]}-{key}.npz"


def load_domains(a, b, cache) -> tuple[Domain, Domain, int]:
    """Return (A, B, n_users). Original user IDs map to one vocabulary, so overlap is real."""
    a, b, cache = Path(a), Path(b), Path(cache)
    cached = _cache_path(a, b, cache)
    if cached.exists():
        z = np.load(cached)
        return (
            Domain(a.name, z["au"], z["ai"], z["at"], int(z["an"])),
            Domain(b.name, z["bu"], z["bi"], z["bt"], int(z["bn"])),
            int(z["n_users"]),
        )
    for path in (a, b):
        if not path.is_file():
            raise FileNotFoundError(f"Missing ID file: {path}; run `cp4285 download` first")

    con = duckdb.connect()
    for name, path in (("a", a), ("b", b)):
        con.read_csv(str(path.resolve()), header=True, all_varchar=True).create_view(f"raw_{name}")
        con.execute(f"""CREATE TABLE {name} AS SELECT user_id, parent_asin,
            CAST("timestamp" AS BIGINT) AS ts FROM raw_{name}""")
        if con.execute(f"SELECT min(ts) < 100000000000 FROM {name}").fetchone()[0]:
            raise ValueError(f"{path} timestamps look like seconds; expected Unix milliseconds")
    con.execute("""CREATE TABLE users AS SELECT user_id, row_number() OVER (ORDER BY user_id) - 1
        AS uid FROM (SELECT user_id FROM a UNION SELECT user_id FROM b)""")
    doms = []
    for name, path in (("a", a), ("b", b)):
        con.execute(f"""CREATE TABLE items_{name} AS SELECT parent_asin,
            row_number() OVER (ORDER BY parent_asin) - 1 AS iid
            FROM (SELECT DISTINCT parent_asin FROM {name})""")
        cols = con.execute(f"""SELECT u.uid, i.iid, t.ts FROM {name} t
            JOIN users u USING (user_id) JOIN items_{name} i USING (parent_asin)""").fetchnumpy()
        n_items = con.execute(f"SELECT count(*) FROM items_{name}").fetchone()[0]
        doms.append(
            Domain(
                path.name,
                cols["uid"].astype(np.int64),
                cols["iid"].astype(np.int64),
                cols["ts"].astype(np.int64),
                int(n_items),
            )
        )
    n_users = con.execute("SELECT count(*) FROM users").fetchone()[0]
    da, db = doms
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(
        cached,
        au=da.user, ai=da.item, at=da.ts, an=da.n_items,
        bu=db.user, bi=db.item, bt=db.ts, bn=db.n_items,
        n_users=n_users,
    )  # fmt: skip
    return da, db, int(n_users)


@dataclass
class Split:
    """Leave-last-out split of domain A.

    valid and test hold one row per target, sorted by user, so a user's rows are adjacent.
    """

    train_user: np.ndarray  # sorted by (user, time)
    train_item: np.ndarray
    train_ts: np.ndarray
    valid: np.ndarray  # (n, 2) user, item
    test: np.ndarray  # (n, 2) user, item
    cutoff: np.ndarray  # (n, 2) user, timestamp of the first validation event (end of training)
    valid_events: np.ndarray  # (n, 3) user, item, timestamp of every validation event


def leave_last_out(d: Domain, n_targets: int = 1) -> Split:
    """Per user, the last n_targets events are test and the n_targets before are validation.

    n_targets = 1 is the official Amazon'23 rule. Users with fewer events lose training
    events first, then validation targets. A user's repeated item counts as one target.
    """
    m = n_targets
    if m < 1:
        raise ValueError("n_targets must be at least 1")
    order = np.lexsort((d.ts, d.user))
    u, i, t = d.user[order], d.item[order], d.ts[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    lens = np.diff(np.r_[starts, len(u)])
    from_end = np.repeat(starts + lens, lens) - 1 - np.arange(len(u))  # 0 = last event
    test = from_end < m
    valid = (from_end >= m) & (from_end < 2 * m)
    train = ~(test | valid)
    vu, vt = u[valid], t[valid]
    first = np.r_[True, vu[1:] != vu[:-1]]
    return Split(
        u[train],
        i[train],
        t[train],
        _targets(u[valid], i[valid]),
        _targets(u[test], i[test]),
        np.c_[vu[first], vt[first]],
        np.c_[vu, i[valid], vt],
    )


def _targets(user: np.ndarray, item: np.ndarray) -> np.ndarray:
    return np.unique(np.c_[user, item], axis=0)  # sorted by user, duplicates dropped


def time_split(d: Domain, cutoff_ms: int, window_ms: int, max_targets: int = 10) -> Split:
    """Next-items split at a global time cutoff T with a window W.

    Validation trains on events before T - W and targets products in [T - W, T); the test
    protocol (with_validation) trains on events before T and targets products in [T, T + W).
    Targets are each user's first max_targets distinct products in the window that the user
    had not reviewed before it. Only users with an earlier event are scored, however few
    events they have. Events from T + W on are never used.
    """
    order = np.lexsort((d.ts, d.user))
    u, i, t = d.user[order], d.item[order], d.ts[order]
    start = cutoff_ms - window_ms
    train = t < start
    valid = (t >= start) & (t < cutoff_ms)
    test = (t >= cutoff_ms) & (t < cutoff_ms + window_ms)
    users = np.unique(u[train])
    return Split(
        u[train],
        i[train],
        t[train],
        _window_targets(u, i, train, valid, d.n_items, max_targets),
        _window_targets(u, i, t < cutoff_ms, test, d.n_items, max_targets),
        np.c_[users, np.full(len(users), start)],
        np.c_[u[valid], i[valid], t[valid]],
    )


def _window_targets(u, i, history, window, n_items: int, cap: int) -> np.ndarray:
    """First `cap` new distinct (user, item) pairs per user in the window, time-sorted input."""
    seen = u[history].astype(np.int64) * n_items + i[history]
    wu, wi = u[window], i[window]
    key = wu.astype(np.int64) * n_items + wi
    keep = ~np.isin(key, seen) & np.isin(wu, u[history])
    _, first = np.unique(key[keep], return_index=True)
    first = np.sort(first)  # back to (user, time) order
    wu, wi = wu[keep][first], wi[keep][first]
    if not len(wu):
        return np.empty((0, 2), np.int64)
    starts = np.flatnonzero(np.r_[True, wu[1:] != wu[:-1]])
    within = np.arange(len(wu)) - np.repeat(starts, np.diff(np.r_[starts, len(wu)]))
    return np.c_[wu, wi][within < cap]


def before(d: Domain, t_ms: int) -> Domain:
    """Only the events before t_ms, so nothing after a cutoff is trained on."""
    m = d.ts < t_ms
    return Domain(d.name, d.user[m], d.item[m], d.ts[m], d.n_items)


def make_split(a: Domain, args) -> tuple[Split, int | None, int | None]:
    """The split named by args.protocol, plus the training end (ms) for validation and test.

    "last": leave-last-out with args.targets events per user (no global training end).
    "time": time_split at args.cutoff (YYYY-MM-DD, UTC) with args.window_days.
    """
    if args.protocol == "last":
        return leave_last_out(a, args.targets), None, None
    cutoff = int(datetime.fromisoformat(args.cutoff).replace(tzinfo=UTC).timestamp() * 1000)
    window = int(args.window_days * 86_400_000)
    return time_split(a, cutoff, window, args.max_targets), cutoff - window, cutoff


def with_validation(split: Split) -> Split:
    """Training data for the test protocol: train + the validation events.

    Test targets are then one step ahead of the history, as in the original SASRec
    evaluation. Events are appended unsorted; build_stream re-sorts by (user, time).
    """
    return Split(
        np.r_[split.train_user, split.valid_events[:, 0]],
        np.r_[split.train_item, split.valid_events[:, 1]],
        np.r_[split.train_ts, split.valid_events[:, 2]],
        split.valid,
        split.test,
        split.cutoff,
        split.valid_events,
    )


def subsample_users(d: Domain, n_users: int, seed: int = 0) -> Domain:
    """Keep a random subset of A's users, for fast iteration."""
    rng = np.random.default_rng(seed)
    keep = rng.choice(np.unique(d.user), size=n_users, replace=False)
    m = np.isin(d.user, keep)
    return Domain(d.name, d.user[m], d.item[m], d.ts[m], d.n_items)


def to_csr(user: np.ndarray, item: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    """Binary user x item matrix (duplicates collapse to 1)."""
    x = sp.csr_matrix((np.ones(len(user), np.float32), (user, item)), shape=shape)
    x.data[:] = 1.0
    return x
