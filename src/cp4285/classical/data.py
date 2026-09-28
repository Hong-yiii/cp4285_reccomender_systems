"""Load two Amazon'23 ID files into integer arrays with a shared user vocabulary.

Interactions are implicit feedback (every rating counts). The split is the official Amazon'23
leave-last-out rule: per user, sorted by time, the last event is test, the second-to-last is
validation and the rest is training. This differs from the pilot's cutoff protocol (see
docs/CLASSICAL.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
    """Leave-last-out split of domain A."""

    train_user: np.ndarray  # sorted by (user, time)
    train_item: np.ndarray
    train_ts: np.ndarray
    valid: np.ndarray  # (n, 2) user, item
    test: np.ndarray  # (n, 2) user, item
    cutoff: np.ndarray  # (n, 2) user, timestamp of the validation event (end of training)


def leave_last_out(d: Domain) -> Split:
    order = np.lexsort((d.ts, d.user))
    u, i, t = d.user[order], d.item[order], d.ts[order]
    last = np.r_[u[1:] != u[:-1], True]  # last event of each user
    second = np.r_[last[1:], False] & ~last
    train = ~(last | second)
    return Split(
        u[train],
        i[train],
        t[train],
        np.c_[u[second], i[second]],
        np.c_[u[last], i[last]],
        np.c_[u[second], t[second]],
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
