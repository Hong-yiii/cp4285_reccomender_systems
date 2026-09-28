import csv
import gzip

import numpy as np
import pytest

from cp4285.classical.contaminate import build_stream
from cp4285.classical.data import Domain, leave_last_out, load_domains, to_csr
from cp4285.classical.evaluate import evaluate
from cp4285.classical.models import (
    QRSVD,
    SLIST,
    IncrementalQRSVD,
    Markov,
    MarkovQRSVD,
    last_items,
    recency_query,
    transitions,
)

DAY = 86_400_000


def synthetic(n_users, n_items, per_user, seed, user_offset=0):
    """Timelines that walk a ring of items, so next-item structure is learnable."""
    rng = np.random.default_rng(seed)
    users, items, ts = [], [], []
    for u in range(n_users):
        start = rng.integers(n_items)
        for step in range(per_user):
            users.append(u + user_offset)
            items.append((start + step) % n_items)
            ts.append(1_600_000_000_000 + (u * per_user + step) * DAY)
    return Domain("s", np.array(users), np.array(items), np.array(ts), n_items)


@pytest.fixture
def ab():
    a = synthetic(60, 30, 8, seed=1)
    b = synthetic(80, 25, 8, seed=2, user_offset=40)  # users 40-59 appear in both
    return a, b, 120


def test_load_domains_shares_original_user_ids(tmp_path):
    def write(path, rows):
        with gzip.open(path, "wt") as f:
            w = csv.writer(f)
            w.writerow(["user_id", "parent_asin", "rating", "timestamp"])
            w.writerows(rows)
        return path

    t = 1_600_000_000_000
    a = write(tmp_path / "a.csv.gz", [("u1", "x", 5, t), ("u2", "y", 4, t + 1)])
    b = write(tmp_path / "b.csv.gz", [("u2", "m", 5, t), ("u3", "n", 5, t)])
    da, db, n_users = load_domains(a, b, tmp_path / "cache")
    assert n_users == 3
    assert len(np.intersect1d(da.user, db.user)) == 1
    again = load_domains(a, b, tmp_path / "cache")  # cached arrays match
    assert np.array_equal(again[0].user, da.user) and again[2] == 3


def test_seconds_are_rejected(tmp_path):
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("user_id,parent_asin,rating,timestamp\nu,i,5,1577836800\n")
    with pytest.raises(ValueError, match="milliseconds"):
        load_domains(path, path, tmp_path / "cache")


def test_leave_last_out_holds_out_the_two_latest_events(ab):
    a, _, _ = ab
    split = leave_last_out(a)
    assert len(split.valid) == len(split.test) == 60
    assert len(split.train_user) == 60 * 6
    for u, item in split.test[:5]:
        timeline = np.sort(a.ts[a.user == u])
        assert a.item[(a.user == u) & (a.ts == timeline[-1])][0] == item
    cutoff = dict(split.cutoff)
    assert all(t < cutoff[u] for u, t in zip(split.train_user, split.train_ts))


def test_disjoint_injection_volume_and_fresh_identities(ab):
    a, b, n = ab
    split = leave_last_out(a)
    s = build_stream(split, a.n_items, b, n, 0.5, "disjoint", seed=0)
    assert s.info["injected"] >= 0.5 * len(split.train_user)
    injected = s.item >= a.n_items
    assert s.user[injected].min() >= n  # never merged into real A users


def test_shared_injection_only_uses_the_past_of_overlapping_users(ab):
    a, b, n = ab
    split = leave_last_out(a)
    s = build_stream(split, a.n_items, b, n, 10.0, "shared", seed=0)
    injected = s.item >= a.n_items
    assert s.info["capped"]
    assert set(s.user[injected]) <= set(range(40, 60))
    cutoff = dict(split.cutoff)
    assert all(t < cutoff[u] for u, t in zip(s.user[injected], s.ts[injected]))


def test_local_models_ignore_disjoint_b_but_fixed_rank_models_do_not(ab):
    a, b, n = ab
    split = leave_last_out(a)
    users = np.unique(split.train_user)
    scores = {}
    for level in (0.0, 1.0):
        s = build_stream(split, a.n_items, b, n, level, "disjoint", seed=0)
        x = to_csr(s.user, s.item, (s.n_users, s.n_items))
        t = transitions(s.user, s.item, s.n_items)
        last = last_items(s.user, s.item, s.n_users, s.n_items)
        q = recency_query(s.user, s.item, s.n_users, s.n_items, 2.0)
        pop = np.asarray(x.sum(axis=0)).ravel()
        slist = SLIST(max_items=10_000).fit(s.user, s.item, s.ts, s.n_users, s.n_items)
        rank = MarkovQRSVD(k=4, beta=0).fit(t, pop)
        scores[level] = {
            "markov": Markov().fit(t).score(last[users])[:, : a.n_items],
            "slist": slist.score(q[users])[:, : a.n_items],
            "share": rank.base_share(a.n_items).mean(),
        }
    assert np.allclose(scores[0.0]["markov"], scores[1.0]["markov"])
    assert np.allclose(scores[0.0]["slist"], scores[1.0]["slist"], atol=1e-4)
    assert scores[0.0]["share"] == pytest.approx(1.0)
    assert scores[1.0]["share"] < 0.9  # B takes some of the k components


def test_qrsvd_recovers_an_exact_low_rank_subspace_and_streams_to_it():
    rng = np.random.default_rng(0)
    x = rng.random((400, 5)) @ rng.random((5, 60))
    batch = QRSVD(k=5, seed=0).fit(_csr(x))
    assert np.allclose(batch.V.T @ batch.V, np.eye(5), atol=1e-4)
    stream = IncrementalQRSVD(k=5, seed=0)
    for chunk in np.array_split(np.arange(400), 8):
        stream.partial_fit(_csr(x[chunk]))
    overlap = np.linalg.svd(batch.V.T @ stream.V, compute_uv=False)
    assert overlap.min() > 0.99  # same 5-dimensional item subspace


def test_evaluate_uses_known_ranks_and_excludes_seen_items():
    class Fixed:
        def score(self, rows):
            return np.tile(np.array([4.0, 3.0, 2.0, 1.0], np.float32), (rows.shape[0], 1))

    seen = to_csr(np.array([0]), np.array([0]), (2, 4))
    pairs = np.array([[0, 1], [1, 3]])  # user 0: item 0 excluded, so item 1 ranks first
    result = evaluate(Fixed(), seen, seen, pairs, 4, ks=(1, 10))
    assert result["recall@1"] == 0.5
    assert result["recall@10"] == 1.0
    assert result["mrr"] == pytest.approx((1 + 1 / 4) / 2)


def _csr(x):
    import scipy.sparse as sp

    return sp.csr_matrix(x.astype(np.float32))
