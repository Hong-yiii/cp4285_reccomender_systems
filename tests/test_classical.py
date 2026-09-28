import csv
import gzip

import numpy as np
import pytest

from cp4285.classical.contaminate import build_stream
from cp4285.classical.data import Domain, leave_last_out, load_domains, to_csr, with_validation
from cp4285.classical.evaluate import evaluate, per_user, sample_pairs
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


def test_several_targets_per_user_and_test_history_includes_validation(ab):
    a, _, _ = ab
    split = leave_last_out(a, n_targets=2)
    assert len(split.valid) == len(split.test) == 60 * 2
    assert len(split.train_user) == 60 * 4
    u = 7
    timeline = a.item[a.user == u][np.argsort(a.ts[a.user == u])]
    assert set(split.test[split.test[:, 0] == u, 1]) == set(timeline[-2:])
    assert set(split.valid[split.valid[:, 0] == u, 1]) == set(timeline[-4:-2])
    history = with_validation(split)
    assert len(history.train_user) == 60 * 6  # test targets now follow the validation events
    assert sorted(history.train_item[history.train_user == u]) == sorted(timeline[:-2])


def test_sample_pairs_keeps_every_target_of_a_sampled_user():
    pairs = np.array([[0, 5], [0, 6], [1, 7], [2, 8], [2, 9], [3, 1]])
    sample = sample_pairs(pairs, 2, seed=0)
    users = np.unique(sample[:, 0])
    assert len(users) == 2
    assert len(sample) == np.isin(pairs[:, 0], users).sum()
    assert np.all(np.diff(np.flatnonzero(np.r_[True, sample[1:, 0] != sample[:-1, 0]])) > 0)


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
    assert result["hit@1"] == 0.5
    assert result["hit@10"] == 1.0
    assert result["mrr"] == pytest.approx((1 + 1 / 4) / 2)


def test_hit_and_recall_differ_with_several_targets():
    # user 0: targets at ranks 0 and 15; user 1: one target at rank 2; user 2: ranks 30, 40
    ranks = np.array([0, 15, 2, 30, 40])
    user = np.array([0, 0, 1, 2, 2])
    s = per_user(ranks, user, k=10)
    assert s["hit"].tolist() == [1, 1, 0]
    assert s["recall"].tolist() == [0.5, 1, 0]
    ideal_two = 1 + 1 / np.log2(3)
    assert s["ndcg"] == pytest.approx([1 / ideal_two, 1 / np.log2(4), 0])
    assert s["rr"] == pytest.approx([1, 1 / 3, 1 / 31])
    assert per_user(np.array([0, 1, 5]), np.zeros(3, int), k=2)["recall"].tolist() == [1.0]
    with pytest.raises(ValueError, match="adjacent"):
        per_user(ranks, np.array([0, 1, 0, 2, 2]))


def test_evaluate_ranks_each_of_a_users_targets():
    class Fixed:
        def score(self, rows):
            return np.tile(np.arange(20, 0, -1, dtype=np.float32), (rows.shape[0], 1))

    seen = to_csr(np.array([0]), np.array([0]), (1, 20))
    pairs = np.array([[0, 1], [0, 12]])  # ranks 0 and 10 once item 0 is excluded
    result = evaluate(Fixed(), seen, seen, pairs, 20, ks=(10,))
    assert result["hit@10"] == 1.0
    assert result["recall@10"] == 0.5


def _csr(x):
    import scipy.sparse as sp

    return sp.csr_matrix(x.astype(np.float32))


def test_paired_change_detects_a_uniform_drop_and_no_change():
    from cp4285.classical.evaluate import paired_change

    rng = np.random.default_rng(0)
    base = rng.random(2000)
    same = paired_change(base, base.copy(), n_boot=200)
    assert same[0] == pytest.approx(0) and same[1] == pytest.approx(0) and same[3] == 1.0
    drop = paired_change(base, base * 0.8, n_boot=200)
    assert drop[0] == pytest.approx(-0.2) and drop[2] < 0 and drop[3] == 0.0


def test_time_split_targets_new_products_in_each_window():
    from cp4285.classical.data import time_split

    day = lambda n: 1_600_000_000_000 + n * DAY
    events = [  # (user, item, day)
        (0, 0, 1), (0, 1, 5), (0, 2, 12), (0, 2, 13), (0, 1, 14), (0, 3, 22), (0, 4, 25),
        (0, 5, 35),
        (1, 6, 15), (1, 7, 21),
        (2, 8, 2),
        (3, 9, 3), (3, 10, 21), (3, 11, 22), (3, 12, 23),
    ]  # fmt: skip
    u, i, d = (np.array(c) for c in zip(*events))
    split = time_split(Domain("a", u, i, day(d), 13), day(20), 10 * DAY, max_targets=2)
    targets = lambda rows: {(int(a), int(b)) for a, b in rows}
    # validation: history before day 10, targets in [10, 20); repeats and seen items dropped
    assert targets(split.valid) == {(0, 2)}
    # test: history before day 20, targets in [20, 30), first two per user; day 35 unused
    assert targets(split.test) == {(0, 3), (0, 4), (1, 7), (3, 10), (3, 11)}
    history = with_validation(split)
    assert sorted(history.train_item[history.train_user == 0]) == [0, 1, 1, 2, 2]
    assert history.train_ts.max() < day(20)


@pytest.mark.parametrize("protocol", ["last", "time"])
def test_study_runs_end_to_end(tmp_path, protocol):
    import argparse
    import json

    from cp4285.classical.experiments import add_arguments, run

    def write(path, n_users, n_items, prefix, seed):
        rng = np.random.default_rng(seed)
        with gzip.open(path, "wt") as f:
            w = csv.writer(f)
            w.writerow(["user_id", "parent_asin", "rating", "timestamp"])
            for u in range(n_users):
                start = rng.integers(n_items)
                for step in range(8):  # one review every 100 days from September 2020
                    t = 1_600_000_000_000 + (step * 100 + u % 50) * DAY
                    w.writerow([f"u{u}", f"{prefix}{(start + step) % n_items}", 5, t])
        return path

    a = write(tmp_path / "a.csv.gz", 120, 40, "e", 1)
    b = write(tmp_path / "b.csv.gz", 150, 30, "m", 2)
    parser = argparse.ArgumentParser()
    add_arguments(parser, tmp_path, {"a": str(a), "b": str(b)})
    args = parser.parse_args(
        ["study", "--protocol", protocol, "--cutoff", "2022-01-01", "--window-days", "180",
         "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out"),
         "--ks", "2", "4", "--betas", "0", "--seeds", "0", "--levels", "0", "0.5",
         "--eval-users", "30", "--boot", "20"]
    )  # fmt: skip
    run(args)
    (report,) = (tmp_path / "out").glob("study_*.json")
    summary = json.loads(report.read_text())
    metric = "recall@10" if protocol == "time" else "hit@10"
    assert summary["primary_metric"] == metric
    assert {r["model"] for r in summary["rows"]} >= {"mostpop", "markov", "qr", "qr_scaled"}
    assert all(0 <= r[metric] <= 1 for r in summary["rows"])
    assert len(summary["changes"]) == 4  # one per model at the 50% level
