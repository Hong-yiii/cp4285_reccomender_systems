"""Classical contamination experiments: A = Electronics, B = Movies & TV by default."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..data import save_json
from .contaminate import build_stream
from .data import leave_last_out, load_domains, subsample_users, to_csr
from .evaluate import evaluate, sample_pairs
from .models import (
    QRSVD,
    SLIST,
    IncrementalQRSVD,
    Markov,
    MarkovQRSVD,
    MostPop,
    last_items,
    recency_query,
    transitions,
)


def _load(args):
    return load_domains(args.a, args.b, args.cache)


def _setup(args):
    t0 = time.time()
    a, b, n_users = _load(args)
    if args.users:
        a = subsample_users(a, args.users, args.seed)
    split = leave_last_out(a)
    held = split.valid if args.split == "valid" else split.test
    pairs = sample_pairs(held, args.eval_users, args.seed)
    print(
        f"loaded in {time.time() - t0:.0f}s: {len(split.train_user):,} A training events, "
        f"{a.n_items:,} A items, {b.n_items:,} B items, {len(pairs):,} evaluation users",
        flush=True,
    )
    return a, b, n_users, split, pairs


def _report(rows, tag, level, info, metrics, **extra):
    rows.append({"model": tag, "level": level, **metrics, **extra, **info})
    tail = "".join(f"  {k}={v:.2f}" for k, v in extra.items())
    print(
        f"{tag:<18} {level:>5.0%}  ndcg@10={metrics['ndcg@10']:.4f}  "
        f"recall@10={metrics['recall@10']:.4f}  mrr={metrics['mrr']:.4f}{tail}",
        flush=True,
    )


def _save(rows, name, args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_json(path, {"scope": "amazon", "args": settings, "rows": rows})
    print(f"saved {path}")


def stats(args):
    a, b, _ = _load(args)
    au, bu = np.unique(a.user), np.unique(b.user)
    shared = np.intersect1d(au, bu)
    shared_events = int(np.isin(b.user, shared).sum())
    print(
        f"A users {len(au):,} | B users {len(bu):,} | shared {len(shared):,} "
        f"({len(shared) / len(au):.1%} of A)"
    )
    print(
        f"A events {len(a.user):,} | B events from shared users {shared_events:,} "
        f"= {shared_events / len(a.user):.1%} of A volume"
    )


def sweep(args):
    a, b, n_users, split, pairs = _setup(args)
    rows = []
    for level in args.levels:
        s = build_stream(split, a.n_items, b, n_users, level, args.design, args.seed)
        info = {k: v for k, v in s.info.items() if k != "design"}
        if s.info["capped"]:
            print(f"  only {s.info['frac_actual']:.0%} of A volume available from B")
        x = to_csr(s.user, s.item, (s.n_users, s.n_items))
        last = last_items(s.user, s.item, s.n_users, s.n_items)
        t = transitions(s.user, s.item, s.n_items, window=args.window)
        pop = np.asarray(x.sum(axis=0)).ravel()
        n_cand = s.n_items if args.full_catalog else a.n_items

        _report(rows, "mostpop", level, info, evaluate(MostPop().fit(x), x, x, pairs, n_cand))
        _report(rows, "markov", level, info, evaluate(Markov().fit(t), last, x, pairs, n_cand))
        t0 = time.time()
        m = MarkovQRSVD(k=args.k, beta=args.beta, seed=args.seed).fit(t, pop)
        _report(
            rows, f"markov-qrsvd k={args.k}", level, info, evaluate(m, last, x, pairs, n_cand),
            base_share=float(m.base_share(a.n_items).mean()), fit_s=time.time() - t0,
        )  # fmt: skip
        if args.slist:
            t0 = time.time()
            m = SLIST(
                alpha=args.slist_alpha,
                session_weight_days=args.slist_session_days,
                max_items=args.slist_items,
            ).fit(s.user, s.item, s.ts, s.n_users, s.n_items)
            q = recency_query(s.user, s.item, s.n_users, s.n_items, args.predict_weight)
            _report(
                rows, f"slist N={args.slist_items // 1000}K", level, info,
                evaluate(m, q, x, pairs, n_cand),
                base_share=m.base_share(a.n_items), fit_s=time.time() - t0,
            )  # fmt: skip
            del m, q
        if args.puresvd:
            m = QRSVD(k=args.k, seed=args.seed).fit(x)
            _report(
                rows, f"puresvd k={args.k}", level, info, evaluate(m, x, x, pairs, n_cand),
                base_share=float(m.base_share(a.n_items).mean()),
            )  # fmt: skip
    _save(rows, f"sweep_{args.design}{'_full' if args.full_catalog else ''}", args)


def stream(args):
    """Incremental PureSVD: A users arrive in time order, then B users (disjoint design)."""
    a, b, n_users, split, pairs = _setup(args)
    s = build_stream(split, a.n_items, b, n_users, max(args.levels), "disjoint", args.seed)
    x_all = to_csr(s.user, s.item, (s.n_users, s.n_items))
    x_a = x_all[:n_users]
    n_cand = s.n_items if args.full_catalog else a.n_items
    model = IncrementalQRSVD(k=args.k, forget=args.forget, seed=args.seed)
    rows = []

    first = np.full(n_users, np.iinfo(np.int64).max)
    np.minimum.at(first, split.train_user, split.train_ts)
    order = np.argsort(first)[: len(np.unique(split.train_user))]
    for chunk in np.array_split(order, args.chunks):
        model.partial_fit(x_a[np.sort(chunk)])
    _report(
        rows, "stream-svd", 0.0, {}, evaluate(model, x_a, x_a, pairs, n_cand),
        base_share=float(model.base_share(a.n_items).mean()),
    )  # fmt: skip

    extra = x_all[n_users:]
    cum = np.cumsum(np.diff(extra.indptr))
    done = 0
    for level in sorted(v for v in args.levels if v > 0):
        upto = min(int(np.searchsorted(cum, level * len(split.train_user)) + 1), extra.shape[0])
        for chunk in np.array_split(np.arange(done, upto), args.chunks):
            if len(chunk):
                model.partial_fit(extra[chunk])
        done = upto
        _report(
            rows, "stream-svd", level, {"injected": int(cum[upto - 1])},
            evaluate(model, x_a, x_a, pairs, n_cand),
            base_share=float(model.base_share(a.n_items).mean()),
        )  # fmt: skip
    _save(rows, f"stream_f{args.forget}", args)


def add_arguments(parser, root: Path, data_cfg: dict):
    """Register `cp4285 classical ...` subcommands on an argparse subparser."""
    commands = parser.add_subparsers(dest="classical_command", required=True)
    groups = {
        "stats": commands.add_parser("stats", help="Measure A/B user overlap"),
        "sweep": commands.add_parser("sweep", help="Refit models at each B contamination level"),
        "stream": commands.add_parser("stream", help="Incremental QR-SVD over the A->B stream"),
    }
    for name, p in groups.items():
        p.add_argument("--a", type=Path, default=Path(data_cfg["a"]), help="A ID file")
        p.add_argument(
            "--b", type=Path, default=Path(data_cfg["b"]),
            help="B ID file, e.g. data/raw/0core/Movies_and_TV.csv.gz for 17.3M events",
        )  # fmt: skip
        p.add_argument("--cache", type=Path, default=root / "data/processed")
        if name == "stats":
            continue
        p.add_argument("--output", type=Path, default=root / "reports/classical")
        p.add_argument("--design", choices=["disjoint", "shared"], default="disjoint")
        p.add_argument("--levels", type=float, nargs="+", default=[0, 0.1, 0.5, 1.0, 2.0])
        p.add_argument("--k", type=int, default=64, help="QR-SVD rank")
        p.add_argument("--beta", type=float, default=0.25, help="popularity down-weighting")
        p.add_argument("--window", type=int, default=1, help="transition window (steps)")
        p.add_argument("--slist", action="store_true", help="also run SLIST (Choi et al. 2021)")
        p.add_argument("--slist-items", type=int, default=20000, help="dense N x N item budget")
        p.add_argument("--slist-alpha", type=float, default=0.5)
        p.add_argument(
            "--slist-session-days", type=float, default=-1,
            help="timeline age decay in days (paper: 256 for Diginetica); <=0 disables",
        )  # fmt: skip
        p.add_argument("--predict-weight", type=float, default=2.0, help="SLIST query decay")
        p.add_argument("--puresvd", action="store_true", help="also run user x item PureSVD")
        p.add_argument("--users", type=int, default=None, help="subsample A users")
        p.add_argument("--eval-users", type=int, default=20000)
        p.add_argument("--split", choices=["valid", "test"], default="valid")
        p.add_argument("--full-catalog", action="store_true", help="rank over A + B items")
        p.add_argument("--seed", type=int, default=4285)
        p.add_argument("--forget", type=float, default=1.0, help="stream: old-data decay")
        p.add_argument("--chunks", type=int, default=10, help="stream: updates per phase")


def run(args):
    {"stats": stats, "sweep": sweep, "stream": stream}[args.classical_command](args)
