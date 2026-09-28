"""Rigorous contamination study for the classical models.

The primary metric is Hit@10 for the next-item task (--protocol last) and Recall@10 for the
next-items task (--protocol time); NDCG@10 only breaks ties when tuning.

1. Tune the QR model (rank k, popularity weight beta) on the validation split, clean data only.
2. Refit on train + validation and score the test split, for several seeds and B levels.
   The same test users are scored everywhere, so every change is a paired comparison.
3. Report each change in the primary metric against clean with a paired bootstrap 95% CI
   and one-sided p-value.
4. Mechanism control: a QR model whose rank grows with the data (k * (1 + level)). If the
   fixed rank budget causes the drop, this model should degrade much less.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..common.utils import save_json
from .contaminate import build_stream
from .data import before, load_domains, make_split, to_csr, with_validation
from .evaluate import paired_change, per_user, rank_targets, sample_pairs
from .models import Markov, MarkovQRSVD, MostPop, last_items, transitions


def _inputs(s):
    x = to_csr(s.user, s.item, (s.n_users, s.n_items))
    return (
        x,
        last_items(s.user, s.item, s.n_users, s.n_items),
        transitions(s.user, s.item, s.n_items),
    )


def primary_metric(args) -> str:
    return "recall" if args.protocol == "time" else "hit"


def tune(split, a, b, n_users, pairs, ks, betas, seed, metric="hit"):
    s = build_stream(split, a.n_items, b, n_users, 0.0, "disjoint", seed)
    x, last, t = _inputs(s)
    pop = np.asarray(x.sum(axis=0)).ravel()
    grid = []
    for k in ks:
        for beta in betas:
            m = MarkovQRSVD(k=k, beta=beta, seed=seed).fit(t, pop)
            ranks = rank_targets(m, last, x, pairs, a.n_items)
            s = per_user(ranks, pairs[:, 0])
            score = {f"{metric}@10": float(s[metric].mean()), "ndcg@10": float(s["ndcg"].mean())}
            grid.append({"k": k, "beta": beta, **score})
            print(
                f"  tune k={k:<4} beta={beta:<5} valid {metric}@10={score[f'{metric}@10']:.4f}",
                flush=True,
            )
    best = max(grid, key=lambda g: (g[f"{metric}@10"], g["ndcg@10"]))
    return best, grid


def run_level(split, a, b, n_users, pairs, level, seed, k, beta, scaled):
    s = build_stream(split, a.n_items, b, n_users, level, "disjoint", seed)
    x, last, t = _inputs(s)
    pop = np.asarray(x.sum(axis=0)).ravel()
    out = {
        "mostpop": (rank_targets(MostPop().fit(x), x, x, pairs, a.n_items), None),
        "markov": (rank_targets(Markov().fit(t), last, x, pairs, a.n_items), None),
    }
    ranks_k = [("qr", k)] + (
        [("qr_scaled", round(k * (1 + level)))] if scaled and level > 0 else []
    )
    for name, kk in ranks_k:
        m = MarkovQRSVD(k=kk, beta=beta, seed=seed).fit(t, pop)
        out[name] = (
            rank_targets(m, last, x, pairs, a.n_items),
            float(m.base_share(a.n_items).mean()),
        )
    return out, s.info


def study(args):
    t_start = time.time()
    a, b, n_users = load_domains(args.a, args.b, args.cache)
    split, valid_end, test_end = make_split(a, args)
    metric = primary_metric(args)
    b_valid = before(b, valid_end) if valid_end is not None else b
    b_test = before(b, test_end) if test_end is not None else b
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.k is not None and args.beta is not None:
        k, beta, grid = args.k, args.beta, "skipped: k and beta given on the command line"
        print(f"1. tuning skipped, using k={k} beta={beta}", flush=True)
    else:
        print("1. tuning on validation (clean)", flush=True)
        valid_pairs = sample_pairs(split.valid, args.eval_users, 4285)
        best, grid = tune(split, a, b_valid, n_users, valid_pairs, args.ks, args.betas, 0, metric)
        k, beta = best["k"], best["beta"]
        print(f"   chosen k={k} beta={beta}", flush=True)

    print("2. test split, seeds x levels", flush=True)
    test_split = with_validation(split)
    test_pairs = sample_pairs(split.test, args.eval_users, 4285)
    if args.protocol == "time":
        tag = f"_time{args.cutoff}_w{args.window_days:g}_c{args.max_targets}"
    else:  # m = 1 keeps the original checkpoint names
        tag = f"_m{args.targets}" if args.targets > 1 else ""
    ckpt = out_dir / f"study_ckpt_k{k}_b{beta}_n{len(np.unique(test_pairs[:, 0]))}{tag}"
    ckpt.mkdir(exist_ok=True)
    user_scores, rows = {}, []
    for seed in args.seeds:
        for level in args.levels:
            f = ckpt / f"seed{seed}_level{level}.npz"
            if f.exists():  # resume: results of a finished (seed, level) are kept on disk
                z = np.load(f, allow_pickle=True)
                res = {n: (z[f"ranks_{n}"], z[f"share_{n}"].item()) for n in z["names"]}
                info = z["info"].item()
                print(f"   seed={seed} level={level} loaded from checkpoint", flush=True)
            else:
                t0 = time.time()
                res, info = run_level(
                    test_split, a, b_test, n_users, test_pairs, level, seed, k, beta, args.scaled
                )
                np.savez(
                    f,
                    names=np.array(list(res)),
                    info=np.array(info, dtype=object),
                    **{f"ranks_{n}": r for n, (r, _) in res.items()},
                    **{f"share_{n}": np.array(s, dtype=object) for n, (_, s) in res.items()},
                )
                print(f"   seed={seed} level={level} done in {time.time() - t0:.0f}s", flush=True)
            for name, (ranks, share) in res.items():
                scores = per_user(ranks, test_pairs[:, 0])
                user_scores[f"{name}|{seed}|{level}"] = scores[metric].astype(np.float32)
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "level": level,
                        "ndcg@10": float(scores["ndcg"].mean()),
                        "hit@10": float(scores["hit"].mean()),
                        "recall@10": float(scores["recall"].mean()),
                        "base_share": share,
                        "injected": info["injected"],
                        "frac_actual": info["frac_actual"],
                    }
                )
                print(
                    f"   seed={seed} level={level:<4} {name:<10} "
                    f"{metric}@10={scores[metric].mean():.4f}"
                    + (f" base_share={share:.2f}" if share is not None else ""),
                    flush=True,
                )

    print("3. paired changes vs clean", flush=True)
    changes = []
    for key, score in user_scores.items():
        name, seed, level = key.split("|")
        if float(level) == 0:
            continue
        ref = "qr" if name == "qr_scaled" else name
        base = user_scores[f"{ref}|{seed}|{args.levels[0]}"]
        pt, lo, hi, p = paired_change(base, score, args.boot, int(seed))
        changes.append(
            {
                "model": name,
                "seed": int(seed),
                "level": float(level),
                "change": pt,
                "ci_low": lo,
                "ci_high": hi,
                "p_not_worse": p,
            }
        )
    summary = {
        "scope": "amazon",
        "stamp": stamp,
        "eval_users": len(np.unique(test_pairs[:, 0])),
        "protocol": args.protocol,
        "primary_metric": f"{metric}@10",
        "k": k,
        "beta": beta,
        "tuning": grid,
        "rows": rows,
        "changes": changes,
        "minutes": round((time.time() - t_start) / 60, 1),
        "args": {kk: str(v) if isinstance(v, Path) else v for kk, v in vars(args).items()},
    }
    path = out_dir / f"study_{stamp}.json"
    save_json(path, summary)  # create-only, like every other report
    npz = out_dir / f"study_{stamp}_per_user.npz"
    if os.path.lexists(npz):
        raise FileExistsError(f"Refusing to overwrite artifact: {npz}")
    np.savez_compressed(npz, **user_scores)  # per-user primary metric
    for c in sorted(changes, key=lambda c: (c["model"], c["level"], c["seed"])):
        print(
            f"   {c['model']:<10} level={c['level']:<4} seed={c['seed']} change={c['change']:+.1%} "
            f"[{c['ci_low']:+.1%}, {c['ci_high']:+.1%}] p={c['p_not_worse']:.3f}",
            flush=True,
        )
    print(f"saved {path}")
