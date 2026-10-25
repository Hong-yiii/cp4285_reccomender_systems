"""Seed-level checks behind the grid results in the main checkout's CONTINUED-TRAINING.md.

    uv run python grid_checks.py RUNS_DIR AGGREGATE_JSON  # AGGREGATE_JSON from grid_aggregate.py
"""

import json
import sys
from pathlib import Path

import numpy as np

G = Path(sys.argv[1])
def load(d, s, b):
    p = G / f"lm-{d}-s{s}-{b}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None
D = ("ins", "vg", "cd"); S = (42, 43, 44)
print("Theta0 quality, blocks 1-3 (mean over seeds): hit@1, ndcg@10; categories")
for d in D:
    h = [np.mean([load(d, s, "init")["cells"][f"state0_block{j}"]["hit@1"] for j in (1,2,3)]) for s in S]
    n = [np.mean([load(d, s, "init")["cells"][f"state0_block{j}"]["ndcg@10"] for j in (1,2,3)]) for s in S]
    print(d, f"{np.mean(h):.3f} ± {np.std(h, ddof=1):.3f}", f"{np.mean(n):.3f}", load(d, 42, "init")["categories"])
def rank(run, cell): return run["cells"][cell]["push"]["mean_rank"]
print("\n(a) attacked-block rank change < 0, per seed (3% core):")
tot = neg = 0
for d in D:
    for c in ("shared", "iso"):
        clean = {s: load(d, s, f"{c}-clean") for s in S}
        for a in ("bw3", "clone3"):
            for s in S:
                r = load(d, s, f"{c}-{a}"); x = rank(r, "state1_block1") - rank(clean[s], "state1_block1")
                tot += 1; neg += x < 0
print(f"  {neg}/{tot}")
print("\n(b) isolated keeps more push than shared, paired by seed and attack")
for d in D:
    out = []
    for cell in ("state2_block2", "state4_block0"):
        wins, diffs = [], []
        for a in ("bw3", "clone3"):
            for s in S:
                iso = rank(load(d, s, f"iso-{a}"), cell) - rank(load(d, s, "iso-clean"), cell)
                sh = rank(load(d, s, f"shared-{a}"), cell) - rank(load(d, s, "shared-clean"), cell)
                wins.append(iso < sh); diffs.append(iso - sh)
        out.append(f"{cell} {sum(wins)}/6 (iso-shared {np.mean(diffs):+.2f})")
    print(f"  {d}: " + "; ".join(out))
print("\n(c) late rollback vs clean final state, mean over blocks 1-4")
for d in D:
    rows = []
    for a in ("bw3", "clone3"):
        for s in S:
            r = load(d, s, f"iso-{a}"); c = load(d, s, "iso-clean")["cells"]
            rb = r["rollback"]
            rows.append((np.mean([rb[f"state4_block{j}"]["push"]["hit@10"] for j in range(1,5)]),
                         np.mean([c[f"state4_block{j}"]["push"]["hit@10"] for j in range(1,5)]),
                         np.mean([rb[f"state4_block{j}"]["push"]["mean_rank"] - c[f"state4_block{j}"]["push"]["mean_rank"] for j in range(1,5)]),
                         np.mean([rb[f"state4_block{j}"]["hit@1"] - c[f"state4_block{j}"]["hit@1"] for j in range(5)])))
    rows = np.array(rows)
    print(f"  {d}: top10 {rows[:,0].mean():.3f} vs clean {rows[:,1].mean():.3f}; rank {rows[:,2].mean():+.2f} (below clean in {int((rows[:,2]>0).sum())}/6); hit@1 {rows[:,3].mean():+.4f} (range {rows[:,3].min():+.4f}..{rows[:,3].max():+.4f})")
print("\n(d) final-state |rank change| vs clean, blocks 0-3: immediate drop vs late rollback")
for d in D:
    early, late = [], []
    for a in ("bw3", "clone3"):
        for s in S:
            c = load(d, s, "iso-clean")["cells"]
            dr = load(d, s, f"iso-{a}-drop")["cells"]; r = load(d, s, f"iso-{a}")["rollback"]
            early.append(np.mean([abs(dr[f"state4_block{j}"]["push"]["mean_rank"] - c[f"state4_block{j}"]["push"]["mean_rank"]) for j in range(4)]))
            late.append(np.mean([abs(r[f"state4_block{j}"]["push"]["mean_rank"] - c[f"state4_block{j}"]["push"]["mean_rank"]) for j in range(4)]))
    hit_own = []
    for s in S:
        c = load(d, s, "iso-clean")["cells"]; dr = load(d, s, "iso-bw3-drop")["cells"]
        hit_own.append(np.mean([dr[f"state{b}_block{b}"]["hit@1"] - c[f"state{b}_block{b}"]["hit@1"] for b in (2,3,4)]))
    print(f"  {d}: early {np.mean(early):.2f}, late {np.mean(late):.2f} (early smaller in {sum(e<l for e,l in zip(early,late))}/6); drop own-block hit@1, blocks 2-4, vs clean {np.mean(hit_own):+.4f}")
print("\n(e) clean RetentionLoss_A hit@1 per condition")
for d in D:
    for c in ("shared", "iso"):
        v = [load(d, s, "init")["cells"]["state0_block0"]["hit@1"] - load(d, s, f"{c}-clean")["cells"]["state4_block0"]["hit@1"] for s in S]
        print(f"  {d} {c}: {np.mean(v):+.4f} ± {np.std(v, ddof=1):.4f}")
print("\n(f) |MixtureEffect_A| hit@1 over every attacked seed-branch")
agg = json.loads(Path(sys.argv[2]).read_text())["domains"]
vals, excl = [], 0
for Dd in agg.values():
    for r in Dd["branches"].values():
        if "mixture_effect_A_paired" in r:
            vals += r["mixture_effect_A_paired"]["values"]; excl += sum(r["mixture_effect_A_paired_ci_excludes_0"])
print(f"  n={len(vals)}, max |ME_A| {max(abs(v) for v in vals):.4f}, CI excludes 0 in {excl}")
