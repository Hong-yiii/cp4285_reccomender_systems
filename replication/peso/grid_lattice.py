"""Metric-to-metric pass over the corrupted-stream grid, focused on the continual-learning
question: how does a poisoned update wash out, what does it cost the old users, and what
do the washout curves share with ordinary forgetting? One paired record per attacked run,
then the cross-metric relations (dose -> spike -> decay -> residue -> cost -> stealth)
that the arm-by-arm contrasts never test. Read-only over the run folders; no GPU.

    uv run python grid_lattice.py RUNS_DIR OUTPUT_JSON [DRIFT_JSON]   # full pass
    uv run python grid_lattice.py RUNS_DIR --inspect [RUN]            # print one key tree

RUNS_DIR holds lm-<domain>-s<seed>-<branch> folders with metrics.json. Schema notes from
the 26-27 October runs: the attack type lives in the per-run `attack` dict (`kind`:
shill / real-history / relabel / demotion / vandalism) while `config.attack` is 'none'
for everything but bandwagon, clone and segment; `slow1` keeps per-block dicts under
`attacks`; isolated attacked runs carry a `rollback` dict (the same run scored with the
block-1 adapter removed) — the delete-only counterfactual; the `-drop*` branches are the
delete-and-retrain interventions. Clean pairing uses the matching runner vintage:
newer-vintage branches (top-level `rival_name`) pair with `clean-watch`, the rest with
`clean`. Rare branches pair with their own clean at the same push target.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

NAME = re.compile(r"lm-(?P<d>[a-z]+)-s(?P<seed>\d+)-(?P<branch>.+)")
BLOCKS = (1, 2, 3, 4)
KIND_LABEL = {"real-history": "hijack", "demotion": "demote", "vandalism": "vandal",
              "relabel": "relabel"}  # fmt: skip


def tree(obj, depth=0):
    """Key tree with scalar samples, for --inspect."""
    pad = "  " * depth
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                print(f"{pad}{key}: {type(value).__name__}[{len(value)}]")
                if depth < 3:
                    tree(value, depth + 1)
            else:
                print(f"{pad}{key} = {str(value)[:60]}")
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0]:")
        if depth < 3:
            tree(obj[0], depth + 1)


def attack_dict(run):
    value = run.get("attack")
    return value if isinstance(value, dict) and value else {}


def kind_of(run, branch):
    """Attack family from the run's own report, falling back to the config."""
    report = attack_dict(run)
    kind = report.get("kind")
    if kind in ("shill", None):  # shill reports do not name the recipe; config does
        kind = run.get("config", {}).get("attack") or kind
    if kind in (None, "none"):
        kind = "poisonbase" if "poisonbase" in branch else None
    return KIND_LABEL.get(kind, kind)


def per_block_attacks(run):
    value = run.get("attacks")
    return value if isinstance(value, dict) and value else {}


def dose_of(run, kind):
    """(pushed target lessons, total corrupted examples) from the kind's own report."""
    report = attack_dict(run)
    if kind == "hijack":
        pushed = report.get("inserted_examples")
        return pushed, pushed
    if kind == "relabel":
        pushed = report.get("examples_relabeled")
        return pushed, pushed
    if kind == "slow":
        blocks = per_block_attacks(run)
        totals = [b.get("shill_examples") for b in blocks.values() if isinstance(b, dict)]
        pushed = [b.get("target_share", 0.5) * b.get("shill_examples")
                  for b in blocks.values() if isinstance(b, dict)]  # fmt: skip
        pushed = sum(p for p in pushed if p is not None) or None
        total = sum(t for t in totals if t is not None) or None
        return pushed, total
    total = report.get("shill_examples")
    if total is None:
        return None, None
    share = report.get("target_share")
    if kind in ("vandal",) or share is None:
        return None, total  # random edits do not carry target items
    return share * total, total


def stealth_of(run):
    report = attack_dict(run)
    blocks = [b for b in per_block_attacks(run).values() if isinstance(b, dict)]
    aucs = [b["separability_auc"] for b in blocks if b.get("separability_auc") is not None]
    jss = [b["js_training_categories"] for b in blocks if b.get("js_training_categories") is not None]
    auc = report.get("separability_auc")
    js = report.get("js_training_categories")
    if auc is None and aucs:
        auc = float(np.mean(aucs))
    if js is None and jss:
        js = float(np.mean(jss))
    return auc, js


def is_clean(run, branch):
    config = run.get("config", {})
    if config.get("attack") not in (None, "none"):
        return False
    if attack_dict(run):
        return False
    return not ("poisonbase" in branch or branch.startswith("init"))


def is_drop(run, branch):
    return bool(run.get("config", {}).get("rollback_after")) or "drop" in branch


def drop_block(branch):
    m = re.search(r"drop(\d*)", branch)
    return int(m.group(1) or 1) if m else None


def cell_of(run, state, block):
    return run.get("cells", {}).get(f"state{state}_block{block}")


def push_at(run, state, block):
    cell = cell_of(run, state, block)
    return cell.get("push", {}).get("mean_rank") if cell and cell.get("push") else None


def delta(attacked, clean, state, block):
    """Paired metric differences on one cell; None when either side lacks the cell."""
    a, c = cell_of(attacked, state, block), cell_of(clean, state, block)
    if not a or not c:
        return None
    out = {}
    if a.get("push") and c.get("push"):
        out["push_rank"] = a["push"].get("mean_rank") - c["push"].get("mean_rank")
        out["push_hit5"] = a["push"].get("hit@5", 0.0) - c["push"].get("hit@5", 0.0)
    for metric in ("ndcg@10", "hit@1"):
        if metric in a and metric in c:
            out[metric.replace("@", "_")] = a[metric] - c[metric]
    return out or None


def decay_slope(points):
    """Log-linear slope of |effect| over blocks 2-4; None when not decaying."""
    pts = [(b, e) for b, e in points if b > 1 and e is not None and abs(e) > 1e-9]
    if len(pts) < 3:
        return None
    blocks = np.array([b for b, _ in pts], float)
    logs = np.log([abs(e) for _, e in pts])
    slope = float(np.polyfit(blocks, logs, 1)[0])
    return slope if slope < 0 else None


def record(run_name, run, clean, branch, kind):
    config = run.get("config", {})
    pushed, total = dose_of(run, kind)
    auc, js = stealth_of(run)
    own = [(b, delta(run, clean, b, b)) for b in BLOCKS]
    final = [(b, delta(run, clean, 4, b)) for b in BLOCKS]
    traj = [(b, d["push_rank"]) for b, d in own if d and d.get("push_rank") is not None]
    retention = delta(run, clean, 4, 0) or {}
    spike = own[0][1] or {}
    sustained = kind in ("slow", "poisonbase", "trigger")
    out = {
        "run": run_name,
        "domain": NAME.fullmatch(run_name)["d"],
        "seed": int(NAME.fullmatch(run_name)["seed"]),
        "branch": branch,
        "kind": kind,
        "family": family_label(run, branch, kind),
        "condition": config.get("condition"),
        "push": config.get("push"),
        "push_name": run.get("push_name"),
        "alpha": config.get("alpha"),
        "pushed_examples": pushed,
        "total_examples": total,
        "separability_auc": auc,
        "js_categories": js,
        "spike_push_rank": spike.get("push_rank"),
        "spike_push_hit5": spike.get("push_hit5"),
        "spike_ndcg": spike.get("ndcg_10"),
        "residue_push_rank": traj[-1][1] if traj else None,
        "residue_push_hit5": (own[-1][1] or {}).get("push_hit5"),
        "residue_same_users": (final[0][1] or {}).get("push_rank"),
        "decay_slope": None if sustained else decay_slope(traj),
        "cost_ndcg_mean_late": (float(np.mean([d["ndcg_10"] for _, d in own[1:]
                                               if d and "ndcg_10" in d]))
                                if any(d and "ndcg_10" in d for _, d in own[1:]) else None),
        "retention_ndcg": retention.get("ndcg_10"),
        "retention_hit1": retention.get("hit_1"),
    }
    if isinstance(run.get("rollback"), dict):
        rb = {"cells": run["rollback"]}  # the rollback dict stores cells at its top level
        vs_clean, vs_intact = [], []
        for b in BLOCKS:
            r, c, i = (push_at(rb, 4, b), push_at(clean, 4, b), push_at(run, 4, b))
            if r is not None and c is not None:
                vs_clean.append(r - c)
            if r is not None and i is not None:
                vs_intact.append(r - i)
        out["rollback_vs_clean_mean"] = float(np.mean(vs_clean)) if vs_clean else None
        out["rollback_vs_intact_mean"] = float(np.mean(vs_intact)) if vs_intact else None

    def segment(cell, base):
        """The trigger-group vs other-user split, paired against the clean run."""
        if not cell or not base or "push_trigger" not in cell or "push_trigger" not in base:
            return {}
        return {
            "trigger_delta": cell["push_trigger"]["mean_rank"] - base["push_trigger"]["mean_rank"],
            "other_delta": cell["push_other"]["mean_rank"] - base["push_other"]["mean_rank"],
            "trigger_users": cell["push_trigger"].get("users"),
        }

    s1 = segment(cell_of(run, 1, 1), cell_of(clean, 1, 1))
    s4 = segment(cell_of(run, 4, 4), cell_of(clean, 4, 4))
    out.update({"trigger_spike": s1.get("trigger_delta"),
                "other_spike": s1.get("other_delta"),
                "trigger_residue": s4.get("trigger_delta"),
                "other_residue": s4.get("other_delta"),
                "trigger_users": s4.get("trigger_users")})
    return out


def family_label(run, branch, kind):
    label = kind or "unknown"
    config = run.get("config", {})
    alpha = config.get("alpha")
    if alpha and kind not in ("relabel", "poisonbase"):
        label += f"@{round(alpha * 100)}%"
    if "rare" in branch or config.get("push_share") == 0.005:
        label += "+rare"
    return label


def rebound_record(run_name, run, clean, branch, intact):
    """Delete-and-retrain intervention: final push rank vs clean and vs the intact run."""
    config = run.get("config", {})
    vs_clean, vs_intact = [], []
    for b in BLOCKS:
        r, c = push_at(run, 4, b), push_at(clean, 4, b)
        if r is not None and c is not None:
            vs_clean.append(r - c)
        if intact is not None:
            i = push_at(intact, 4, b)
            if r is not None and i is not None:
                vs_intact.append(r - i)
    return {
        "run": run_name,
        "domain": NAME.fullmatch(run_name)["d"],
        "seed": int(NAME.fullmatch(run_name)["seed"]),
        "branch": branch,
        "condition": config.get("condition"),
        "drop_block": drop_block(branch),
        "rebound_vs_clean_mean": float(np.mean(vs_clean)) if vs_clean else None,
        "rebound_vs_intact_mean": float(np.mean(vs_intact)) if vs_intact else None,
    }


def spearman(x, y):
    def ranks(values):
        order = np.argsort(values)
        out = np.empty(len(values), float)
        out[order] = np.arange(len(values), dtype=float)
        return out
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(ranks(np.asarray(x, float)), ranks(np.asarray(y, float)))[0, 1])


def relation(records, key_x, key_y, rows, note, signed=False):
    pairs = [(r[key_x], r[key_y]) for r in records
             if r.get(key_x) is not None and r.get(key_y) is not None]
    if len(pairs) < 3:
        return
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    rho = spearman(x, y)
    if signed:
        agree = sum((a < 0) == (b < 0) for a, b in pairs)
        signs = f"{agree}/{len(pairs)} agree in sign"
    else:
        signs = "—"
    rows.append(f"| {key_x} -> {key_y} | {len(pairs)} | {rho if rho is None else f'{rho:+.2f}'} "
                f"| {signs} | {note} |")  # fmt: skip


def main():
    argv = [a for a in sys.argv[1:] if a != "--inspect"]
    inspect = "--inspect" in sys.argv
    runs_dir = Path(argv[0])
    groups = {}
    for folder in sorted(runs_dir.iterdir()):
        m = NAME.fullmatch(folder.name)
        if m and (folder / "metrics.json").exists():
            groups.setdefault((m["d"], int(m["seed"])), {})[m["branch"]] = folder
    if inspect:
        target = argv[1] if len(argv) > 1 else next(iter(next(iter(groups.values())).values()))  # fmt: skip
        run = json.loads((runs_dir / target / "metrics.json").read_text())
        print(f"# {target}\nkey tree (scalars truncated):")
        tree(run)
        return
    output = Path(argv[1])
    drift = {}
    if len(argv) > 2:
        agg = json.loads(Path(argv[2]).read_text()).get("domains", {})
        drift = {d: float(np.mean(v.get("natural_drift_js", [np.nan])))
                 for d, v in agg.items() if v.get("natural_drift_js")}  # fmt: skip
    records, rebounds, skipped = [], [], []
    prefix = {"isolated": "iso", "shared": "shared"}
    for (d, seed), branches in sorted(groups.items()):
        runs = {b: json.loads((f / "metrics.json").read_text()) for b, f in branches.items()}
        branch_runs = {b: r for b, r in runs.items()
                       if r.get("config", {}).get("stage") == "branch"}  # fmt: skip
        for branch, run in sorted(branch_runs.items()):
            if not kind_of(run, branch):
                continue
            config = run.get("config", {})
            watch = "rival_name" in run
            cond = prefix.get(config.get("condition"), config.get("condition"))
            mate = None
            for base in (f"{cond}-clean-rare", f"{cond}-clean-watch", f"{cond}-clean"):
                if base not in runs or not is_clean(runs[base], base):
                    continue
                if base.endswith("watch") and not watch:
                    continue  # plain-vintage runs pair with the plain clean
                if runs[base].get("config", {}).get("push") != config.get("push"):
                    continue  # the clean must score the same push target
                mate = base
                break
            if mate is None:
                skipped.append(branch)
                continue
            clean = runs[mate]
            if is_drop(run, branch):
                fam = (family_label(run, branch, kind_of(run, branch)) or "")
                intact = next((runs[b] for b in branch_runs
                               if b != branch and not is_drop(runs[b], b)
                               and kind_of(runs[b], b)
                               and family_label(runs[b], b, kind_of(runs[b], b)) == fam), None)  # fmt: skip
                rebounds.append(rebound_record(f"lm-{d}-s{seed}-{branch}", run, clean, branch, intact))
                continue
            rec = record(f"lm-{d}-s{seed}-{branch}", run, clean, branch, kind_of(run, branch))
            rec["domain_drift_js"] = drift.get(rec["domain"])
            records.append(rec)
    print(f"{len(records)} attacked runs paired; {len(rebounds)} interventions; "
          f"{len(skipped)} without a clean match"
          + (f": {', '.join(skipped[:10])}" if skipped else ""))  # fmt: skip
    one_off = [r for r in records if r["decay_slope"] is not None]
    rows = ["| relation | n | Spearman rho | signs | note |", "| --- | --- | --- | --- | --- |"]
    for fam in sorted({r["family"].split("@")[0] for r in records if "@" in (r["family"] or "")}):
        subset = [r for r in records if (r["family"] or "").startswith(fam + "@")]
        relation(subset, "pushed_examples", "spike_push_rank", rows, f"within {fam}")
    relation(one_off, "spike_push_rank", "residue_push_rank", rows, "one-off attacks", signed=True)
    for cond in ("isolated", "shared"):
        subset = [r for r in one_off if r.get("condition") == cond]
        relation(subset, "spike_push_rank", "residue_push_rank", rows,
                 f"one-off, {cond} layout only", signed=True)
    relation(one_off, "spike_push_rank", "residue_same_users", rows,
             "one-off, block-1 users at final state", signed=True)
    relation(one_off, "spike_push_rank", "decay_slope", rows, "one-off attacks")
    relation(records, "separability_auc", "spike_push_rank", rows, "stealth vs immediate")
    relation(records, "separability_auc", "residue_push_rank", rows, "stealth vs lasting")
    relation(one_off, "spike_push_rank", "cost_ndcg_mean_late", rows, "promotion vs task cost")
    relation(one_off, "spike_push_rank", "retention_ndcg", rows, "promotion vs old-user cost",
             signed=True)
    relation(one_off, "domain_drift_js", "decay_slope", rows,
             "natural drift vs washout (n = domains)")
    relation(one_off, "domain_drift_js", "residue_push_rank", rows,
             "natural drift vs lasting residue (n = domains)")
    rb = [r for r in records if r.get("rollback_vs_intact_mean") is not None]
    relation(rb, "residue_push_rank", "rollback_vs_intact_mean", rows,
             "scoring-time delete: intact residue vs what deletion restores", signed=True)
    relation(rb, "spike_push_rank", "rollback_vs_clean_mean", rows,
             "scoring-time delete: spike vs post-delete rank")
    print("\n".join(rows))
    families = {}
    for r in records:
        families.setdefault(r["family"], []).append(r)

    def mean_of(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return f"{np.mean(vals):+.3f}" if vals else "—"

    print("\nfamily means (spike, residue, same-users, hit5, dose pushed, auc):")
    for name, rs in sorted(families.items()):
        print(f"  {name}: n={len(rs)} spike {mean_of(rs, 'spike_push_rank')} "
              f"residue {mean_of(rs, 'residue_push_rank')} "
              f"same-users {mean_of(rs, 'residue_same_users')} "
              f"hit5 {mean_of(rs, 'spike_push_hit5')} "
              f"pushed {mean_of(rs, 'pushed_examples')} auc {mean_of(rs, 'separability_auc')}")  # fmt: skip
    rare = [r for r in records if (r["family"] or "").endswith("+rare")]
    if rare:
        print("\nrare-target vs standard share (frequency-dependent forgetting):")
        for name in sorted({r["family"].replace("+rare", "") for r in rare}):
            for tag, subset in (("rare", [r for r in rare if r["family"] == name + "+rare"]),
                                ("std", [r for r in records if r["family"] == name])):
                if subset:
                    print(f"  {name} [{tag}]: n={len(subset)} "
                          f"spike {mean_of(subset, 'spike_push_rank')} "
                          f"residue {mean_of(subset, 'residue_push_rank')} "
                          f"retention {mean_of(subset, 'retention_ndcg')}")  # fmt: skip
    if rb:
        print("\nscoring-time delete (rollback dict; mean over blocks 1-4):")
        for r in sorted(rb, key=lambda x: (x["family"], x["domain"], x["seed"]))[:12]:
            print(f"  {r['run']}: vs clean {r['rollback_vs_clean_mean']:+.2f} "
                  f"vs intact {r['rollback_vs_intact_mean']:+.2f}")  # fmt: skip
    if rebounds:
        print("\ndelete-and-retrain interventions:")
        for r in sorted(rebounds, key=lambda x: (x["condition"], x.get("drop_block") or 0)):
            print(f"  {r['run']}: after block {r.get('drop_block')} "
                  f"vs clean {r.get('rebound_vs_clean_mean'):+.2f} "
                  f"vs intact {r.get('rebound_vs_intact_mean') if r.get('rebound_vs_intact_mean') is not None else '—'}")  # fmt: skip
    if any(r["kind"] == "trigger" for r in records):
        print("\ntrigger runs are in the records, but the segment split needs the per-user "
              "dumps (dumps/state*_block*.jsonl), not metrics.json — analyze separately.")  # fmt: skip
    trig = [r for r in records if r.get("trigger_residue") is not None]
    if trig:
        print("\nsegment cut, paired vs clean (trigger group = history has the trigger):")
        for fam in sorted({r["family"] for r in trig}):
            for cond in ("isolated", "shared"):
                sub = [r for r in trig if r["family"] == fam and r["condition"] == cond]
                if not sub:
                    continue

                def stat(rs, key):
                    vals = [r[key] for r in rs if r.get(key) is not None]
                    if not vals:
                        return "—"
                    return (f"{np.mean(vals):+.3f} "
                            f"({sum(v < 0 for v in vals)}/{len(vals)} promoted)")  # fmt: skip

                diffs = [r["trigger_residue"] - r["other_residue"] for r in sub
                         if r.get("trigger_residue") is not None
                         and r.get("other_residue") is not None]  # fmt: skip
                diff = (f"{np.mean(diffs):+.3f} ({sum(d < 0 for d in diffs)}/{len(diffs)} "
                        f"trig < other)") if diffs else "—"  # fmt: skip
                print(f"  {fam} [{cond}]: n={len(sub)} "
                      f"spike trig {stat(sub, 'trigger_spike')} "
                      f"other {stat(sub, 'other_spike')} | "
                      f"residue trig {stat(sub, 'trigger_residue')} "
                      f"other {stat(sub, 'other_residue')} | residue differential {diff}")  # fmt: skip
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.write_text(json.dumps({"runs": records, "relations": rows,
                                  "interventions": rebounds}, indent=1))  # fmt: skip
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
