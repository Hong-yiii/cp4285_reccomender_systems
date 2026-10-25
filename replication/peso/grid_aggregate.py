"""Aggregate the corrupted-stream grid: per domain and branch, mean and SD over seeds.

    uv run python grid_aggregate.py RUNS_DIR DATA_DIR CATEGORIES_DIR OUTPUT_JSON  # from the main checkout

RUNS_DIR holds lm-<domain>-s<seed>-<branch> folders (metrics.json, dumps/).
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from cp4285.experiment.adapter_blocks import load, summarize
from cp4285.experiment.shilling import js, training_examples

DOMAINS = {"ins": "Instruments_cl_ori", "vg": "Video_Games_cl_0.6_0_5_3_0",
           "cd": "CDs_and_Vinyl_cl_0.6_0_5_3_0"}  # fmt: skip
NAME = re.compile(r"lm-(?P<d>[a-z]+)-s(?P<seed>\d+)-(?P<branch>.+)")


def ranks(run, cell):
    path = run / "dumps" / f"{cell}.jsonl"
    rows = [json.loads(x) for x in path.open()]
    return {r["user"]: r["rank"] for r in rows}


def paired_hit1(attacked, clean, cell, seed, draws=2000):
    """Mean per-user Hit@1 difference on the same users, with a 95% bootstrap interval."""
    a, c = ranks(attacked, cell), ranks(clean, cell)
    users = sorted(set(a) & set(c))
    d = np.array([(a[u] == 1) - (c[u] == 1) for u in users], float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975)), "users": len(users)}  # fmt: skip


def drift(data, cats_dir, name):
    blocks, category, _ = load(data / name, cats_dir / f"{name}-cat2/{name}.index.cat2.json",
                               cats_dir / f"{name}-cat2/category_stats.json")  # fmt: skip
    mixes = []
    for users in blocks:
        mixes.append(Counter(category[item] for _, item in training_examples(users)))
    return [js(mixes[b], mixes[b + 1]) for b in range(len(mixes) - 1)]


def push(cell, k=10):
    return cell["push"][f"hit@{k}"]


def per_run(folder, run, row, clean):
    cells = run["cells"]
    out = {
        "adaptation_hit1": row["adaptation"]["hit@1"],
        "retention_loss_A_hit1": row["retention_loss_A"]["hit@1"],
        "push_top10": {k: push(c) for k, c in cells.items() if "push" in c},
        "push_mean_rank": {k: c["push"]["mean_rank"] for k, c in cells.items() if "push" in c},
    }
    if "mixture_effect_A" in row:
        out["mixture_effect_A_hit1"] = row["mixture_effect_A"]["hit@1"]
        out["push_rank_change"] = {k: v["mean_rank"] for k, v in row["push_vs_clean"].items()}
        out["push_top10_change"] = {k: v["hit@10"] for k, v in row["push_vs_clean"].items()}
        out["mixture_effect_A_paired"] = paired_hit1(folder, clean, "state4_block0",
                                                     run["config"]["seed"])  # fmt: skip
    if "rollback" in run:
        out["rollback_push_top10"] = {k: push(c) for k, c in run["rollback"].items()}
        out["rollback_hit1"] = {k: c["hit@1"] for k, c in run["rollback"].items()}
    if "shills" in row:
        out["shill_auc"] = row["shills"]["separability_auc"]
        out["shill_js"] = row["shills"]["js_training_categories"]
    out["push_name"] = run.get("push_name") or str(run["config"]["push"])
    return out


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1
            else None, "n": len(values), "values": values}  # fmt: skip


def fold(rows):
    """Seed rows (dicts of numbers or dicts of numbers) -> the same shape with mean_sd leaves."""
    out = {}
    for key in rows[0]:
        first = rows[0][key]
        if isinstance(first, dict) and "mean" in first and "lo" in first:
            out[key] = mean_sd([r[key]["mean"] for r in rows if key in r])
            out[key + "_ci_excludes_0"] = [not (r[key]["lo"] <= 0 <= r[key]["hi"]) for r in rows if key in r]
        elif isinstance(first, dict):
            out[key] = {k: mean_sd([r[key].get(k) for r in rows if key in r]) for k in first}
        elif isinstance(first, str):
            out[key] = sorted({r[key] for r in rows})
        else:
            out[key] = mean_sd([r.get(key) for r in rows])
    return out


def main():
    runs_dir, data, cats, output = map(Path, sys.argv[1:5])
    groups = defaultdict(dict)
    for folder in sorted(runs_dir.iterdir()):
        m = NAME.fullmatch(folder.name)
        if m and (folder / "metrics.json").exists():
            groups[(m["d"], int(m["seed"]))][m["branch"]] = folder
    table = defaultdict(lambda: defaultdict(list))
    for (d, seed), branches in sorted(groups.items()):
        if "init" not in branches:
            continue
        others = {b: f for b, f in branches.items() if b != "init"}
        summary = summarize(branches["init"] / "metrics.json", [f / "metrics.json" for f in others.values()])
        runs = {b: json.loads((f / "metrics.json").read_text()) for b, f in others.items()}
        for b, folder in others.items():
            run = runs[b]
            row = summary["branches"][folder.name]
            key = (run["config"]["condition"], run["config"]["push"])
            clean = next((others[c] for c, r in runs.items()
                          if r["config"]["attack"] == "none"
                          and (r["config"]["condition"], r["config"]["push"]) == key), None)  # fmt: skip
            table[d][b].append(per_run(folder, run, row, clean))
        init = json.loads((branches["init"] / "metrics.json").read_text())
        table[d]["init"].append({"push_top10": {k: push(c) for k, c in init["cells"].items()},
                                 "state0_block0_hit1": init["cells"]["state0_block0"]["hit@1"]})  # fmt: skip
    result = {"domains": {}}
    for d, branches in table.items():
        result["domains"][d] = {
            "natural_drift_js": drift(data, cats, DOMAINS[d]),
            "branches": {b: fold(rows) for b, rows in sorted(branches.items())},
        }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=1))
    print(output)


if __name__ == "__main__":
    main()
