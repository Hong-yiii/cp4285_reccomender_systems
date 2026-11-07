"""Does a poison move the ranking less as category labels get finer?

The fourth STePS image shows forgetting at 30, 75 and 299 Instruments categories. This
script asks the companion question for the corrupted-stream study: at each label rung, how
far does a 3% attack move the pushed category, how much of that survives three clean
updates, and what does removing the poisoned adapter do?

Inputs are `cp4285 adapter-blocks` run folders under one `--root`. Every level names the
run prefix that holds its seeds (`lm-ins` for level 2, `lm-ins-cat3`, `lm-ins-cat4`).
Attacked and clean runs are paired on user through the runner's `--dumps`, using the same
endpoints as the dose summary (mean rank change, top-5, top-10 churn, normalised gain).

    python granularity_attack.py OUTPUT --root <lm-runs> \
        --level 2 lm-ins --level 3 lm-ins-cat3 --level 4 lm-ins-cat4

Writes one create-only JSON report and prints a markdown table. A missing cell is reported,
never imputed.
"""

import argparse
import json
import statistics
from pathlib import Path

from cp4285.experiment import dose

ARMS = ("iso-clean", "shared-clean", "iso-bw3", "shared-bw3", "iso-bw3-drop")
PAIRS = {
    "immediate_iso": ("iso-bw3", "state1_block1", "iso-clean", "state1_block1"),
    "immediate_shared": ("shared-bw3", "state1_block1", "shared-clean", "state1_block1"),
    "persistence_shared": ("shared-bw3", "state4_block1", "shared-clean", "state4_block1"),
    "persistence_iso": ("iso-bw3", "state4_block1", "iso-clean", "state4_block1"),
    "final_iso": ("iso-bw3", "state4_block4", "iso-clean", "state4_block4"),
    "rollback_iso": ("iso-bw3", "rollback_block1", "iso-clean", "state4_block1"),
    "drop_immediate": ("iso-bw3-drop", "dropped1_block1", "iso-clean", "state1_block1"),
}
SUMMARY = ("hit@1", "ndcg@10", "mrr@10")


def load(run_dir):
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text()), Path(run_dir)


def cell_path(run, name):
    """A scored cell lives in `cells`; an end-of-run rollback cell uses its dump name."""
    if name.startswith("rollback_block"):
        return f"state4_block{name.removeprefix('rollback_block')}" in run.get("rollback", {})
    return name in run.get("cells", {})


def cell_value(run, name):
    """The metrics cell behind a dump name (rollback dumps use the state4 metrics key)."""
    if name.startswith("rollback_block"):
        return run.get("rollback", {}).get(f"state4_block{name.removeprefix('rollback_block')}")
    return run.get("cells", {}).get(name)


def from_cells(run, cell, clean, clean_cell):
    """Aggregate fallback when a dump predates push_rank (the October level-2 grid)."""
    attacked = cell_value(run, cell)["push"]
    reference = cell_value(clean, clean_cell)["push"]
    rank = reference["mean_rank"]
    gain = (rank - attacked["mean_rank"]) / (rank - 1) if rank and rank > 1 else None
    return {
        "users": attacked["users"], "source": "metrics", "mean_rank":
        attacked["mean_rank"] - rank, "ci_low": None, "ci_high": None,
        "top1": attacked.get("hit@1"), "top5": attacked.get("hit@5"),
        "top10": attacked.get("hit@10"),
        "clean_top5": reference["hit@5"], "churn": None, "normalised_gain": gain,
        "top10_change": None, "mean_abs_rank": None,
    }  # fmt: skip


def endpoints(run, run_dir, cell, clean, clean_dir, clean_cell, push, seed):
    try:
        pairing = dose.pair(dose.load_rows(run_dir, cell), dose.load_rows(clean_dir, clean_cell),
                            push)  # fmt: skip
    except (ValueError, FileNotFoundError):
        return from_cells(run, cell, clean, clean_cell)
    out = dose.endpoints(pairing, seed=seed + 17, draws=1000)
    out["source"] = "dumps"
    pairs = zip(pairing["attacked_top"], pairing["clean_top"])
    overlaps = [len(set(a) & set(c)) / max(len(a), len(c))
                for a, c in pairs if a is not None and c is not None]  # fmt: skip
    out["top10_change"] = 1 - sum(overlaps) / len(overlaps) if overlaps else None
    out["mean_abs_rank"] = sum(abs(d) for d in pairing["deltas"]) / len(pairing["deltas"])
    return out


def seed_view(root, prefix, seed):
    names = {arm: Path(root) / f"{prefix}-s{seed}-{arm}" for arm in ARMS}
    runs = {}
    for arm, path in names.items():
        if (path / "metrics.json").exists():
            runs[arm] = load(path)
    if "iso-clean" not in runs and "shared-clean" not in runs:
        return {"seed": seed, "missing": [a for a in ARMS if a not in runs]}
    clean, _ = runs.get("iso-clean") or runs["shared-clean"]
    push = clean.get("config", {}).get("push")
    out = {"seed": seed, "categories": clean.get("categories"), "push": push,
           "push_name": clean.get("push_name"), "pairs": {}, "task": {}}  # fmt: skip
    for label, (arm, cell, clean_arm, clean_cell) in PAIRS.items():
        if arm not in runs or clean_arm not in runs:
            out["pairs"][label] = None
            continue
        run, run_dir = runs[arm]
        ref, ref_dir = runs[clean_arm]
        if not cell_path(run, cell) or not cell_path(ref, clean_cell):
            out["pairs"][label] = None
            continue
        out["pairs"][label] = endpoints(run, run_dir, cell, ref, ref_dir, clean_cell, push, seed)
    if "iso-bw3" in runs and "iso-clean" in runs:
        attacked_run, _ = runs["iso-bw3"]
        reference_run, _ = runs["iso-clean"]
        for metric in SUMMARY:
            attacked = attacked_run["cells"]["state1_block1"].get(metric)
            reference = reference_run["cells"]["state1_block1"].get(metric)
            out["task"][metric] = None if attacked is None else attacked - reference
    out["clean_rank"] = clean["cells"]["state1_block1"].get("push", {}).get("mean_rank")
    out["clean_top5"] = clean["cells"]["state1_block1"].get("push", {}).get("hit@5")
    out["clean_top10"] = clean["cells"]["state1_block1"].get("push", {}).get("hit@10")
    return out


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}  # fmt: skip


def aggregate(seeds):
    out = {"summary": {}}
    for label in PAIRS:
        rows = [s["pairs"].get(label) for s in seeds if s.get("pairs")]
        out["summary"][label] = {
            key: mean_sd([r[key] for r in rows if r])
            for key in ("mean_rank", "mean_abs_rank", "top5", "top10", "top10_change",
                    "churn", "normalised_gain")
        }
    out["summary"]["task_hit@1"] = mean_sd([s["task"].get("hit@1") for s in seeds])
    out["summary"]["clean_rank"] = mean_sd([s.get("clean_rank") for s in seeds])
    out["summary"]["clean_top5"] = mean_sd([s.get("clean_top5") for s in seeds])
    out["summary"]["clean_top10"] = mean_sd([s.get("clean_top10") for s in seeds])
    return out


def report_level(root, prefix, level, seeds):
    rows = [seed_view(root, prefix, seed) for seed in seeds]
    valid = [s for s in rows if s.get("pairs")]
    if not valid:
        return {"level": level, "prefix": prefix, "seeds": rows, "summary": None}
    return {"level": level, "prefix": prefix, "categories": valid[0]["categories"],
            "push": valid[0]["push"], "push_name": valid[0]["push_name"],
            "seeds": rows, **aggregate(valid)}  # fmt: skip


def fmt(summary, key, field="mean_rank", digits=2, signed=True):
    value = summary.get(key)
    if not value:
        return "n/a"
    stat = value if "mean" in value else value.get(field)
    if not stat or stat.get("mean") is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{stat['mean']:{sign}.{digits}f} ± {stat['sd']:.{digits}f}"


def markdown(result):
    lines = ["### Poisoning across category granularity", "",
             ("Instruments, 3% bandwagon attack in block 1, isolated adapters unless stated. "
              "Mean ± sd over seeds; negative rank change means the target moved up."), "",
             ("| Level | Categories | Target | Clean rank | Immediate (iso) | Top-5 lift (pp) | "
              "Top-10 lift (pp) | After 3 clean updates | Rollback at end | Drop immediately | "
              "Top-10 slots changed |"),
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for level in result["levels"]:
        summary = level.get("summary")
        if not summary:
            lines.append(f"| {level['level']} | {level.get('categories', '?')} | missing |")
            continue
        immediate = summary["immediate_iso"]["top5"]
        clean = summary["clean_top5"]
        lift = ("n/a" if immediate is None or clean is None
                else f"{(immediate['mean'] - clean['mean']) * 100:+.1f}")  # fmt: skip
        top10 = summary["immediate_iso"]["top10"]
        clean10 = summary.get("clean_top10")
        lift10 = ("n/a" if top10 is None or clean10 is None
                  else f"{(top10['mean'] - clean10['mean']) * 100:+.1f}")  # fmt: skip
        change = summary["immediate_iso"]["top10_change"]
        churn = ("n/a" if change is None
                 else f"{change['mean'] * 100:.1f}% ± {change['sd'] * 100:.1f}")  # fmt: skip
        lines.append(
            f"| {level['level']} | {level['categories']} | {level['push_name']} | "
            f"{fmt(summary, 'clean_rank')} | {fmt(summary, 'immediate_iso')} | "
            f"{lift} | {lift10} | {fmt(summary, 'persistence_iso')} | "
            f"{fmt(summary, 'rollback_iso')} | {fmt(summary, 'drop_immediate')} | {churn} |"
        )
    lines += ["", ("| Level | Shared, immediate | Shared, after 3 clean | "
                    "Isolated, immediate | Isolated, after 3 clean | "
                    "Isolated rollback at end | Isolated drop immediately |"),
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for level in result["levels"]:
        summary = level.get("summary")
        if not summary:
            continue
        lines.append(
            f"| {level['level']} | {fmt(summary, 'immediate_shared')} | "
            f"{fmt(summary, 'persistence_shared')} | {fmt(summary, 'immediate_iso')} | "
            f"{fmt(summary, 'persistence_iso')} | {fmt(summary, 'rollback_iso')} | "
            f"{fmt(summary, 'drop_immediate')} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--root", required=True, help="Folder holding the run folders")
    parser.add_argument("--level", nargs=2, action="append", required=True,
                        metavar=("LEVEL", "PREFIX"))  # fmt: skip
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    seeds = [int(x) for x in args.seeds.split(",") if x]
    result = {"root": str(args.root), "seeds": seeds, "levels": []}
    for level, prefix in args.level:
        result["levels"].append(report_level(args.root, prefix, int(level), seeds))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(markdown(result))


if __name__ == "__main__":
    main()
