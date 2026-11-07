"""Plan files for the category-granularity poisoning grid (Instruments levels 2-4).

The fourth STePS image compares forgetting at 30, 75 and 299 categories. This plan runs the
corrupted-stream protocol from `docs/CONTINUED-TRAINING.md` at each of those rungs, so the
pushed category's rank change can be compared across granularity. Level 2 already exists in
the October grid; levels 3 and 4 are new.

Each level needs its own category index and stats under `<workspace>/categories/
Instruments_cl_ori-cat<N>/`, and its PESO-format blocks under `<workspace>/peso/data/`
(`Instruments_cl_ori` for level 2, `Instruments_cl_cat<N>` deeper). The plan only writes
launch lines; it never submits them.

    python granularity_plans.py OUTPUT_DIR WORKSPACE --levels 3,4 --seeds 42,43,44

Writes `runs.tsv` with columns run, data, categories, names and argv. `submit_granularity.sh`
consumes it.
"""

import argparse
from pathlib import Path

PUSH_SHARE = 0.013
ARMS = ("init", "shared-clean", "iso-clean", "shared-bw3", "iso-bw3", "iso-bw3-drop")
HIJACK_ARMS = ("shared-hijack3", "iso-hijack3")


def run_name(level, seed, arm):
    prefix = "lm-ins" if level == 2 else f"lm-ins-cat{level}"
    return f"{prefix}-s{seed}-{arm}"


def paths(workspace, level):
    workspace = Path(workspace)
    data = workspace / "peso/data" / ("Instruments_cl_ori" if level == 2
                                     else f"Instruments_cl_cat{level}")  # fmt: skip
    cat_dir = workspace / "categories" / f"Instruments_cl_ori-cat{level}"
    index = cat_dir / f"Instruments_cl_ori.index.cat{level}.json"
    return data, index, cat_dir / "category_stats.json"


def args_for(workspace, level, seed, arm):
    workspace = Path(workspace)
    init = workspace / "lm-runs" / run_name(level, seed, "init")
    common = ["--push-share", str(PUSH_SHARE), "--dumps"]
    if arm == "init":
        return ["--stage", "init", *common]
    base = ["--stage", "branch", "--init", str(init), "--seed", str(seed), *common]
    if arm.endswith("-clean"):
        condition = "shared" if arm.startswith("shared") else "isolated"
        return [*base, "--condition", condition]
    condition = "shared" if arm.startswith("shared") else "isolated"
    attack = "bandwagon" if "-bw" in arm else "real-history"
    extra = (["--attack", "bandwagon"] if attack == "bandwagon"
             else ["--corruption", "real-history"])
    extra += ["--alpha", "0.03"]
    if arm.endswith("-drop"):
        extra.append("--rollback-after")
    return [*base, "--condition", condition, *extra]


def plan(workspace, levels, seeds, hijack_seeds, gres="h100-47", gres_map=None):
    """One record per run; `gres_map` assigns a GPU type to a level and seed group."""
    runs = []
    gres_map = gres_map or {}
    for level in levels:
        data, index, names = paths(workspace, level)
        arms = list(ARMS)
        for seed in seeds:
            card = gres_map.get((level, seed), gres)
            arms_for_seed = arms + (list(HIJACK_ARMS) if seed in hijack_seeds else [])
            for arm in arms_for_seed:
                runs.append({
                    "run": run_name(level, seed, arm), "level": level, "seed": seed,
                    "arm": arm, "data": str(data), "categories": str(index),
                    "names": str(names), "argv": args_for(workspace, level, seed, arm),
                    "gres": card,
                })  # fmt: skip
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("workspace")
    parser.add_argument("--levels", default="3,4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--hijack-seeds", default="", help="Seeds that also run real-history attacks")
    parser.add_argument("--gres", default="h100-47", help="Default GPU type")
    parser.add_argument("--gres-map", default="",
                        help="Per group overrides, e.g. 3:42=h100-47,4:42=h200-141")
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    hijack = {int(x) for x in args.hijack_seeds.split(",") if x}
    gres_map = {}
    for item in args.gres_map.split(","):
        if item:
            key, card = item.split("=")
            level, seed = (int(x) for x in key.split(":"))
            gres_map[(level, seed)] = card
    if any(level not in (2, 3, 4) for level in levels):
        raise ValueError("This grid is defined for levels 2, 3 and 4")
    runs = plan(args.workspace, levels, seeds, hijack, args.gres, gres_map)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "runs.tsv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("x") as f:
        for r in runs:
            f.write("\t".join([r["run"], r["data"], r["categories"], r["names"],
                               " ".join(r["argv"]), r["gres"]]) + "\n")  # fmt: skip
    print(f"{len(runs)} runs -> {path}")


if __name__ == "__main__":
    main()
