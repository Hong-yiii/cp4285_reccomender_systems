"""Plan files for adapter_grid.sbatch: the corrupted-stream grid over three PESO domains.

Each plan is one domain and seed, its init first, so one job (and its afterany follow-up,
which skips finished runs) covers it. Every run pushes the category whose blocks 1-4 test
share is nearest PUSH_SHARE; the rare-target runs use RARE_SHARE.

    python grid_plans.py OUTPUT_DIR WORKSPACE  # WORKSPACE is the cluster's $W
"""

import sys
from pathlib import Path

DOMAINS = {"ins": "Instruments_cl_ori", "vg": "Video_Games_cl_0.6_0_5_3_0",
           "cd": "CDs_and_Vinyl_cl_0.6_0_5_3_0"}  # fmt: skip
SEEDS = (42, 43, 44)
PUSH_SHARE, RARE_SHARE = 0.013, 0.005
ATTACKS = {"bw": "bandwagon", "clone": "clone"}
CONDITIONS = {"shared": "shared", "iso": "isolated"}


def branch(condition, attack=None, alpha=None, drop=False):
    args = f"--condition {CONDITIONS[condition]}"
    if attack:
        args += f" --attack {ATTACKS[attack]} --alpha {alpha}"
    return args + (" --rollback-after" if drop else "")


def label(condition, attack=None, alpha=None, drop=False):
    if not attack:
        return f"{condition}-clean"
    return f"{condition}-{attack}{round(alpha * 100)}" + ("-drop" if drop else "")


CORE = [("shared",), ("iso",)] + [
    (c, a, 0.03) for c in CONDITIONS for a in ATTACKS
] + [("iso", a, 0.03, True) for a in ATTACKS]
DOSE = [(c, a, alpha) for alpha in (0.01, 0.05) for c in CONDITIONS for a in ATTACKS]
RARE = [("iso",)] + [("iso", a, 0.03) for a in ATTACKS]


def runs(short, seed, base, specs, share, suffix=""):
    return [(f"lm-{short}-s{seed}-{label(*s)}{suffix}",
             f"{base} --push-share {share} {branch(*s)}") for s in specs]  # fmt: skip


def plans(workspace):
    out = {}
    for short in DOMAINS:
        for seed in SEEDS:
            init = f"lm-{short}-s{seed}-init"
            common = f"--seed {seed} --dumps"
            lines = [(init, f"--stage init {common} --push-share {PUSH_SHARE}")]
            base = f"--stage branch --init {workspace}/lm-runs/{init} {common}"
            lines += runs(short, seed, base, CORE, PUSH_SHARE)
            if seed == SEEDS[0]:
                lines += runs(short, seed, base, RARE, RARE_SHARE, "-rare")
            out[f"core-{short}-s{seed}"] = lines
            if short == "ins":
                out[f"dose-{short}-s{seed}"] = runs(short, seed, base, DOSE, PUSH_SHARE)
    return out


def main():
    output, workspace = Path(sys.argv[1]), sys.argv[2]
    output.mkdir(parents=True, exist_ok=True)
    for name, lines in plans(workspace).items():
        path = output / f"{name}.tsv"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        path.write_text("".join(f"{run}\t{args}\n" for run, args in lines))
        print(name, len(lines))


if __name__ == "__main__":
    main()
