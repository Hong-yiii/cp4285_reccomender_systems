"""Plan files for the focused wave 1, continual-learning edition: the rare-target
replication (frequency-dependent forgetting). The TSVs feed adapter_grid.sbatch exactly
like grid_plans.py's; every file is create-only.

    python focus_plans.py OUTPUT_DIR WORKSPACE   # WORKSPACE is the cluster's $W

Arms (18 new runs, about 3.5 GPU-hours, isolated layout only): rare-target clean, bandwagon
3% and clone 3% for seeds 43-44, three domains — seed 42's rare runs exist from the core
grid. Pairing uses the matching clean at the same push target, so each rare attacked run
measures promotion, residue and old-user retention against its own clean baseline.

The low-and-slow-that-stop arm (1% in blocks 1-2 only) is deferred: the runner's
--placement accepts block|all|init only, so "attack then stop" needs a placement change
plus tests; it returns here once that lands. The earlier dose-matched real-vs-fake arms
were cut after the metric lattice answered the dose question (per-lesson efficiency differs
by kind; see docs/PESO-REPLICATION.md, "Metric-to-metric lattice").
"""

import sys
from pathlib import Path

DOMAINS = {"ins": "Instruments_cl_ori", "vg": "Video_Games_cl_0.6_0_5_3_0",
           "cd": "CDs_and_Vinyl_cl_0.6_0_5_3_0"}  # fmt: skip
SEEDS = (42, 43, 44)
PUSH_SHARE, RARE_SHARE = 0.013, 0.005
ATTACKS = {"bw": "bandwagon", "clone": "clone"}  # verified against the cluster's plan files
RARE_ARMS = [("iso-clean-rare", None, None, ""), ("iso-bw3-rare", "bw", 0.03, ""),
             ("iso-clone3-rare", "clone", 0.03, "")]  # fmt: skip
# Discriminator (CDs, seed 42): is Blue Note's persistence the label semantics or the
# domain's buried-tail geometry? Push a second jazz label (Verve, 0.29% test share) and a
# non-label genre (Reggae, 0.17%), each tracking the other beside it — one clean run covers
# both targets, so three runs answer what six would. Reference: Blue Note 0.55%, in the
# lattice as bandwagon@3%+rare.
RARE2 = [("iso-clean-rare2", "--push 57 --rival 44"),
         ("iso-bw3-rare2a", "--push 57 --rival 44 --attack bandwagon --alpha 0.03"),
         ("iso-bw3-rare2b", "--push 44 --rival 57 --attack bandwagon --alpha 0.03")]  # fmt: skip
# Deferred until the runner grows a "corrupt blocks 1-2 then stop" placement:
# STOP_ARM = ("iso-slow12-stop", "slow", 0.01, "--attack-blocks 1,2")


def branch(attack, alpha, extra):
    args = "--condition isolated"
    if attack:
        args += f" --attack {ATTACKS[attack]} --alpha {alpha}"
    return args + (f" {extra}" if extra else "")


def plans(workspace):
    out = {}
    for short in DOMAINS:
        for seed in SEEDS:
            init = f"lm-{short}-s{seed}-init"
            common = f"--seed {seed} --dumps"
            rare_base = (f"--stage branch --init {workspace}/lm-runs/{init} {common} "
                         f"--push-share {RARE_SHARE}")  # fmt: skip
            lines = []
            if seed != SEEDS[0]:  # seed 42's rare runs exist from the core grid
                lines += [(f"lm-{short}-s{seed}-{name}", f"{rare_base} {branch(a, alpha, extra)}")
                          for name, a, alpha, extra in RARE_ARMS]
            if short == "cd" and seed == SEEDS[0]:
                base2 = f"--stage branch --init {workspace}/lm-runs/{init} {common}"
                lines += [(f"lm-{short}-s{seed}-{name}", f"{base2} --condition isolated {extra}")
                          for name, extra in RARE2]  # fmt: skip
            if lines:
                out[f"focus-{short}-s{seed}"] = lines
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
