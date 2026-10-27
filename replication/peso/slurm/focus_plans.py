"""Plan files for the focused wave 1, continual-learning edition: the two arms that measure
forgetting rather than attack potency. The TSVs feed adapter_grid.sbatch exactly like
grid_plans.py's; every file is create-only.

    python focus_plans.py OUTPUT_DIR WORKSPACE   # WORKSPACE is the cluster's $W

Arms (27 new runs, about 5 GPU-hours, isolated layout only):
- rare-target replication, seeds 43-44, three domains (18 runs): do updates on rare
  categories persist longer and cost the pretraining users more? (frequency-dependent
  forgetting; seed 42's rare runs exist from the core grid)
- low-and-slow that stops (9 runs): 1% tampering in blocks 1-2 only, then clean — the
  forgetting curve of a sustained update once the pressure stops, against the one-off
  3% arms. Needs --attack-blocks (a runner flag to confirm on the cluster).

Attack names and flags are placeholders until wave 0's `grid_lattice.py --inspect`
confirms the cluster's config.attack strings and supported flags; regenerate before
submitting. The earlier dose-matched real-vs-fake arms (B) are cut: they measure attack
potency, not forgetting, and return only if the lattice shows update type changes the
washout rate.
"""

import sys
from pathlib import Path

DOMAINS = {"ins": "Instruments_cl_ori", "vg": "Video_Games_cl_0.6_0_5_3_0",
           "cd": "CDs_and_Vinyl_cl_0.6_0_5_3_0"}  # fmt: skip
SEEDS = (42, 43, 44)
PUSH_SHARE, RARE_SHARE = 0.013, 0.005
ATTACKS = {"bw": "bandwagon", "clone": "clone", "slow": "slow"}  # verify via --inspect
RARE_ARMS = [("iso-clean-rare", None, None, ""), ("iso-bw3-rare", "bw", 0.03, ""),
             ("iso-clone3-rare", "clone", 0.03, "")]  # fmt: skip
STOP_ARM = ("iso-slow12-stop", "slow", 0.01, "--attack-blocks 1,2")  # flag to confirm


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
            base = (f"--stage branch --init {workspace}/lm-runs/{init} {common} "
                    f"--push-share {PUSH_SHARE}")  # fmt: skip
            lines = [(init, f"--stage init {common} --push-share {PUSH_SHARE}")]
            lines.append((f"lm-{short}-s{seed}-{STOP_ARM[0]}",
                          f"{base} {branch(STOP_ARM[1], STOP_ARM[2], STOP_ARM[3])}"))
            if seed != SEEDS[0]:
                rare_base = (f"--stage branch --init {workspace}/lm-runs/{init} {common} "
                             f"--push-share {RARE_SHARE}")  # fmt: skip
                lines += [(f"lm-{short}-s{seed}-{name}", f"{rare_base} {branch(a, alpha, extra)}")
                          for name, a, alpha, extra in RARE_ARMS]
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
