"""The corrupted-stream run matrix: what to launch for the attack catalogue.

    uv run cp4285 robustness-plan --data-root <dir> --categories-root <dir> \
        --model <model dir> --stage A

Prints one JSON object per run ({domain, seed, branch, argv}) whose `argv` is a ready
`cp4285 adapter-blocks` invocation, so a launcher only consumes lines. Stage A is the
core -- the rollback-timing sweep, low-and-slow and hijacked accounts; stage B adds
demotion, the hidden trigger, relabel, a poisoned starting state and vandalism. The
init and clean-reference runs repeat the existing grid's on purpose: the runner refuses
to overwrite an existing output folder, so a launcher can replay the plan safely and
skip lines whose folder is already there. The plan only names runs; nothing is
launched, downloaded or billed here.
"""

import json
from pathlib import Path

from .utils import save_json

# PESO-format domain folders behind the short names the grid's branch labels use.
DOMAIN_DIRS = {"ins": "Instruments_cl_ori", "vg": "Video_Games_cl_0.6_0_5_3_0",
               "cd": "CDs_and_Vinyl_cl_0.6_0_5_3_0"}  # fmt: skip


def branch_specs(args):
    """(branch name, extra argv) in launch order; init runs come first so branches can
    point --init at them. The existing grid's --rollback-after runs are the sweep's
    drop-after-1 point, so the plan emits only the later rollback timings. Stage B runs
    all carry the rival and trigger measurement rules and get matching clean-reference
    runs, so the summary compares like with like under one (condition, push, rival) key."""
    a, slow = str(args.alpha), str(args.slow_alpha)
    watch = ["--rival-share", str(args.rival_share), "--trigger-share", str(args.trigger_share)]

    def branch(name, condition, *extra):
        return (name, ["--stage", "branch", "--condition", condition, *extra])

    stage_a = [
        ("init", ["--stage", "init"]),
        branch("shared-clean", "shared"),
        branch("iso-clean", "isolated"),
        branch("iso-bw3-drop2", "isolated", "--attack", "bandwagon", "--alpha", a,
               "--rollback-block", "2"),
        branch("iso-bw3-drop3", "isolated", "--attack", "bandwagon", "--alpha", a,
               "--rollback-block", "3"),
        branch("iso-clone3-drop2", "isolated", "--attack", "clone", "--alpha", a,
               "--rollback-block", "2"),
        branch("iso-clone3-drop3", "isolated", "--attack", "clone", "--alpha", a,
               "--rollback-block", "3"),
        branch("shared-slow1", "shared", "--attack", "bandwagon", "--alpha", slow,
               "--placement", "all"),
        branch("iso-slow1", "isolated", "--attack", "bandwagon", "--alpha", slow,
               "--placement", "all"),
        branch("shared-hijack3", "shared", "--corruption", "real-history", "--alpha", a),
        branch("iso-hijack3", "isolated", "--corruption", "real-history", "--alpha", a),
    ]
    stage_b = [
        branch("shared-clean-watch", "shared", *watch),
        branch("iso-clean-watch", "isolated", *watch),
        branch("shared-demote3", "shared", "--corruption", "demotion", "--alpha", a, *watch),
        branch("iso-demote3", "isolated", "--corruption", "demotion", "--alpha", a, *watch),
        branch("shared-trig3", "shared", "--attack", "segment", "--alpha", a, *watch),
        branch("iso-trig3", "isolated", "--attack", "segment", "--alpha", a, *watch),
        branch("shared-relabel", "shared", "--corruption", "relabel",
               "--relabel-frac", str(args.relabel_frac), *watch),
        branch("iso-relabel", "isolated", "--corruption", "relabel",
               "--relabel-frac", str(args.relabel_frac), *watch),
        branch("shared-vandal3", "shared", "--corruption", "vandalism", "--alpha", a, *watch),
        branch("iso-vandal3", "isolated", "--corruption", "vandalism", "--alpha", a, *watch),
        ("init-poison", ["--stage", "init", "--placement", "init", "--attack", "bandwagon",
                         "--alpha", a, *watch]),
        branch("shared-poisonbase", "shared", *watch),
        branch("iso-poisonbase", "isolated", *watch),
    ]
    return stage_a + (stage_b if args.stage in ("B", "all") else [])


def add_arguments(parser):
    parser.add_argument("--data-root", required=True, help="Folder with the domain folders")
    parser.add_argument("--categories-root", required=True,
                        help="Folder with <domain>-cat2 index and category_stats.json files")
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--output-root", default="runs", help="Run folders go under here")
    parser.add_argument("--domains", default=",".join(DOMAIN_DIRS),
                        help="Comma-separated keys: " + ", ".join(sorted(DOMAIN_DIRS)))
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--stage", choices=("A", "B", "all"), default="A")
    parser.add_argument("--alpha", type=float, default=0.03, help="Core corruption share")
    parser.add_argument("--slow-alpha", type=float, default=0.01, help="Low-and-slow share")
    parser.add_argument("--push-share", type=float, default=0.013,
                        help="Pushed-category rule, resolved per domain by the runner")
    parser.add_argument("--rival-share", type=float, default=0.05, help="Rival-category rule")
    parser.add_argument("--trigger-share", type=float, default=0.1, help="Trigger-category rule")
    parser.add_argument("--relabel-frac", type=float, default=0.1, help="Relabelled item share")
    parser.add_argument("--output", help="Also save the plan here (create-only)")


def plan(args):
    """One JSON object per run, in launch order; init-poison branches reference its folder."""
    roots = Path(args.data_root), Path(args.categories_root), Path(args.output_root)
    seeds = [int(s) for s in args.seeds.split(",")]
    keys = [k for k in args.domains.split(",") if k]
    unknown = [k for k in keys if k not in DOMAIN_DIRS]
    if unknown:
        raise ValueError(f"Unknown domain keys {unknown}; choose from {sorted(DOMAIN_DIRS)}")
    runs = []
    for key in keys:
        domain = DOMAIN_DIRS[key]
        data = roots[0] / domain
        cat_dir = roots[1] / f"{domain}-cat2"
        shared = ["--data", str(data), "--categories", str(cat_dir / f"{domain}.index.cat2.json"),
                  "--names", str(cat_dir / "category_stats.json"), "--model", args.model,
                  "--push-share", str(args.push_share)]  # fmt: skip
        for seed in seeds:
            poisoned = init_output = None
            for branch, extra in branch_specs(args):
                output = roots[2] / f"lm-{key}-s{seed}-{branch}"
                argv = shared + ["--seed", str(seed)] + extra
                if branch == "init":
                    init_output = output
                if branch == "init-poison":
                    poisoned = output
                if branch.endswith("poisonbase"):
                    if poisoned is None:
                        raise ValueError("The poisoned-base branches need the init-poison run")
                    argv += ["--init", str(poisoned)]
                elif extra[0] == "--stage" and extra[1] == "branch":
                    if init_output is None:
                        raise ValueError("Branch runs need the init run; emit the init spec first")
                    argv += ["--init", str(init_output)]
                runs.append({"domain": key, "seed": seed, "branch": branch,
                             "argv": argv + ["--output", str(output)]})  # fmt: skip
    if args.output:
        save_json(args.output, runs)
    for run_ in runs:
        print(json.dumps(run_))
    return runs
