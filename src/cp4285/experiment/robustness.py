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
    if getattr(args, "stage", "A") == "strata":
        return strata_specs(args)
    if getattr(args, "stage", "A") == "dose":
        return dose_specs(args)
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


def strata_specs(args):
    """Account-stratified real-history arms: top and bottom of activity/reach, at a fixed
    account count and at a fixed event budget, plus the value-invariant k grid. Every arm
    uses the same alpha, so the event budget is matched by construction; the budget choice
    only controls how that fixed total is spread over the selected accounts."""
    metrics = [m for m in args.strata_metrics.split(",") if m]
    budgets = [b for b in args.strata_budgets.split(",") if b]
    ks = [int(k) for k in args.strata_k.split(",") if k]
    if any(m not in ("activity", "reach") for m in metrics):
        raise ValueError("--strata-metrics chooses from activity,reach")
    if any(b not in ("accounts", "events", "value") for b in budgets):
        raise ValueError("--strata-budgets chooses from accounts,events,value")
    if (args.strata_share is None) == (args.strata_accounts is None):
        raise ValueError("Give exactly one of --strata-share and --strata-accounts")
    if args.strata_share is not None and not 0 < args.strata_share < 1:
        raise ValueError("--strata-share must be strictly between 0 and 1")
    if args.strata_accounts is not None and args.strata_accounts < 1:
        raise ValueError("--strata-accounts must be at least 1")
    if "value" in budgets and any(k < 1 for k in ks):
        raise ValueError("--strata-k must be at least 1")
    condition = args.strata_condition
    prefix = "iso" if condition == "isolated" else "shared"
    common = ["--corruption", "real-history", "--alpha", str(args.alpha), "--eval-strata"]
    if args.strata_accounts is not None:
        common += ["--user-accounts", str(args.strata_accounts)]
    else:
        common += ["--user-share", str(args.strata_share)]
    stage = [("init", ["--stage", "init", "--eval-strata"]),
             (f"{prefix}-clean-strata", ["--stage", "branch", "--condition", condition,
                                          "--eval-strata"])]  # fmt: skip
    for metric in metrics:
        for side in ("top", "bottom"):
            for budget in budgets:
                if budget == "value":
                    for k in ks:
                        name = f"{prefix}-hijack-{metric}-{side}-valuek{k}"
                        extra = common + ["--user-stratum", metric, "--user-side", side,
                                          "--user-budget", "value", "--value-k", str(k)]  # fmt: skip
                        stage.append((name, ["--stage", "branch", "--condition", condition,
                                             *extra]))  # fmt: skip
                else:
                    name = f"{prefix}-hijack-{metric}-{side}-{budget}"
                    extra = common + ["--user-stratum", metric, "--user-side", side,
                                      "--user-budget", budget]  # fmt: skip
                    stage.append((name, ["--stage", "branch", "--condition", condition, *extra]))
    return stage


def dose_specs(args):
    """A threshold sweep: one clean reference and one attacked branch per dose and seed.
    Doses are shares of the attacked block's training examples, the catalogue's alpha."""
    doses = []
    for raw in args.doses.split(","):
        raw = raw.strip()
        if not raw:
            continue
        dose = float(raw)
        if not 0 < dose < 1:
            raise ValueError("--doses must be strictly between 0 and 1")
        doses.append(dose)
    doses = sorted(set(doses))
    if not doses:
        raise ValueError("--doses needs at least one value")
    if args.dose_corruption == "relabel":
        raise ValueError("A dose sweep is defined for user-level corruptions, not relabel")
    condition = args.dose_condition
    prefix = "iso" if condition == "isolated" else "shared"
    stage = [("init", ["--stage", "init"]),
             (f"{prefix}-clean-dose", ["--stage", "branch", "--condition", condition])]  # fmt: skip
    for dose in doses:
        bp = round(dose * 10_000)
        name = f"{prefix}-hijack-{bp}bp"
        extra = ["--corruption", args.dose_corruption, "--alpha", str(dose)]
        if args.dose_corruption == "shill":
            extra = ["--attack", "bandwagon", "--alpha", str(dose)]
        stage.append((name, ["--stage", "branch", "--condition", condition, *extra]))
    return stage


def add_arguments(parser):
    parser.add_argument("--data-root", required=True, help="Folder with the domain folders")
    parser.add_argument("--categories-root", required=True,
                        help="Folder with <domain>-cat2 index and category_stats.json files")
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--output-root", default="runs", help="Run folders go under here")
    parser.add_argument("--domains", default=",".join(DOMAIN_DIRS),
                        help="Comma-separated keys: " + ", ".join(sorted(DOMAIN_DIRS)))
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--stage", choices=("A", "B", "strata", "dose", "all"), default="A")
    parser.add_argument("--alpha", type=float, default=0.03, help="Core corruption share")
    parser.add_argument("--slow-alpha", type=float, default=0.01, help="Low-and-slow share")
    parser.add_argument("--push-share", type=float, default=0.013,
                        help="Pushed-category rule, resolved per domain by the runner")
    parser.add_argument("--rival-share", type=float, default=0.05, help="Rival-category rule")
    parser.add_argument("--trigger-share", type=float, default=0.1, help="Trigger-category rule")
    parser.add_argument("--relabel-frac", type=float, default=0.1, help="Relabelled item share")
    parser.add_argument("--strata-metrics", default="reach", help="activity,reach")
    parser.add_argument("--strata-share", type=float, default=0.1, help="Account share per side")
    parser.add_argument("--strata-accounts", type=int, help="Explicit accounts per side")
    parser.add_argument("--strata-budgets", default="accounts,events",
                        help="accounts,events,value")
    parser.add_argument("--strata-k", default="2,4", help="Value-invariant k grid")
    parser.add_argument("--strata-condition", choices=("shared", "isolated"), default="isolated")
    parser.add_argument("--doses", default="0.001,0.0025,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--dose-corruption", default="real-history",
                        help="User-level corruption for the dose sweep")
    parser.add_argument("--dose-condition", choices=("shared", "isolated"), default="isolated")
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
