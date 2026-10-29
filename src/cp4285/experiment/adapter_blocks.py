"""Corrupted-stream study: a category language model with adapters on PESO's blocks.

The proposed variant in docs/CONTINUED-TRAINING.md. Each item is named by its PR 14
category. Users, blocks and splits follow upstream continual_data_v2.py, as in
`cp4285 blocks`: users with at least three events in a block train on all but the last
two, validate on the second-to-last and test on the last; a target needs a history.

Stage `init` trains the shared block-0 state Theta_0, saves it, and scores it on every
block (the frozen condition). Stage `branch` loads Theta_0, begins continuation once and
updates on blocks 1-4 under one condition:

- shared: one adapter keeps training, the plain continued-training learner;
- isolated: each block trains a new adapter on top of the frozen earlier ones.

A corruption (experiment/corruption.py) poisons the update stream: shill profiles, or
one of the catalogue's corruptions of real activity. `--push` names the promoted
category and is measured in clean runs too, as the reference. Corrupted users never
enter an evaluation set, and a relabelled category map is used for training rows only.
`--placement` chooses where the corruption enters: one update block (default), every
update block, or block 0 of the init stage, where it sits in the base weights. After
each block the run scores that block's test users, and after block 4 every earlier
block; an isolated attacked run also scores its final state with the attacked block's
adapter switched off (rollback), and `--rollback-block` moves that rollback later in
the stream to sweep its timing. Every cell reports the shared ranking metrics over all
categories (lower index wins ties) and, where tracked, the pushed category's mean rank,
top-1 and top-10 rate among genuine users whose own target is another category, split
by a trigger category on request, with the rival category tracked beside it.
"""

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .corruption import KINDS, apply_users, relabel_map
from .metrics import from_ranks
from .shilling import ATTACKS, PREFIX
from .utils import save_json

BLOCKS = range(5)
KS = (1, 5, 10)


def add_arguments(parser):
    parser.add_argument("--stage", choices=("init", "branch"), required=True)
    parser.add_argument("--data", required=True, help="Folder with PESO's *.all.block_{0..4}.json")
    parser.add_argument(
        "--categories", required=True, help="category_index.py index: item -> [<cat_k>]"
    )
    parser.add_argument("--names", required=True, help="category_stats.json with category names")
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--revision", default="", help="Recorded base-model revision")
    parser.add_argument("--output", required=True, help="New run folder")
    parser.add_argument("--init", help="branch: the init run folder holding theta0.pt")
    parser.add_argument("--condition", choices=("shared", "isolated"), default="shared")
    parser.add_argument("--attack", choices=("none", *ATTACKS), default="none")
    parser.add_argument(
        "--corruption",
        choices=KINDS,
        default="shill",
        help="How the stream is corrupted; shill uses --attack for the filler profile",
    )
    parser.add_argument(
        "--placement",
        choices=("block", "all", "init"),
        default="block",
        help="Corrupt the attacked block only, every update block, or block 0 of the init stage",
    )
    parser.add_argument("--push", type=int, help="Pushed category; also measured without an attack")
    parser.add_argument(
        "--push-share",
        type=float,
        help="Choose the pushed category: test-target share nearest this (blocks 1-4)",
    )
    parser.add_argument(
        "--rollback-after",
        action="store_true",
        help="isolated: delete the attacked block's adapter right after that block",
    )
    parser.add_argument(
        "--rollback-block",
        type=int,
        help="isolated: delete the attacked block's adapter after this later block instead",
    )
    parser.add_argument("--segment", type=int, help="Segment attack's filler category")
    parser.add_argument(
        "--alpha", type=float, default=0.03, help="Shill share of training examples"
    )
    parser.add_argument("--attack-block", type=int, default=1)
    parser.add_argument("--target-share", type=float, default=0.5, help="Shill events that push")
    parser.add_argument(
        "--rival", type=int, help="Demotion's mimicked category, tracked beside the pushed one"
    )
    parser.add_argument(
        "--rival-share",
        type=float,
        help="Choose the rival category like --push-share, excluding the pushed one",
    )
    parser.add_argument(
        "--trigger",
        type=int,
        help="Split the pushed category's measurement on this category in the user's history",
    )
    parser.add_argument(
        "--trigger-share",
        type=float,
        help="Choose the trigger category like --push-share, excluding the pushed one; a "
        "segment attack with no --segment then fills from it",
    )
    parser.add_argument(
        "--relabel-frac",
        type=float,
        help="relabel: share of a block's other items re-listed under the pushed category",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs0", type=int, default=1, help="Epochs on block 0")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per update block")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--continuation-lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=1000, help="Block-0 evaluation sample")
    parser.add_argument("--max-train0", type=int, default=0, help="Cap block-0 examples (0: all)")
    parser.add_argument(
        "--max-train", type=int, default=0, help="Cap update-block examples (0: all)"
    )
    parser.add_argument(
        "--max-test", type=int, default=0, help="Cap every block's test users (0: all)"
    )
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dumps", action="store_true", help="Write per-user ranks (run folder)")


def load(data_dir, categories_path, names_path):
    """Blocks, item -> 0-based category, and category names."""
    data_dir = Path(data_dir)
    blocks = [json.loads(next(data_dir.glob(f"*.all.block_{b}.json")).read_text()) for b in BLOCKS]
    index = json.loads(Path(categories_path).read_text())
    category = {
        item: int(v[0].removeprefix("<cat_").removesuffix(">")) for item, v in index.items()
    }
    labels = json.loads(Path(names_path).read_text())["categories"]
    names = [labels[f"<cat_{k}>"]["subcategory"] for k in range(len(labels))]
    return blocks, category, names


def split(users, category):
    """One block's training, validation and test examples over categories."""
    out = {"train": [], "valid": [], "test": [], "skipped_empty_history": 0}
    for user, data in users.items():
        if len(data["current_block"]) < 3:
            continue
        seq = [category[i] for i in data["previous_block"] + data["current_block"]]
        for t in range(len(data["previous_block"]), len(seq) - 2):
            if t:
                out["train"].append({"user": user, "history": seq[:t], "target": seq[t]})
            else:
                out["skipped_empty_history"] += 1
        if not user.startswith(PREFIX):  # shills train but are never evaluated
            out["valid"].append({"user": user, "history": seq[:-2], "target": seq[-2]})
            out["test"].append({"user": user, "history": seq[:-1], "target": seq[-1]})
    return out


def fit(learner, rows, epochs, batch, rng, clock=None):
    start, losses = time.perf_counter(), []
    for _ in range(epochs):
        order = rows[:]
        rng.shuffle(order)
        steps = [learner.train(order[i : i + batch]) for i in range(0, len(order), batch)]
        losses.append(float(np.mean(steps)))
    if clock is not None:
        clock["train_seconds"] = clock.get("train_seconds", 0.0) + time.perf_counter() - start
    return losses


def ranks_of(scores, columns):
    """1-based rank of each row's column; lower index wins ties."""
    picked = scores.gather(1, columns[:, None])
    index = torch.arange(scores.shape[1])[None, :]
    return 1 + (scores > picked).sum(1) + ((scores == picked) & (index < columns[:, None])).sum(1)


def evaluate(learner, rows, push=None, dumps=None, clock=None, rival=None, trigger=None):
    if any(r["user"].startswith(PREFIX) for r in rows):
        raise ValueError("Injected users must never be evaluated")
    start = time.perf_counter()
    scores = learner.score([r["history"] for r in rows])
    if clock is not None:
        clock["score_seconds"] = clock.get("score_seconds", 0.0) + time.perf_counter() - start
        clock["scored_users"] = clock.get("scored_users", 0) + len(rows)
    targets = torch.tensor([r["target"] for r in rows])
    ranks = ranks_of(scores, targets)
    cell = {"users": len(rows), **from_ranks(ranks.numpy(), KS)}

    def watch(name, column, mask):
        """Rank stats of one tracked category for the masked genuine users."""
        picked = ranks_of(scores, torch.full_like(targets, column))[mask].double()
        cell[name] = {"users": int(mask.sum()), "mean_rank": float(picked.mean()),
                      **{f"hit@{k}": float((picked <= k).double().mean()) for k in KS}}  # fmt: skip

    if push is not None:
        genuine = targets != push
        watch("push", push, genuine)
        if trigger is not None:  # a hidden trigger shows only in the segment it conditions on
            has = torch.tensor([trigger in r["history"] for r in rows]) & genuine
            if has.any():
                watch("push_trigger", push, has)
            if (~has & genuine).any():
                watch("push_other", push, ~has & genuine)
    if rival is not None:  # demotion's point is that the rival falls, so it is tracked too
        watch("rival", rival, targets != rival)
    if dumps is not None:
        dumps.parent.mkdir(parents=True, exist_ok=True)
        with dumps.open("x") as f:
            for r, rank in zip(rows, ranks.tolist()):
                f.write(json.dumps({"user": r["user"], "target": r["target"], "rank": rank}) + "\n")
    return cell


def corrupt_once(users, category, args, push, seed):
    """Apply the run's corruption once; returns (users, category for training, report).

    The shill, real-history, demotion and vandalism corruptions return edited users and
    the original category; relabel returns the users unchanged and a remapped category,
    so evaluation keeps the true labels while training records what the listings claim.
    """
    if args.corruption == "relabel":
        train_map, report = relabel_map(users, category, push, args.relabel_frac, seed)
        return users, train_map, report
    poisoned, report = apply_users(users, category, push, args.corruption, args.alpha, seed,
                                   attack=args.attack, segment=args.segment,
                                   target_share=args.target_share, rival=args.rival)  # fmt: skip
    return poisoned, category, report


def stream_for(users, category, block, args, push, attacked):
    """One update block's corrupted training inputs, per the corruption and placement.

    Placement `block` corrupts only the attacked block, `all` every update block; the
    init stage corrupts block 0 directly with `corrupt_once` under placement `init`.
    """
    here = attacked and (args.placement == "all"
                         or (args.placement == "block" and block == args.attack_block))  # fmt: skip
    if not here:
        return users, category, None
    return corrupt_once(users, category, args, push, args.seed + block)


def learner_for(args, names):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..models.category_lm import CategoryLMLearner

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return CategoryLMLearner(
        model, tokenizer, names, learning_rate=args.lr,
        continuation_learning_rate=args.continuation_lr, rank=args.rank, history=args.history,
        score_batch=args.score_batch, device=args.device, model_id=Path(args.model).name,
        revision=args.revision,
    )  # fmt: skip


def nearest_share(data, share, exclude=()):
    """The category whose share of blocks 1-4 test targets is nearest `share` -- the rule
    behind --push-share, so no watched category is picked by hand."""
    shares = Counter(r["target"] for d in data[1:] for r in d["test"])
    total = sum(shares.values())
    choices = [c for c in sorted(shares) if c not in exclude]
    return min(choices, key=lambda c: (abs(shares[c] / total - share), c))


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    attacked = args.corruption != "shill" or args.attack != "none"
    if args.corruption == "relabel" and not (args.relabel_frac and 0 < args.relabel_frac < 1):
        raise ValueError("The relabel corruption needs --relabel-frac in (0, 1)")
    if args.corruption == "demotion" and args.rival is None and args.rival_share is None:
        raise ValueError("The demotion corruption needs --rival or --rival-share")
    if args.trigger is not None and args.push is None and args.push_share is None:
        raise ValueError("--trigger splits the push measurement and needs --push or --push-share")
    if args.rollback_after and args.rollback_block is not None:
        raise ValueError("Use --rollback-after or --rollback-block, not both")
    drop_at = args.attack_block if args.rollback_after else args.rollback_block
    if drop_at is not None and (args.condition != "isolated" or not attacked):
        raise ValueError("A rollback needs isolated adapters and an attack")
    if drop_at is not None and args.placement == "all":
        raise ValueError("A rollback needs one attacked block; --placement all attacks every block")
    if drop_at is not None and args.placement == "init":
        raise ValueError("A corrupted starting state sits in the base weights and cannot be rolled back")
    if drop_at is not None and drop_at < args.attack_block:
        raise ValueError("--rollback-block cannot precede the attacked block")
    random.seed(args.seed)
    np.random.seed(args.seed)
    blocks, category, names = load(args.data, args.categories, args.names)
    data = [split(users, category) for users in blocks]
    tests = []
    for b, d in enumerate(data):
        caps = [c for c in (args.max_test, args.max_users if b == 0 else 0) if c]
        rows = d["test"]
        if caps and len(rows) > min(caps):
            rows = sorted(
                random.Random(args.seed + b).sample(rows, min(caps)), key=lambda r: r["user"]
            )
        tests.append(rows)
    if args.push_share is not None:  # the rule, so no domain's target is picked by hand
        args.push = nearest_share(data, args.push_share)
    if args.rival is None and args.rival_share is not None:
        args.rival = nearest_share(data, args.rival_share, exclude=(args.push,))
    if args.trigger is None and args.trigger_share is not None:
        args.trigger = nearest_share(data, args.trigger_share, exclude=(args.push,))
        if args.attack == "segment" and args.segment is None:
            args.segment = args.trigger  # a hidden-trigger run's segment is its trigger
    clock = {}
    learner = learner_for(args, names)
    dump = (lambda name: output / "dumps" / f"{name}.jsonl") if args.dumps else (lambda _: None)
    result = {"stage": args.stage, "config": {k: v for k, v in vars(args).items() if k != "func"},
              "categories": len(names), "push_name": None if args.push is None else names[args.push],
              "rival_name": None if args.rival is None else names[args.rival],
              "trigger_name": None if args.trigger is None else names[args.trigger],
              "blocks": [{k: (len(v) if isinstance(v, list) else v)
                                                    for k, v in d.items()} for d in data]}  # fmt: skip
    if args.stage == "init":
        rows = data[0]["train"]
        if args.placement == "init":  # the corruption sits in the base weights Theta_0
            if not attacked:
                raise ValueError("--placement init needs a corruption or attack")
            if args.corruption != "vandalism" and args.push is None:
                raise ValueError("This corruption needs --push, the promoted category")
            users, train_map, report = corrupt_once(blocks[0], category, args, args.push, args.seed)
            rows = split(users, train_map)["train"]
            report["block"] = 0
            result["attack"] = report
        if args.max_train0:
            rows = random.Random(args.seed).sample(rows, min(args.max_train0, len(rows)))
        result["losses"] = {
            "block0": fit(learner, rows, args.epochs0, args.batch, random.Random(args.seed), clock)
        }
        output.mkdir(parents=True)
        torch.save(learner.state(), output / "theta0.pt")
        result["cells"] = {
            f"state0_block{j}": evaluate(
                learner, tests[j], args.push, dump(f"state0_block{j}"), clock,
                rival=args.rival, trigger=args.trigger
            )
            for j in BLOCKS
        }
    else:
        if args.init is None:
            raise ValueError("A branch needs --init, the init run folder")
        init_metrics = Path(args.init) / "metrics.json"
        if drop_at is not None and init_metrics.exists():
            base_attack = json.loads(init_metrics.read_text()).get("attack")
            if base_attack and base_attack.get("block") == 0:
                raise ValueError(
                    "This init's starting data was corrupted; it sits in the base weights "
                    "and no adapter rollback can remove it"
                )  # fmt: skip
        learner.load_state(torch.load(Path(args.init) / "theta0.pt", weights_only=False))
        learner.begin_continuation()
        push = args.push  # measured whenever given, so clean runs give the reference rank
        if attacked and args.corruption != "vandalism" and push is None:
            raise ValueError("This corruption needs --push, the promoted category")
        result["cells"], result["losses"] = {}, {}
        for b in range(1, 5):
            users, train_map, report = stream_for(blocks[b], category, b, args, push, attacked)
            if report is not None:
                report["block"] = b
                if args.placement == "all":  # one report per attacked block
                    result.setdefault("attacks", {})[f"block{b}"] = report
                else:
                    result["attack"] = report
            rows = split(users, train_map)["train"]
            if args.max_train and len(rows) > args.max_train:  # smoke runs; keeps the share
                rows = random.Random(args.seed + b).sample(rows, args.max_train)
            if args.condition == "isolated":
                learner.add_stream(f"s{b}")
            result["losses"][f"block{b}"] = fit(learner, rows, args.epochs, args.batch,
                                                random.Random(args.seed + b), clock)  # fmt: skip
            result["cells"][f"state{b}_block{b}"] = evaluate(
                learner, tests[b], push, dump(f"state{b}_block{b}"), clock,
                rival=args.rival, trigger=args.trigger
            )
            if drop_at is not None and b == drop_at:
                # The rest of the stream builds without the poisoned adapter.
                learner.drop(f"s{args.attack_block}")
                result["cells"][f"dropped{b}_block{b}"] = evaluate(
                    learner, tests[b], push, dump(f"dropped{b}_block{b}"), clock,
                    rival=args.rival, trigger=args.trigger
                )
        for j in range(4):
            result["cells"][f"state4_block{j}"] = evaluate(
                learner, tests[j], push, dump(f"state4_block{j}"), clock,
                rival=args.rival, trigger=args.trigger
            )
        if args.condition == "isolated" and attacked and drop_at is None:
            attacked_adapter = f"s{args.attack_block}"
            learner.set_active([a for a in learner.adapters if a != attacked_adapter])
            result["rollback"] = {
                f"state4_block{j}": evaluate(
                    learner, tests[j], push, dump(f"rollback_block{j}"), clock,
                    rival=args.rival, trigger=args.trigger
                )
                for j in BLOCKS
            }
            learner.set_active(learner.adapters)
    result["timing"] = clock
    result["learner"] = learner.describe()
    result["torch_version"] = str(torch.__version__)
    save_json(output / "metrics.json", result)
    return result


SUMMARY = ("hit@1", "ndcg@10", "mrr@10")


def is_clean(config):
    """True when a run's stream is uncorrupted, under the old and new argument names."""
    return config.get("corruption", "shill") == "shill" and config.get("attack", "none") == "none"


def summarize(init_path, branch_paths):
    """Per branch: adaptation, RetentionLoss_A and backward transfer, and against the clean
    run of the same condition and pushed category, MixtureEffect_A and the push effect, plus
    rollback, rival and trigger effects.

    A is block 0 (the original task, R_A its test users). RetentionLoss_A is Theta_0's score
    on A minus the final state's (positive is worse); backward transfer is the mean over
    blocks 0-3 of the final state minus each block's own state. Push effects are attacked
    minus clean: a negative rank change means the watched category moved up. A dropped
    adapter's cell (`drop_effect`) is compared with the clean run's state at the same block:
    a positive rank change there is the rollback-timing sweep's over-correction.
    """
    init = json.loads(Path(init_path).read_text())
    runs = {Path(p).parent.name: json.loads(Path(p).read_text()) for p in branch_paths}
    clean = {(r["config"]["condition"], r["config"].get("push"), r["config"].get("rival")): r
             for r in runs.values() if is_clean(r["config"])}  # fmt: skip
    out = {"init": str(init_path), "branches": {}}
    for name, run in runs.items():
        cells, config = run["cells"], run["config"]
        drop_at = config.get("rollback_block")
        if drop_at is None and config.get("rollback_after"):
            drop_at = config["attack_block"]
        condition = config["condition"]
        if drop_at is not None:
            condition += "+drop" if drop_at == config["attack_block"] else f"+drop{drop_at}"
        row = {"condition": condition, "attack": config["attack"], "push": config.get("push"),
               "rival": config.get("rival"), "drop_at": drop_at,
               "alpha": config["alpha"] if not is_clean(config) else 0.0,
               "attack_block": config["attack_block"]}  # fmt: skip
        own = [cells[f"state{b}_block{b}"] for b in range(1, 5)]
        row["adaptation"] = {m: float(np.mean([c[m] for c in own])) for m in SUMMARY}
        row["retention_loss_A"] = {m: init["cells"]["state0_block0"][m] - cells["state4_block0"][m]
                                   for m in SUMMARY}  # fmt: skip
        own_of = {
            0: init["cells"]["state0_block0"],
            **{b: cells[f"state{b}_block{b}"] for b in (1, 2, 3)},
        }
        row["backward_transfer"] = {m: float(np.mean([cells[f"state4_block{j}"][m] - own_of[j][m]
                                                      for j in range(4)])) for m in SUMMARY}  # fmt: skip
        # Same condition and pushed category; a dropped run compares with the clean isolated one.
        base = clean.get((config["condition"], config.get("push"), config.get("rival")))
        if base is not None and base is not run:
            row["mixture_effect_A"] = {m: cells["state4_block0"][m] - base["cells"]["state4_block0"][m]
                                       for m in SUMMARY}  # fmt: skip
            row["push_vs_clean"] = {
                key: {"mean_rank": cells[key]["push"]["mean_rank"] - base["cells"][key]["push"]["mean_rank"],
                      "hit@10": cells[key]["push"]["hit@10"] - base["cells"][key]["push"]["hit@10"]}
                for key in cells if "push" in cells[key] and "push" in base["cells"].get(key, {})
            }  # fmt: skip
            row["rival_vs_clean"] = {
                key: {"mean_rank": cells[key]["rival"]["mean_rank"] - base["cells"][key]["rival"]["mean_rank"],
                      "hit@10": cells[key]["rival"]["hit@10"] - base["cells"][key]["rival"]["hit@10"]}
                for key in cells if "rival" in cells[key] and "rival" in base["cells"].get(key, {})
            }  # fmt: skip
            row["trigger_vs_clean"] = {
                key: {"mean_rank": cells[key]["mean_rank"] - base["cells"][key]["mean_rank"],
                      "users": cells[key]["users"]}
                for key in ("push_trigger", "push_other")
                if key in cells and key in base["cells"]
            }  # fmt: skip
            if drop_at is not None:
                dropped = f"dropped{drop_at}_block{drop_at}"
                same_state = f"state{drop_at}_block{drop_at}"
                if "push" in cells.get(dropped, {}) and "push" in base["cells"].get(same_state, {}):
                    row["drop_effect"] = {
                        "mean_rank": cells[dropped]["push"]["mean_rank"]
                        - base["cells"][same_state]["push"]["mean_rank"],
                        "hit@10": cells[dropped]["push"]["hit@10"]
                        - base["cells"][same_state]["push"]["hit@10"],
                    }  # fmt: skip
            if "rollback" in run:
                row["rollback_vs_clean"] = {
                    key: {"mean_rank": c["push"]["mean_rank"] - base["cells"][key]["push"]["mean_rank"],
                          "hit@1": c["hit@1"] - base["cells"][key]["hit@1"]}
                    for key, c in run["rollback"].items() if key in base["cells"]
                }  # fmt: skip
        if "attack" in run:
            row["shills"] = {k: run["attack"][k] for k in
                             ("shill_users", "alpha", "js_training_categories", "separability_auc")
                             if k in run["attack"]}  # fmt: skip
        out["branches"][name] = row
    return out


def markdown(summary):
    header = ("| Branch | Condition | Attack | Drop | Own-block Hit@1 | RetentionLoss_A Hit@1 "
              "| Backward transfer Hit@1 | MixtureEffect_A Hit@1 | Push rank change, attacked block "
              "| Push rank change, final state on block 0 | Shill AUC |")  # fmt: skip
    lines = [header, "| --- " * 11 + "|"]

    def fmt(value):
        return "" if value is None else f"{value:+.4f}"

    for name, row in summary["branches"].items():
        push = row.get("push_vs_clean", {})
        b = row["attack_block"]
        attacked = push.get(f"state{b}_block{b}", {}).get("mean_rank")
        final = push.get("state4_block0", {}).get("mean_rank")
        auc = row.get("shills", {}).get("separability_auc")
        lines.append(
            f"| {name} | {row['condition']} | {row['attack']} {row['alpha']:.2f} "
            f"| {row.get('drop_at') or ''} "
            f"| {row['adaptation']['hit@1']:.4f} | {fmt(row['retention_loss_A']['hit@1'])} "
            f"| {fmt(row['backward_transfer']['hit@1'])} "
            f"| {fmt(row.get('mixture_effect_A', {}).get('hit@1'))} | {fmt(attacked)} | {fmt(final)} "
            f"| {'' if auc is None else f'{auc:.3f}'} |"
        )
    return "\n".join(lines)
