"""Commands are independent stages so dataset/model decisions stay reviewable."""

import argparse
import csv
import json
import platform
import tomllib
from pathlib import Path

from .experiment.data import CATEGORIES, DOMAINS, audit, download
from .experiment.neural_data import prepare
from .experiment.utils import millis, save_json

ROOT = Path(__file__).resolve().parents[2]


def config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for key in ("a", "b", "prepared"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    cfg["training"]["output"] = str(ROOT / cfg["training"]["output"])
    return cfg


def demo(output):
    from .experiment.pilot import run

    base = Path(output)
    if base.exists():
        raise FileExistsError(f"Demo directory exists: {base}")
    base.mkdir(parents=True)
    cfg = config(ROOT / "configs/pilot.toml")
    cfg["data"].update(max_users=24)
    cfg["training"].update(epochs=1, batch_size=8, cycles=2, updates_per_cycle=2)
    for domain in DOMAINS:
        with (base / f"{domain}.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
            for user in range(24):
                for step in range(52):
                    stamp = (
                        millis("2020-01-01") + step * 86400000
                        if step < 40
                        else millis("2021-01-02") + (step - 40) * 86400000
                    )
                    writer.writerow(
                        [
                            f"synthetic-user-{user}",
                            f"{domain}-item-{(user * 7 + step) % 97}",
                            5,
                            stamp,
                        ]
                    )
    a, b = [base / f"{domain}.csv" for domain in DOMAINS]
    save_json(base / "audit.json", audit(a, b, cfg["data"]["initial_cutoff"]))
    prepared = base / "prepared.json"
    cfg["data"].update(a=str(a.resolve()), b=str(b.resolve()), prepared=str(prepared.resolve()))
    cfg["training"]["output"] = str((base / "pilot").resolve())
    save_json(prepared, prepare(a, cfg["data"]))
    return run(prepared, cfg, base / "pilot", synthetic=True)


def meta_command(args):
    from .experiment import metadata

    category = getattr(args, "category", None)
    chosen = CATEGORIES if category == "all" else [category]
    if args.command == "meta-download":
        for category in chosen:
            # "all" resumes: finished categories are skipped, a downloaded file is reused.
            if args.category == "all" and metadata.items_path(args.directory, category).exists():
                print(f"Skipping existing {category}")
                continue
            if not metadata.raw_path(args.directory, category).exists():
                print(json.dumps(metadata.download(category, args.directory)))
            print(json.dumps(metadata.convert(category, args.directory)))
    elif args.command == "meta-audit":
        result = metadata.audit(args.directory, args.output, args.level)
        print(json.dumps({k: v["rows"] for k, v in result["domains"].items()}, indent=2))
    elif args.command == "meta-classify":
        for category in chosen:
            if not metadata.items_path(args.directory, category).exists():
                print(f"Skipping {category}: no converted metadata")
                continue
            if (
                args.category == "all"
                and (Path(args.output) / "metrics" / f"{category}.json").exists()
            ):
                print(f"Skipping classified {category}")
                continue
            result = metadata.classify(
                args.directory,
                args.output,
                category,
                level=args.level,
                min_items=args.min_items,
                sample=args.sample,
                precision=args.precision,
                seed=args.seed,
            )
            print(json.dumps(result))
    elif args.command == "meta-group":
        result = metadata.group(args.directory, args.output, args.level, args.ratings)
        save_json(Path(args.output) / "group.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "meta-publish":
        totals = json.loads((Path(args.output) / "group.json").read_text())["label_sources"]
        print(metadata.upload(args.output, args.repo, args.level, totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Installed versions and device availability")
    dl = commands.add_parser("download", help="Download official full 5-core ID files explicitly")
    dl.add_argument("--domain", choices=[*CATEGORIES, "both", "all"], required=True)
    dl.add_argument("--directory", default=str(ROOT / "data/raw"))
    au = commands.add_parser("audit", help="Audit both configured local files")
    au.add_argument("--output", default=str(ROOT / "reports/dataset-audit.json"))
    commands.add_parser("prepare", help="Prepare bounded Electronics pilot partitions")
    pi = commands.add_parser("pilot", help="Train A, verify reload and continue on A")
    pi.add_argument("--output")
    de = commands.add_parser("demo", help="Synthetic end-to-end software check, not Amazon results")
    de.add_argument("--output", default=str(ROOT / "runs/synthetic-smoke"))
    fl = commands.add_parser("flips", help="Flip counts and retention matrix from top-k dumps")
    fl.add_argument("--predictions", required=True, help="Directory of state*_block*.jsonl dumps")
    fl.add_argument("--output", required=True, help="New JSON report path")
    fl.add_argument("--k", type=int, default=10)
    fl.add_argument("--groups", help="JSON {test block: {user: label}} for sliced flip counts")
    meta = argparse.ArgumentParser(add_help=False)
    meta.add_argument("--directory", default=str(ROOT / "data/meta"))
    meta.add_argument("--output", default=str(ROOT / "data/meta/out"))
    meta.add_argument("--level", type=int, default=2, help="Path depth that defines a subcategory")
    md = commands.add_parser(
        "meta-download", parents=[meta], help="Download item metadata and keep grouping columns"
    )
    md.add_argument("--category", choices=[*CATEGORIES, "all"], required=True)
    commands.add_parser("meta-audit", parents=[meta], help="Category coverage and full tree")
    mc = commands.add_parser(
        "meta-classify", parents=[meta], help="Predict subcategories for unlabelled items"
    )
    mc.add_argument("--category", choices=[*CATEGORIES, "all"], required=True)
    mc.add_argument("--min-items", type=int, default=200, help="Smallest trainable class")
    mc.add_argument("--sample", type=int, default=200_000, help="Labelled items to embed")
    mc.add_argument("--precision", type=float, default=0.8, help="Held-out target precision")
    mc.add_argument("--seed", type=int, default=0)
    mg = commands.add_parser(
        "meta-group", parents=[meta], help="Partitioned item groups and density summary"
    )
    mg.add_argument("--ratings", default=str(ROOT / "data/raw"), help="5-core rating files")
    mp = commands.add_parser(
        "meta-publish", parents=[meta], help="Upload derived tables to a private HF dataset"
    )
    mp.add_argument("--repo", required=True, help="Hugging Face dataset id, e.g. user/name")
    bl = commands.add_parser(
        "blocks", help="SASRec, with or without a category channel, on PESO's blocks"
    )
    from .experiment.blocks import add_arguments as block_arguments

    block_arguments(bl)
    bc = commands.add_parser(
        "block-categories", help="PR 14-rule category labels for a PESO-format dataset"
    )
    from .experiment.block_categories import add_arguments as category_arguments

    category_arguments(bc)
    ab = commands.add_parser(
        "adapter-blocks",
        help="Category language model with adapters on PESO's blocks; optional shill stream",
    )
    from .experiment.adapter_blocks import add_arguments as adapter_arguments

    adapter_arguments(ab)
    asum = commands.add_parser("adapter-summary", help="Outcomes of `cp4285 adapter-blocks` runs")
    asum.add_argument("init", help="The init run's metrics.json")
    asum.add_argument("branches", nargs="+", help="Branch runs' metrics.json files")
    asum.add_argument("--json", help="Also save the summary here (create-only)")
    rp = commands.add_parser(
        "robustness-plan", help="Run matrix for the corrupted-stream attack catalogue"
    )
    from .experiment.robustness import add_arguments as robustness_arguments

    robustness_arguments(rp)
    ds = commands.add_parser(
        "dose-summary", help="Critical-mass dose curve from paired attacked and clean runs"
    )
    ds.add_argument("runs", nargs="+", help="Attacked run folders or metrics.json files")
    ds.add_argument("--clean", nargs="+", required=True, help="Clean reference runs")
    ds.add_argument("--cell", default="state1_block1", help="Scored cell to compare")
    ds.add_argument("--json", help="Also save the summary here (create-only)")
    ds.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap draws per run")
    ds.add_argument("--step", type=float, default=1.0, help="Rank-change threshold")
    ds.add_argument("--top5", type=float, default=0.5, help="Secondary top-5 threshold")
    bs = commands.add_parser("blocks-summary", help="Seeds and arms of `cp4285 blocks`")
    bs.add_argument("runs", nargs="+", help="metrics.json files")
    bs.add_argument("--json", help="Also save the summary here (create-only)")
    cl = commands.add_parser("classical", help="Classical comparator under B contamination")
    from .experiment.classical import add_arguments

    pre = argparse.ArgumentParser(add_help=False)  # config supplies the classical data defaults
    pre.add_argument("--config", default=str(ROOT / "configs/pilot.toml"))
    try:
        add_arguments(cl, ROOT, config(pre.parse_known_args()[0].config)["data"])
        args = parser.parse_args()
        cfg = config(args.config)
        d = cfg["data"]
        if args.command == "doctor":
            import duckdb
            import torch

            print(
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "torch": str(torch.__version__),
                        "duckdb": duckdb.__version__,
                        "cuda": torch.cuda.is_available(),
                        "mps": torch.backends.mps.is_available(),
                    },
                    indent=2,
                )
            )
        elif args.command == "download":
            chosen = {"both": DOMAINS, "all": CATEGORIES}.get(args.domain, [args.domain])
            for domain in chosen:
                if args.domain == "all" and (Path(args.directory) / f"{domain}.csv.gz").exists():
                    print(f"Skipping existing {domain}")
                    continue
                try:
                    print(json.dumps(download(domain, args.directory), indent=2))
                except ValueError as exc:
                    # Six small categories have no official 5-core file; report and continue.
                    if args.domain != "all":
                        raise
                    print(f"Skipping {domain}: {exc}")
        elif args.command.startswith("meta-"):
            meta_command(args)
        elif args.command == "audit":
            result = audit(d["a"], d["b"], d["initial_cutoff"], d["min_rating"])
            save_json(args.output, result)
            print(f"Audit saved: {args.output}")
        elif args.command == "prepare":
            result = prepare(d["a"], d)
            save_json(d["prepared"], result)
            print(json.dumps(result["metadata"], indent=2))
        elif args.command == "pilot":
            from .experiment.pilot import run

            output = args.output or cfg["training"]["output"]
            run(d["prepared"], cfg, output)
            print(f"Pilot saved: {output}")
        elif args.command == "flips":
            from .experiment.flips import report

            groups = json.loads(Path(args.groups).read_text()) if args.groups else None
            result = report(args.predictions, args.k, groups)
            save_json(args.output, result)
            print(json.dumps({"matrix": result["matrix"], "bwt": result["backward_transfer"]}))
            print(f"Flip report saved: {args.output}")
        elif args.command == "classical":
            from .experiment.classical import run as classical

            classical(args)
        elif args.command == "blocks":
            from .experiment.blocks import run as blocks

            result = blocks(args)
            print(f"Blocks run saved: {args.output} ({result['parameters']:,} parameters)")
        elif args.command == "block-categories":
            from .experiment.block_categories import run as block_categories

            stats = block_categories(args)
            print(
                json.dumps({k: stats[k] for k in ("source", "level", "label_source", "classifier")})
            )
            print(f"{len(stats['categories'])} categories; saved {args.output}")
        elif args.command == "adapter-blocks":
            from .experiment.adapter_blocks import run as adapter_blocks

            result = adapter_blocks(args)
            print(f"Adapter-blocks {args.stage} run saved: {args.output}")
        elif args.command == "adapter-summary":
            from .experiment.adapter_blocks import markdown as adapter_markdown
            from .experiment.adapter_blocks import summarize as adapter_summarize

            summary = adapter_summarize(args.init, args.branches)
            if args.json:
                save_json(args.json, summary)
            print(adapter_markdown(summary))
        elif args.command == "robustness-plan":
            from .experiment.robustness import plan

            plan(args)
        elif args.command == "dose-summary":
            from .experiment.dose import markdown as dose_markdown
            from .experiment.dose import save as save_dose
            from .experiment.dose import summary as dose_summary

            result = dose_summary(args.runs, args.clean, cell=args.cell,
                                  draws=args.bootstrap, step=args.step, top5=args.top5)  # fmt: skip
            if args.json:
                save_dose(result, args.json)
            print(dose_markdown(result))
        elif args.command == "blocks-summary":
            from .experiment.blocks import markdown, summarize

            summary = summarize(args.runs)
            if args.json:
                save_json(args.json, summary)
            print(markdown(summary))
        elif args.command == "demo":
            result = demo(args.output)
            print(
                json.dumps(
                    {
                        "scope": result["scope"],
                        "reload_difference": result["checkpoint_reload_max_difference"],
                        "output": args.output,
                    },
                    indent=2,
                )
            )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
