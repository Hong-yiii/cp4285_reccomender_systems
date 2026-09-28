"""Commands are independent stages so dataset/model decisions stay reviewable."""

import argparse
import csv
import json
import platform
import tomllib
from pathlib import Path

from .data import DOMAINS, audit, download, millis, prepare, save_json

ROOT = Path(__file__).resolve().parents[2]


def config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for key in ("a", "b", "prepared"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    cfg["training"]["output"] = str(ROOT / cfg["training"]["output"])
    return cfg


def demo(output):
    from .pilot import run

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Installed versions and device availability")
    dl = commands.add_parser("download", help="Download official full 5-core ID files explicitly")
    dl.add_argument("--domain", choices=[*DOMAINS, "both"], required=True)
    dl.add_argument("--directory", default=str(ROOT / "data/raw"))
    au = commands.add_parser("audit", help="Audit both configured local files")
    au.add_argument("--output", default=str(ROOT / "reports/dataset-audit.json"))
    commands.add_parser("prepare", help="Prepare bounded Electronics pilot partitions")
    pi = commands.add_parser("pilot", help="Train A, verify reload and continue on A")
    pi.add_argument("--output")
    de = commands.add_parser("demo", help="Synthetic end-to-end software check, not Amazon results")
    de.add_argument("--output", default=str(ROOT / "runs/synthetic-smoke"))
    cl = commands.add_parser("classical", help="Classical comparator under B contamination")
    from .classical.experiments import add_arguments

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
            for domain in DOMAINS if args.domain == "both" else [args.domain]:
                print(json.dumps(download(domain, args.directory), indent=2))
        elif args.command == "audit":
            result = audit(d["a"], d["b"], d["initial_cutoff"], d["min_rating"])
            save_json(args.output, result)
            print(f"Audit saved: {args.output}")
        elif args.command == "prepare":
            result = prepare(d["a"], d)
            save_json(d["prepared"], result)
            print(json.dumps(result["metadata"], indent=2))
        elif args.command == "pilot":
            from .pilot import run

            output = args.output or cfg["training"]["output"]
            run(d["prepared"], cfg, output)
            print(f"Pilot saved: {output}")
        elif args.command == "classical":
            from .classical.experiments import run as classical

            classical(args)
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
