"""Category-target PESO across seeds: own-block scores, baselines and backward transfer.

Reads one category_report.py report per seed, each built from that seed's lora and peso
matrix dumps (state0 is the shared block-0 checkpoint, which is Pretrain):

    python replication/peso/category_summary.py <new summary.json> \
        replication/peso/results/category/seed4{2,3,4}/category_report.json

Own-block cells pair state b with block b. The paper averages blocks 1-3; blocks 1-4
are reported as well. Backward transfer is the mean over blocks 0-3 of state 4 minus
the block's own state. Baselines are scored on the same users as the model.
Standard deviation is across seeds (n - 1).
"""

import json
import statistics
import sys
from pathlib import Path

METRICS = ("hit@1", "hit@5", "hit@10", "ndcg@10", "mrr@10")
AVERAGES = {"blocks 1-3 (paper)": (1, 2, 3), "blocks 1-4": (1, 2, 3, 4)}
BASELINES = ("popular", "history")


def spread(values):
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "seeds": values,
    }


def own(report, method, block, part="all"):
    """Pretrain is single LoRA's state 0; the others pair state b with block b."""
    source, state = ("lora", 0) if method == "pretrain" else (method, block)
    return report["methods"][source][f"state{state}_block{block}"][part]


def summarize(reports):
    out = {"averages": {}, "paired_peso_minus_lora": {}, "backward_transfer": {}}
    for label, blocks in AVERAGES.items():
        rows = {}
        for method in ("pretrain", "lora", "peso", *BASELINES):
            values = {m: [] for m in METRICS}
            for r in reports:
                cells = [
                    own(r, "lora", b, method) if method in BASELINES else own(r, method, b)
                    for b in blocks
                ]
                for m in METRICS:
                    values[m].append(statistics.mean(c[m] for c in cells))
            rows[method] = {m: spread(v) for m, v in values.items()}
        out["averages"][label] = rows
        out["paired_peso_minus_lora"][label] = {
            m: spread([p - q for p, q in zip(rows["peso"][m]["seeds"], rows["lora"][m]["seeds"])])
            for m in METRICS
        }
    for method in ("lora", "peso"):
        values = {m: [] for m in METRICS}
        for r in reports:
            cells = r["methods"][method]
            for m in METRICS:
                drops = [
                    cells[f"state4_block{j}"]["all"][m] - own(r, method, j)[m]
                    if j
                    else cells["state4_block0"]["all"][m] - own(r, "pretrain", 0)[m]
                    for j in range(4)
                ]
                values[m].append(statistics.mean(drops))
        out["backward_transfer"][method] = {m: spread(v) for m, v in values.items()}
    return out


def markdown(summary):
    lines = []
    for label, rows in summary["averages"].items():
        lines += [f"{label}:", "", "| Method | " + " | ".join(METRICS) + " |"]
        lines.append("| --- |" + " --- |" * len(METRICS))
        for method, cells in rows.items():
            fmt = [f"{c['mean']:.4f} ± {c['sd']:.4f}" if c["sd"] is not None else f"{c['mean']:.4f}"
                   for c in cells.values()]  # fmt: skip
            lines.append(f"| {method} | " + " | ".join(fmt) + " |")
        diff = summary["paired_peso_minus_lora"][label]
        lines.append(
            "| PESO − LoRA (paired) | "
            + " | ".join(f"{diff[m]['mean']:+.4f}" for m in METRICS)
            + " |"
        )
        lines.append("")
    lines += ["Backward transfer (state 4 minus own state, blocks 0-3):", ""]
    lines.append("| Method | " + " | ".join(METRICS) + " |")
    lines.append("| --- |" + " --- |" * len(METRICS))
    for method, cells in summary["backward_transfer"].items():
        lines.append(
            f"| {method} | " + " | ".join(f"{c['mean']:+.4f}" for c in cells.values()) + " |"
        )
    return "\n".join(lines)


def main(output, *paths):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reports = [json.loads(Path(p).read_text()) for p in paths]
    summary = {"reports": list(paths), **summarize(reports)}
    with output.open("x") as f:
        json.dump(summary, f, indent=2)
    print(markdown(summary))


if __name__ == "__main__":
    main(*sys.argv[1:])
