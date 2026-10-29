"""Compare replicated PESO results with the paper's Table 2 (Instruments).

Reads upstream continual_test.py result files, laid out as <run>/results/<tag>.json;
the committed copies are under replication/peso/results/:

    python replication/peso/summarize.py replication/peso/results/seed4{2,3,4}

The paper averages D2-D4 (code blocks 1-3); blocks 1-4 are reported as well because
its text also describes stages 2-5. Standard deviation is across seeds (n - 1).
"""

import json
import statistics
import sys
from pathlib import Path

TAGS = {
    "pretrain": "QUICK_ws20.msl-1.temp0.8.sflora_pretrain.blocks5.json",
    "lora": "QUICK_ws20.msl-1.temp0.8.sflora_cl_0.0.blocks5.json",
    "peso": "QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5.json",
}
PAPER = {  # arXiv 2510.25093v3, Table 2, Instruments (PDF p. 8)
    "pretrain": {"hit@5": 0.0166, "hit@10": 0.0216, "ndcg@5": 0.0115, "ndcg@10": 0.0131},
    "lora": {"hit@5": 0.0181, "hit@10": 0.0253, "ndcg@5": 0.0127, "ndcg@10": 0.0150},
    "peso": {"hit@5": 0.0193, "hit@10": 0.0268, "ndcg@5": 0.0138, "ndcg@10": 0.0162},
}
METRICS = ("hit@5", "hit@10", "ndcg@5", "ndcg@10")
AVERAGES = {"blocks 1-3 (paper)": (1, 2, 3), "blocks 1-4": (1, 2, 3, 4)}


def per_block(path):
    blocks = json.loads(Path(path).read_text())["all_block_results"]
    return {int(k.split("_")[1]): v["all_prompt_results"][0] for k, v in blocks.items()}


def runs_blocks(runs):
    """{method: [per-block metrics for each run]} in the order the runs are given."""
    return {m: [per_block(Path(run) / "results" / tag) for run in runs] for m, tag in TAGS.items()}


def spread(values):
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize(runs):
    table = {}
    for method, seeds in runs_blocks(runs).items():
        table[method] = {"per_seed_blocks": seeds, "averages": {}}
        for label, blocks in AVERAGES.items():
            values = {m: [statistics.mean(s[b][m] for b in blocks) for s in seeds] for m in METRICS}
            table[method]["averages"][label] = {
                m: {
                    **spread(v),
                    "seeds": v,
                    "paper": PAPER[method][m],
                    "relative_to_paper": statistics.mean(v) / PAPER[method][m] - 1,
                }
                for m, v in values.items()
            }
    return table


def paired(runs, a="lora", b="peso", blocks=(1, 2, 3)):
    """Per-run b - a on the block average; runs share each seed's block-0 checkpoint."""
    seeds = runs_blocks(runs)
    out = {}
    for m in METRICS:
        diffs = [
            statistics.mean(sb[i][m] for i in blocks) - statistics.mean(sa[i][m] for i in blocks)
            for sa, sb in zip(seeds[a], seeds[b])
        ]
        out[m] = {**spread(diffs), "per_run": diffs, "paper": PAPER[b][m] - PAPER[a][m]}
    return out


def markdown(table):
    lines = []
    for label in AVERAGES:
        runs = len(table["lora"]["per_seed_blocks"])
        lines += [f"\n{label}, {runs} seed(s): mean ± sd (paper)"]
        lines += ["| Method | " + " | ".join(METRICS) + " |", "| --- |" + " --- |" * len(METRICS)]
        for method in TAGS:
            cells = []
            for m in METRICS:
                c = table[method]["averages"][label][m]
                sd = f" ± {c['sd']:.4f}" if c["sd"] is not None else ""
                cells.append(f"{c['mean']:.4f}{sd} ({c['paper']:.4f})")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(runs):
    table = summarize(runs)
    print(markdown(table))
    return table


if __name__ == "__main__":
    args = sys.argv[1:]
    output = None
    if "--json" in args:
        output = Path(args[args.index("--json") + 1])
        args = [a for a in args if a not in ("--json", str(output))]
    result = main(args)
    if output:
        with output.open("x") as f:
            json.dump(result, f, indent=2)
