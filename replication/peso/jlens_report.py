"""Summarize the J-lens probe: readout quality, internal vs output change, and drift split.

Reads jlens.py outputs (`<jlens>/<method>/state{i}/{lens.pt,readouts.jsonl,head_drift.pt}`).
State 0 is the shared block-0 checkpoint; PESO's state 0 is single LoRA's.

    python jlens_report.py /workspace/peso-runs/runs/seed42/jlens <new report.json>

Ranks are of the target item's first semantic-ID code among the 256 first-code tokens.
"""

import json
import math
import statistics
import sys
from pathlib import Path

import torch


def load_readouts(root, method):
    rows = {}
    for state in range(5):
        path = root / method / f"state{state}" / "readouts.jsonl"
        if not path.exists() and method == "peso" and state == 0:
            path = root / "lora" / "state0" / "readouts.jsonl"  # shared block-0 checkpoint
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rows[state, r["block"], r["index"]] = r
    return rows


def spearman(x, y):
    """Spearman correlation with average ranks for ties (rank changes tie often)."""
    from scipy.stats import spearmanr

    rho = spearmanr(x, y).statistic
    return None if math.isnan(rho) else float(rho)  # NaN when a series is constant


def readout_quality(rows, lenses):
    out = {}
    for (state, block, _), r in rows.items():
        for lens in lenses:
            out.setdefault(f"state{state}", {}).setdefault(lens, []).append(r[lens])
    return {
        state: {
            lens: {
                "median_rank": statistics.median(v),
                "mrr": statistics.mean(1 / x for x in v),
                "top10": sum(x <= 10 for x in v) / len(v),
            }
            for lens, v in by_lens.items()
        }
        for state, by_lens in sorted(out.items())
    }


def internal_vs_output(rows, layer, last_state=4):
    """From each block's own state to the last state: change in mid-layer vs output rank."""
    d_final, d_mid, worse_out_held_mid = [], [], 0
    for (state, block, index), r in rows.items():
        if state != block or block == 0 or block >= last_state:
            continue
        later = rows.get((last_state, block, index))
        if later is None:
            continue
        df, dm = later["final"] - r["final"], later[f"jlens_l{layer}"] - r[f"jlens_l{layer}"]
        d_final.append(df)
        d_mid.append(dm)
        worse_out_held_mid += df > 0 and dm <= 0
    return {
        "users": len(d_final),
        "mean_final_rank_change": statistics.mean(d_final),
        "mean_midlayer_rank_change": statistics.mean(d_mid),
        "spearman_final_vs_mid": spearman(d_final, d_mid),
        "share_output_worse_but_mid_not": worse_out_held_mid / len(d_final),
    }


def drift_split(root, method):
    ref = torch.load(root / "lora" / "state0" / "lens.pt")
    out = {}
    for state in range(1, 5):
        lens_path = root / method / f"state{state}" / "lens.pt"
        head_path = root / method / f"state{state}" / "head_drift.pt"
        if not (lens_path.exists() and head_path.exists()):
            continue
        rows, head = torch.load(lens_path)["rows"], torch.load(head_path)["rows"]
        out[f"state{state}"] = {}
        for layer, r in rows.items():
            base = ref["rows"][layer]
            total = r - base
            head_part = head[layer]
            pathway = total - head_part  # U_0 (J_i - J_0)
            scale = base.norm().item()
            out[f"state{state}"][f"l{layer}"] = {
                "total_rel": total.norm().item() / scale,
                "head_rel": head_part.norm().item() / scale,
                "pathway_rel": pathway.norm().item() / scale,
            }
    return out


def main(root, output):
    root = Path(root)
    report = {"source": str(root), "methods": {}}
    for method in ("lora", "peso"):
        rows = load_readouts(root, method)
        layers = sorted({int(k[7:]) for r in rows.values() for k in r if k.startswith("jlens_l")})
        lenses = ["final", *[f"jlens_l{x}" for x in layers], *[f"logit_l{x}" for x in layers]]
        report["methods"][method] = {
            "readout_quality": readout_quality(rows, lenses),
            "internal_vs_output": {f"l{x}": internal_vs_output(rows, x) for x in layers},
            "drift_split": drift_split(root, method),
        }
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:3])["methods"], indent=1)[:4000])
