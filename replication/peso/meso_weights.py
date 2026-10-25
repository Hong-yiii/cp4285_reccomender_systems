"""Meso-level weight analysis of saved PESO block states (seed 42, CPU only).

M1, module drift map: for every LoRA module (16 layers x 7 projections) the change of its
weight update W = B A during each block b, ||W_b - W_{b-1}||_F, from the per-block
snapshots each block-4 checkpoint keeps (A_matrices.b, B_matrices.b). Summaries split the
drift by layer band and by attention vs MLP, and compare each method with single LoRA.

M2, output rows against popularity: for each first-code token <a_g> (the broad item
group), the output-layer row's change from block 0 to block 4 (relative drift and log
norm ratio), against the log ratio of the group's training share in block 4 vs block 0
(popularity_shift.py counts). Spearman correlations across the 256 groups.

    python meso_weights.py <ckpt root> <data dir> <index.json> <new report.json> \
        <method>=<run dir name> [...]

Run beside popularity_shift.py (its block counting is reused).
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from popularity_shift import load_blocks, share
from safetensors import safe_open
from scipy.stats import spearmanr
from transformers import AutoTokenizer

BANDS = {"layers 0-3": range(0, 4), "layers 4-7": range(4, 8), "layers 8-11": range(8, 12),
         "layers 12-15": range(12, 16)}  # fmt: skip
MODULE = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.A$")


def module_drift(path):
    """drift[b][(layer, kind, proj)] = ||B_b A_b - B_{b-1} A_{b-1}||_F for blocks 1-4."""
    drift = defaultdict(dict)
    with safe_open(path, "pt") as f:
        for key in f.keys():  # noqa: SIM118
            match = MODULE.match(key)
            if not match:
                continue
            layer, kind, proj = int(match[1]), match[2], match[3]
            stem = key[: -len(".A")]
            w = [
                f.get_tensor(f"{stem}.B_matrices.{b}").float() @ f.get_tensor(f"{stem}.A_matrices.{b}").float()
                for b in range(5)
            ]  # fmt: skip
            for b in range(1, 5):
                drift[b][(layer, kind, proj)] = float((w[b] - w[b - 1]).norm())
    return drift


def summarize(drift):
    out = {}
    for b, cells in drift.items():
        total = math.sqrt(sum(v**2 for v in cells.values()))
        sq = lambda keep: math.sqrt(sum(v**2 for k, v in cells.items() if keep(k)))  # noqa: E731
        out[f"block{b}"] = {
            "total": total,
            "by_band": {n: sq(lambda k, r=r: k[0] in r) for n, r in BANDS.items()},
            "by_kind": {kind: sq(lambda k, kind=kind: k[1] == kind) for kind in ("self_attn", "mlp")},
            "by_layer": [sq(lambda k, layer=layer: k[0] == layer) for layer in range(16)],
            "by_proj": {p: sq(lambda k, p=p: k[2] == p) for p in sorted({k[2] for k in cells})},
        }  # fmt: skip
    return out


def output_rows(block_dir, tokens):
    with safe_open(Path(block_dir) / "model.safetensors", "pt") as f:
        rows = f.get_slice("lm_head.modules_to_save.default.weight")
        return torch.stack([rows[i : i + 1][0].float() for i in tokens])


def main(ckpt_root, data_dir, index_path, output, *methods):
    runs = dict(m.split("=", 1) for m in methods)
    index = json.loads(Path(index_path).read_text())
    _, groups = load_blocks(data_dir, index)
    first = sorted({f"<a_{v[0][3:-1]}>" for v in index.values()}, key=lambda t: int(t[3:-1]))
    pop = [math.log(share(groups[4], g) / share(groups[0], g)) for g in first]
    report = {"methods": {}, "groups": len(first)}
    for method, run in runs.items():
        root = Path(ckpt_root) / run
        tokenizer = AutoTokenizer.from_pretrained(root / "block_4")
        ids = tokenizer.convert_tokens_to_ids(first)
        r0, r4 = output_rows(root / "block_0", ids), output_rows(root / "block_4", ids)
        rel = ((r4 - r0).norm(dim=1) / r0.norm(dim=1)).tolist()
        logn = (r4.norm(dim=1) / r0.norm(dim=1)).log().tolist()
        report["methods"][method] = {
            "module_drift": summarize(module_drift(root / "block_4" / "model.safetensors")),
            "output_rows_code1": {
                "rel_drift_median": sorted(rel)[len(rel) // 2],
                "spearman_rel_drift_vs_popularity": spearmanr(rel, pop)._asdict(),
                "spearman_log_norm_vs_popularity": spearmanr(logn, pop)._asdict(),
                "log_norm_rising_groups": sum(n for n, p in zip(logn, pop) if p > 0) / max(1, sum(p > 0 for p in pop)),
                "log_norm_falling_groups": sum(n for n, p in zip(logn, pop) if p < 0) / max(1, sum(p < 0 for p in pop)),
            },
        }  # fmt: skip
        print(method, json.dumps(report["methods"][method]["output_rows_code1"]), flush=True)
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(*sys.argv[1:])
