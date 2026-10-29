"""Item-level J-lens at state 4: a penalty variant against single LoRA (direction O8).

Reads jlens_items.py outputs (`<jlens_items>/<method>/state4/{lens.pt,readouts.jsonl}`, run
with --reference at single LoRA's state-0 lens). It reports the drift split by code and layer
relative to the shared block-0 lens, and per-code readout ranks on the same users, for old
blocks 0-3 and the newest block 4. Aggregates only.

    python variant_jlens_compare.py <jlens_items dir> <variant> <new report.json>
"""

import json
import statistics
import sys
from pathlib import Path

import torch

root, variant, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ref = torch.load(root / "lora" / "state0" / "lens.pt")["rows"]
report = {"variant": variant, "drift_state4": {}, "readouts_state4": {}}
for method in ("lora", variant):
    lens = torch.load(root / method / "state4" / "lens.pt")
    own, at_ref = lens["rows"], lens["rows_ref"]
    report["drift_state4"][method] = {
        f"c{c}": {
            f"l{l}": {
                "total_rel": float((own[c][l] - ref[c][l]).norm() / ref[c][l].norm()),
                "head_rel": float((own[c][l] - at_ref[c][l]).norm() / ref[c][l].norm()),
                "inner_rel": float((at_ref[c][l] - ref[c][l]).norm() / ref[c][l].norm()),
            }
            for l in own[c]
        }
        for c in range(4)
    }
rows = {}
for method in ("lora", variant):
    for line in (root / method / "state4" / "readouts.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(method, {})[r["block"], r["index"]] = r
shared = sorted(set(rows["lora"]) & set(rows[variant]))
for scope, keep in (("old_blocks_0_3", lambda b: b < 4), ("new_block_4", lambda b: b == 4)):
    keys = [k for k in shared if keep(k[0])]
    out_scope = {"users": len(keys)}
    for c in range(4):
        a = [rows["lora"][k][f"final_c{c}"] for k in keys]
        v = [rows[variant][k][f"final_c{c}"] for k in keys]
        out_scope[f"c{c}"] = {
            "mrr_lora": statistics.mean(1 / x for x in a),
            "mrr_variant": statistics.mean(1 / x for x in v),
            "share_variant_better": statistics.mean(y < x for x, y in zip(a, v)),
            "share_variant_worse": statistics.mean(y > x for x, y in zip(a, v)),
        }
    out_scope["loglik_final"] = {
        "lora": statistics.mean(rows["lora"][k]["loglik_final"] for k in keys),
        "variant": statistics.mean(rows[variant][k]["loglik_final"] for k in keys),
    }
    report["readouts_state4"][scope] = out_scope
with out.open("x") as f:
    json.dump(report, f, indent=2)
for m, d in report["drift_state4"].items():
    print(m)
    for c in range(4):
        print(
            f"  code{c + 1}: "
            + " | ".join(
                f"l{l} {d[f'c{c}'][f'l{l}']['total_rel']:.2f}/{d[f'c{c}'][f'l{l}']['head_rel']:.2f}/{d[f'c{c}'][f'l{l}']['inner_rel']:.2f}"
                for l in (0, 4, 8, 12, 14)
            )
        )
for scope, d in report["readouts_state4"].items():
    print(
        scope, "users", d["users"], "loglik", {k: round(v, 3) for k, v in d["loglik_final"].items()}
    )
    for c in range(4):
        x = d[f"c{c}"]
        print(
            f"  code{c + 1}: MRR lora {x['mrr_lora']:.3f} variant {x['mrr_variant']:.3f} | variant better {x['share_variant_better']:.0%} worse {x['share_variant_worse']:.0%}"
        )
