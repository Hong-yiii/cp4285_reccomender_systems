"""Transfer matrix, flips and the PESO-vs-single-LoRA head-to-head from prediction dumps.

Reads `<predictions>/<method>/state{i}_block{j}.jsonl` written by peso_eval.py. State 0 is
the shared block-0 checkpoint, so PESO's state-0 rows are single LoRA's. Uses
cp4285.experiment.flips (or a sibling flips.py on machines without the package).

    python transfer_report.py <predictions> <user-groups.json> <new report.json>
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from cp4285.experiment import flips
except ImportError:  # e.g. on the GPU pod, with flips.py copied alongside
    import flips


def report(predictions, groups_path, ks=(10, 20)):
    predictions = Path(predictions)
    groups = json.loads(Path(groups_path).read_text())
    out = {"source": str(predictions), "methods": {}, "head_to_head": {}}
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {}
        for method in ("lora", "peso"):
            d = Path(tmp) / method
            shutil.copytree(predictions / method, d)
            for shared in (predictions / "lora").glob("state0_block*.jsonl"):
                if not (d / shared.name).exists():
                    shutil.copy2(shared, d / shared.name)
            dirs[method] = d
            out["methods"][method] = {f"k{k}": flips.report(d, k, groups) for k in ks}
        lora, peso = flips.load(dirs["lora"]), flips.load(dirs["peso"])
        for k in ks:
            per_pair = {}
            for (state, block), rows in sorted(peso.items()):
                if state == 0 or (state, block) not in lora:
                    continue
                # "before" = single LoRA, "after" = PESO: lost = LoRA hit, PESO miss.
                per_pair[f"state{state}_block{block}"] = flips.flips(
                    lora[state, block], rows, k, groups.get(str(block))
                )
            out["head_to_head"][f"k{k}"] = per_pair
    return out


if __name__ == "__main__":
    predictions, groups_path, output = sys.argv[1:4]
    result = report(predictions, groups_path)
    with Path(output).open("x") as f:
        json.dump(result, f, indent=2)
    for k, pairs in result["head_to_head"].items():
        print(k)
        for name, counts in pairs.items():
            a = counts["all"]
            print(
                f"  {name}: PESO-only hits {a['gained']}, LoRA-only hits {a['lost']}, p={a['mcnemar_p']}"
            )
