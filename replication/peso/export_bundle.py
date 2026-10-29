"""Export the trained parts of a PESO replication run as a small, loadable bundle.

Each saved block checkpoint repeats the full Llama-3.2-1B weights. The base model is
frozen and downloadable from Hugging Face (licence accepted per user), so the bundle
keeps only what training changed: the LoRA adapters, including every block's saved
snapshot, and the trained embedding/output-layer copies (`modules_to_save`). Upstream
`load_continual_model` loads with strict=False on top of a fresh base model, so these
partial files restore every block state exactly. Unused `original_module` copies and
frozen `orig` base weights are dropped.

    python export_bundle.py /workspace/peso-runs/runs/seed42 /workspace/peso-bundle/seed42

The bundle is a derivative of Llama 3.2 weights: distribute it only under the Llama 3.2
Community Licence (include the licence, "Built with Llama"), privately, e.g. to the team.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

DATASET_DIR = Path("ckpt/Instruments_cl_ori/lama3-1b_edim32_beta0.0/data_v2")
RUNS = {
    "pretrain": "QUICK_ws20.msl-1.temp0.8.sflora_pretrain.blocks5",
    "lora": "QUICK_ws20.msl-1.temp0.8.sflora_cl_0.0.blocks5",
    "peso": "QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5",
}
TOKENIZER_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def trained(key):
    """Trainable or saved-adapter tensors; frozen base copies are dropped."""
    if ".orig." in key or ".original_module." in key:
        return False
    return ".modules_to_save." in key or key.endswith((".A", ".B")) or "_matrices." in key


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def export(run, out):
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {out}")
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    manifest = {"source_run": str(run), "files": {}}
    for method, tag in RUNS.items():
        src, dst = run / DATASET_DIR / tag, staging / method
        dst.mkdir(parents=True)
        for name in TOKENIZER_FILES:
            shutil.copy2(src / name, dst / name)
        last = 0 if method == "pretrain" else 4
        state = {
            k: v
            for k, v in load_file(src / f"block_{last}" / "model.safetensors").items()
            if trained(k)
        }
        (dst / f"block_{last}").mkdir()
        save_file(state, dst / f"block_{last}" / "model.safetensors")
        for block in range(last):  # earlier blocks: their embedding/output-layer subsets
            (dst / f"block_{block}").mkdir()
            shutil.copy2(
                src / f"block_{block}" / "modules_to_save.safetensors", dst / f"block_{block}"
            )
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(staging))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    staging.rename(out)
    return manifest


if __name__ == "__main__":
    result = export(*sys.argv[1:3])
    print(json.dumps({k: v["bytes"] for k, v in result["files"].items()}, indent=2))
