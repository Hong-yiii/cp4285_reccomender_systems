"""Rebuild a GitHub weight release (release_bundle.py) as a pod_run.py run's checkpoints.

Mirrors notebooks/03_peso_colab.ipynb: every asset is checked against the release's
MANIFEST.json SHA-256, the shared tokenizer and config files go into each method's
folder, and block 0's modules_to_save is derived from Pretrain's block-0 weights. The
release holds blocks 0-3 for testing, so evaluate blocks 1-3 (the paper's average).

    PESO_WORKSPACE=... python release_ckpt.py <release url> <new run name>

Run with the PESO venv (safetensors and torch); the run lands where pod_run.py's
paths() expects it, for the released Instruments_cl_ori data.
"""

import hashlib
import http.client
import json
import shutil
import sys
import urllib.request

from pod_run import DATASET, paths

METHODS = ("pretrain", "lora", "peso")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, target, attempts=3):
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as f:
                shutil.copyfileobj(response, f, length=1 << 24)
            return
        except (OSError, http.client.HTTPException) as error:  # start the file again
            print(f"retrying {url}: {error}", flush=True)
    raise RuntimeError(f"Could not download {url}")


def main(release, run):
    from safetensors.torch import load_file, save_file

    if DATASET != "Instruments_cl_ori":
        raise ValueError("The release was trained on Instruments_cl_ori; unset PESO_DATASET")
    root, _, _ = paths(run, "pretrain")
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}; choose a new run name")
    assets = root.with_name(root.name + ".release")
    assets.mkdir(parents=True, exist_ok=True)  # resumable: verified files are kept
    download(f"{release}/MANIFEST.json", assets / "MANIFEST.json")
    manifest = json.loads((assets / "MANIFEST.json").read_text())
    for asset, meta in manifest["assets"].items():
        target = assets / asset
        if not (target.exists() and sha256(target) == meta["sha256"]):
            print(f"downloading {asset}", flush=True)
            download(f"{release}/{asset}", target)
            if sha256(target) != meta["sha256"]:
                raise ValueError(f"Checksum mismatch: {asset}")
    staging = root.with_name(root.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    out = {m: staging / paths(run, m)[1].relative_to(root) for m in METHODS}
    shared = [a for a, meta in manifest["assets"].items() if "/" not in meta["path"]]
    for asset, meta in manifest["assets"].items():
        method, _, rest = meta["path"].partition("/")
        if method in out:
            (out[method] / rest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(assets / asset, out[method] / rest)
    for method in METHODS:
        for asset in shared:
            shutil.copy2(assets / asset, out[method] / manifest["assets"][asset]["path"])
    block0 = load_file(out["pretrain"] / "block_0" / "model.safetensors")
    subset = {k: v for k, v in block0.items() if ".modules_to_save.default." in k}
    for path in manifest["derived"]:
        method, _, rest = path.partition("/")
        (out[method] / rest).parent.mkdir(parents=True, exist_ok=True)
        save_file(subset, out[method] / rest)
    record = {"release": release, "seed": manifest.get("seed"), "name": manifest.get("name")}
    (staging / "release.json").write_text(json.dumps(record, indent=2))
    staging.rename(root)  # appears complete or not at all
    shutil.rmtree(assets)
    print(json.dumps({"run": str(root), **record}))


if __name__ == "__main__":
    main(*sys.argv[1:])
