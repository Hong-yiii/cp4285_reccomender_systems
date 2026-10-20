"""Flatten an export_bundle.py bundle into GitHub-release assets, storing block 0 once.

Release assets have no folders, so each file becomes `<method>.<block>.<file>`.
Single LoRA and PESO start from the pretraining stage's block-0 checkpoint, so their
block-0 embedding/output-layer subsets are not shipped: the notebook derives them
from `pretrain.block_0.model.safetensors` (MANIFEST lists them under "derived").
Tokenizer and config files are identical across methods and shipped once.

Llama 3.2 Community Licence terms for redistributed derivatives are added: the licence
text, a NOTICE with the required attribution, and a README naming the weights
"Llama-3.2-1B-PESO-replication" with "Built with Llama".
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

NAME = "Llama-3.2-1B-PESO-replication"
NOTICE = (
    "Llama 3.2 is licensed under the Llama 3.2 Community License, "
    "Copyright © Meta Platforms, Inc. All Rights Reserved.\n"
)
README = f"""# {NAME} (seed {{seed}})

Built with Llama.

Trained weights from a replication of PESO (Yoo et al., ICLR 2026, arXiv 2510.25093) on its
released Instruments blocks. The files hold only what training changed (LoRA adapters with
every block's snapshot, and the trained embedding/output-layer copies). Load them on top of
meta-llama/Llama-3.2-1B (revision 4e20de362430) with the upstream PESO code at commit
edbec5f, as notebooks/03_peso_colab.ipynb in the cp4285 course repository does.

Distributed under the Llama 3.2 Community License (LICENSE.txt) and Meta's Acceptable Use
Policy; see NOTICE.txt. Course-project research artifact; no warranty.
"""
SHARED = ("config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def same_block0(pretrain_model, subset):
    """True when a method's block-0 subset equals the pretraining block-0 tensors."""
    import torch
    from safetensors.torch import load_file

    full, part = load_file(pretrain_model), load_file(subset)
    return all(k in full and torch.equal(full[k], v) for k, v in part.items())


def flatten(bundle, out, licence, seed):
    bundle, out = Path(bundle), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    manifest = {"name": NAME, "seed": seed, "assets": {}, "derived": {}}

    def add(src, path):
        asset = path.replace("/", ".")
        shutil.copy2(src, staging / asset)
        manifest["assets"][asset] = {
            "path": path,
            "bytes": src.stat().st_size,
            "sha256": sha256(src),
        }

    for name in SHARED:
        digests = {sha256(bundle / m / name) for m in ("pretrain", "lora", "peso")}
        if len(digests) != 1:
            raise ValueError(f"{name} differs across methods; ship it per method instead")
        add(bundle / "pretrain" / name, name)
    for path in sorted(p for p in bundle.rglob("*.safetensors")):
        rel = path.relative_to(bundle).as_posix()
        if rel.split("/")[0] in ("lora", "peso") and "/block_0/" in f"/{rel}":
            if not same_block0(bundle / "pretrain/block_0/model.safetensors", path):
                raise ValueError(f"{rel} differs from the pretraining block 0; ship it instead")
            manifest["derived"][rel] = "modules_to_save keys of pretrain/block_0/model.safetensors"
            continue
        add(path, rel)
    shutil.copy2(licence, staging / "LICENSE.txt")
    (staging / "NOTICE.txt").write_text(NOTICE)
    (staging / "README.md").write_text(README.format(seed=seed))
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    staging.rename(out)
    return manifest


if __name__ == "__main__":
    bundle, out, licence, seed = sys.argv[1:5]
    result = flatten(bundle, out, licence, int(seed))
    print(json.dumps({k: v["bytes"] for k, v in result["assets"].items()}, indent=2))
