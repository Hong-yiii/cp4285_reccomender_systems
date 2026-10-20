"""Build a run's trained-weights bundle on Modal and upload it to a private Hugging Face repo.

Runs export_bundle.py against a finished run on the `cp4285-peso` volume and uploads
the result with the workspace's `huggingface` secret. The repo is created private.
Llama 3.2 Community Licence terms ride along: the base model's licence file and a
"Built with Llama" notice are added to the repo.

    modal run replication/peso/modal_bundle.py --seed 43 --repo <user>/<private-repo>
"""

from pathlib import Path

import modal

app = modal.App("cp4285-peso-bundle")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("safetensors==0.5.3", "huggingface_hub==0.32.3", "numpy==1.26.4")
    .add_local_file(Path(__file__).with_name("export_bundle.py"), "/root/export_bundle.py")
    .add_local_file(Path(__file__).with_name("release_bundle.py"), "/root/release_bundle.py")
)
volume = modal.Volume.from_name("cp4285-peso")

CARD = """---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B
---
# CP4285 PESO replication: trained weights (seed {seed})

Built with Llama. Trained LoRA adapters (every block state) and embedding/output-layer
copies from a replication of PESO (Yoo et al., ICLR 2026) on its released Instruments
blocks. Load on top of `meta-llama/Llama-3.2-1B` at revision 4e20de362430 with the
upstream PESO code at commit edbec5f; see the cp4285 course repository's
`docs/PESO-REPLICATION.md` and `notebooks/03_peso_colab.ipynb`. Distributed under the
Llama 3.2 Community License (LICENSE.txt). Private course-project artifact.
"""


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=16384,
    timeout=3 * 3600,
)
def bundle(seed: int, repo: str):
    import sys

    from huggingface_hub import HfApi, hf_hub_download

    sys.path.insert(0, "/root")
    from export_bundle import export

    out = Path(f"/root/bundle/seed{seed}")
    manifest = export(f"/vol/runs/seed{seed}", out)
    licence = hf_hub_download(
        "meta-llama/Llama-3.2-1B",
        "LICENSE.txt",
        revision="4e20de362430cd3b72f300e6b0f18e50e7166e08",
    )
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=licence, path_in_repo="LICENSE.txt", repo_id=repo)
    api.upload_file(
        path_or_fileobj=CARD.format(seed=seed).encode(), path_in_repo="README.md", repo_id=repo
    )
    api.upload_folder(folder_path=str(out), path_in_repo=f"seed{seed}", repo_id=repo)
    info = api.repo_info(repo)
    return {
        "repo": repo,
        "private": info.private,
        "sha": info.sha,
        "bytes": sum(v["bytes"] for v in manifest["files"].values()),
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=16384,
    timeout=3 * 3600,
)
def release(seed: int):
    """GitHub-release assets on the volume (bundle/<name>-seed<seed>), for upload with gh."""
    import sys

    from huggingface_hub import hf_hub_download

    sys.path.insert(0, "/root")
    from export_bundle import export
    from release_bundle import NAME, flatten

    out = Path(f"/vol/bundle/{NAME}-seed{seed}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    staged = Path(f"/root/bundle/seed{seed}")
    export(f"/vol/runs/seed{seed}", staged)
    licence = hf_hub_download(
        "meta-llama/Llama-3.2-1B",
        "LICENSE.txt",
        revision="4e20de362430cd3b72f300e6b0f18e50e7166e08",
    )
    manifest = flatten(staged, out, licence, seed)
    volume.commit()
    return {"path": str(out), "assets": len(manifest["assets"]),
            "bytes": sum(v["bytes"] for v in manifest["assets"].values())}  # fmt: skip


@app.local_entrypoint()
def main(seed: int = 43, repo: str = "", target: str = "hf"):
    if target != "github" and not repo:
        raise SystemExit("--repo <user>/<private-repo> is required for the Hugging Face upload")
    print(release.remote(seed) if target == "github" else bundle.remote(seed, repo))
