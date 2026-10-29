"""Execute notebooks/03_peso_colab.ipynb as a fresh user would, on a Colab-like GPU.

A clean container (Python 3.12, no project code, no caches) runs the notebook file
itself with nbconvert, magics included. The only credential is a Hugging Face token
(HF_TOKEN from the workspace secret), as in Colab's Secrets panel. The notebook clones
its REPO at BRANCH from GitHub, so push the branch (and publish its release) first.

The container only executes and returns raw data; all parsing happens locally, so a
parsing bug cannot lose a finished run. Cell errors are kept in the executed notebook
(--allow-errors) rather than discarding it.

    modal run replication/peso/modal_colab_test.py --gpu A100-40GB --save <new .ipynb>
"""

import json
import subprocess
from pathlib import Path

import modal

# Inside the container this file is /root/modal_colab_test.py, where parents[2] does not
# exist; .parent stops at "/". Modal reads the local file only when deploying.
NOTEBOOK = Path(__file__).resolve().parent.parent.parent / "notebooks" / "03_peso_colab.ipynb"
app = modal.App("cp4285-colab-test")
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("nbconvert==7.16.6", "nbclient==0.10.2", "ipykernel==6.29.5")
    .add_local_file(NOTEBOOK, "/test/03_peso_colab.ipynb")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=4 * 3600,
)
def run_notebook():
    import os

    # The workspace secret may use another key name; the notebook reads HF_TOKEN.
    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    token = next((os.environ[n] for n in names if os.environ.get(n)), None)
    if token is None:
        raise RuntimeError(f"Secret 'huggingface' defines none of {names}")
    cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors",
           "--ExecutePreprocessor.timeout=-1", "--output", "executed.ipynb",
           "/test/03_peso_colab.ipynb"]  # fmt: skip
    env = {**os.environ, "HF_TOKEN": token}
    proc = subprocess.run(cmd, cwd="/test", env=env, capture_output=True, text=True, check=False)
    print(proc.stderr[-3000:], flush=True)  # nbconvert's own log, for the container log
    executed = Path("/test/executed.ipynb")
    return {
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-3000:],
        "notebook": executed.read_text() if executed.exists() else None,
    }


def text(output):
    """A stream output's text; nbformat stores it as a string or a list of lines."""
    value = output.get("text", "")
    return value if isinstance(value, str) else "".join(value)


def summary(notebook):
    """The comparison cell's printed table and every cell error, from an executed notebook."""
    cells = [c for c in json.loads(notebook)["cells"] if c["cell_type"] == "code"]
    compare = next(c for c in cells if "RECORDED = " in "".join(c["source"]))
    errors = [
        f"cell {i}: {o['ename']}: {o['evalue']}"
        for i, c in enumerate(cells)
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    return "".join(text(o) for o in compare.get("outputs", [])), errors


@app.local_entrypoint()
def main(gpu: str = "A100-40GB", save: str = ""):
    result = run_notebook.with_options(gpu=gpu).remote()
    print("returncode:", result["returncode"])
    if not result["notebook"]:
        print(result["stderr_tail"])
        return
    if save:
        with Path(save).open("x") as f:
            f.write(result["notebook"])
    table, errors = summary(result["notebook"])
    print(table or "(comparison cell printed nothing)")
    print("cell errors:", errors or "none")
