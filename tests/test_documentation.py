"""Execute the public recipes without Jupyter or hidden interpreter state."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("artifact", ["guide", "notebook"])
def test_documented_lifecycles_run_in_a_fresh_process(artifact):
    if artifact == "guide":
        text = (ROOT / "docs/PACKAGES-AND-BUILDING.md").read_text()
        cells = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    else:
        notebook = json.loads((ROOT / "notebooks/01_shared_lifecycle.ipynb").read_text())
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert all(c["outputs"] == [] and c["execution_count"] is None for c in code_cells)
        cells = ["".join(c["source"]) for c in code_cells]
    assert len(cells) == 2
    result = subprocess.run(
        [sys.executable, "-c", "\n\n".join(cells)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_replication_notebook_runs_from_committed_results_in_a_fresh_process():
    notebook = json.loads((ROOT / "notebooks/02_peso_replication.ipynb").read_text())
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code_cells)
    result = subprocess.run(
        [sys.executable, "-c", "\n\n".join("".join(c["source"]) for c in code_cells)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Numbers come from the committed metric files, never from ignored run folders.
    assert "| peso | 0.0177 ± 0.0011 (0.0193)" in result.stdout
    assert "-0.0001 ± 0.0002 | +0.0012" in result.stdout


def test_colab_notebook_records_match_committed_seed43_results():
    notebook = json.loads((ROOT / "notebooks/03_peso_colab.ipynb").read_text())
    code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] for c in notebook["cells"] if c["cell_type"] == "code")
    source = next(c for c in code_cells if "RECORDED = {" in c)
    recorded = ast.literal_eval(re.search(r"RECORDED = (\{.*?\n\})", source, re.DOTALL)[1])
    tags = {
        "pretrain": "sflora_pretrain",
        "lora": "sflora_cl_0.0",
        "peso": "sflora_kldiv_latest_cl_2.0",
    }
    for method, tag in tags.items():
        path = (
            ROOT
            / f"replication/peso/results/seed43/results/QUICK_ws20.msl-1.temp0.8.{tag}.blocks5.json"
        )
        blocks = json.loads(path.read_text())["all_block_results"]
        for block, (ndcg, hit) in recorded[method].items():
            metrics = blocks[f"block_{block}"]["all_prompt_results"][0]
            assert round(metrics["ndcg@10"], 6) == ndcg and round(metrics["hit@10"], 6) == hit
