"""Execute the public recipes without Jupyter or hidden interpreter state."""

import ast
import json
import re
import shutil
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
    # Seed-42 analysis reports: transfer matrix, flips, penalty size and J-lens.
    assert "backward transfer: NDCG@10 -0.0061, NDCG@20 -0.0065" in result.stdout
    assert "| lora | all | 606 | 496 | 467 | 45% |" in result.stdout
    assert "smallest p over 20 pairs: 0.06" in result.stdout
    assert "| block1 | 7.31e-07 | 7.29e-07 | 1.41e-06 | 4.7e-04 | 3.8% |" in result.stdout
    assert "| peso | l4 | 0.357 | 0.212 | 0.270 |" in result.stdout
    assert "| PESO, size-normalized λ=400 | -0.0029 | 0.0140 | 0.0169 | 0.0165 |" in result.stdout
    assert "90 users right only with the penalty, 75 only without" in result.stdout


def test_colab_notebook_compares_against_the_cloned_repositorys_results(tmp_path):
    notebook = json.loads((ROOT / "notebooks/03_peso_colab.ipynb").read_text())
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code_cells)
    sources = ["".join(c["source"]) for c in code_cells]
    assert 'RELEASE = f"https://github.com/{REPO}/releases/download/' in sources[0]
    layout = next(s for s in sources if "TAGS = {" in s)
    tags = ast.literal_eval(re.search(r"TAGS = (\{.*?\n\})", layout, re.DOTALL)[1])
    # A reproduction identical to our run: committed seed-43 files in the notebook's layout.
    committed = ROOT / "replication/peso/results/seed43/results"
    for method, (tag, _) in tags.items():
        for block in (1, 2, 3):
            shutil.copy(committed / f"{tag}.json", tmp_path / f"{method}.block{block}.json")
    prelude = (
        f"import json\nfrom pathlib import Path\nrepo = Path({str(ROOT)!r})\nSEED = 43\n"
        f"TAGS = {tags!r}\nblocks = (1, 2, 3)\nresults = Path({str(tmp_path)!r})\n"
    )
    compare = next(s for s in sources if "RECORDED = " in s)
    result = subprocess.run(
        [sys.executable, "-c", prelude + compare],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "now -0.0004; recorded -0.0004; paper +0.0012" in result.stdout


def test_colab_jlens_side_note_reads_the_committed_seed42_report():
    notebook = json.loads((ROOT / "notebooks/03_peso_colab.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    data = next(s for s in sources if "JLENS = " in s)  # the plot cell needs matplotlib
    prelude = f"import json\nfrom pathlib import Path\nrepo = Path({str(ROOT)!r})\n"
    result = subprocess.run(
        [sys.executable, "-c", prelude + data],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "final output 0.213, chance 0.024" in result.stdout
    assert "jlens lens at layers (4, 8, 12): 0.097, 0.095, 0.107" in result.stdout
    assert "peso: 0.15 0.21 0.30 0.36 | 0.07 0.12 0.16 0.21 | 0.12 0.16 0.24 0.27" in result.stdout


def test_colab_item_level_side_note_reads_the_committed_seed42_report():
    notebook = json.loads((ROOT / "notebooks/03_peso_colab.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    data = next(s for s in sources if "ITEMS = " in s)  # the plot cell needs matplotlib
    prelude = f"import json\nfrom pathlib import Path\nrepo = Path({str(ROOT)!r})\n"
    result = subprocess.run(
        [sys.executable, "-c", prelude + data],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "code 1 (broad) 41%, code 2 43%, code 3 24%, code 4 (exact) 17%" in result.stdout
    assert "code 4 (exact): 0.727; 0.108 / 0.198 / 0.681" in result.stdout
    assert "layer 8: 15.8% (chance pairing 22.4%)" in result.stdout
