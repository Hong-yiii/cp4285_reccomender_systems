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
