"""Execute the public recipes without Jupyter or hidden interpreter state."""

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
