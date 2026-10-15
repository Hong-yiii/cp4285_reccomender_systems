"""The contributor guide is executable sample code, not a second implementation."""

import ast
import re
import subprocess
import sys
from pathlib import Path


def test_documentation_examples(tmp_path):
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs/PACKAGES-AND-BUILDING.md").read_text(encoding="utf-8")
    blocks = re.findall(r"^```python\n(.*?)^```", guide, flags=re.MULTILINE | re.DOTALL)
    assert blocks, "The development guide must contain runnable examples"
    code = "\n\n".join(blocks)
    definitions = {
        node.name: node
        for node in ast.parse(code).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    for name, path in (
        ("CountingRecommender", "src/cp4285/models/counting.py"),
        ("evaluate", "src/cp4285/experiment/evaluation.py"),
    ):
        source = ast.parse((root / path).read_text(encoding="utf-8"))
        definition = next(node for node in source.body if getattr(node, "name", None) == name)
        assert ast.dump(definitions[name]) == ast.dump(definition), f"Update the {name} example"
    subprocess.run(
        [sys.executable, "-"], input=code, text=True, cwd=tmp_path, check=True, timeout=30
    )
    assert len(list((tmp_path / "runs").glob("*/results.json"))) == 1
