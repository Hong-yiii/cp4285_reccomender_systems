"""The contributor guide is executable sample code, not a second implementation."""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def test_neural_dependency_is_cuda_only():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert config["project"]["dependencies"] == []  # Core stays standard-library-only.
    assert config["project"]["optional-dependencies"]["neural"] == [
        "torch==2.14.1+cu126; sys_platform == 'linux' or sys_platform == 'win32'"
    ]
    assert config["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cu126", "extra": "neural"}
    ]
    assert config["tool"]["uv"]["index"] == [
        {"name": "pytorch-cu126", "url": "https://download.pytorch.org/whl/cu126", "explicit": True}
    ]


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
