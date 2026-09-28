"""Protect the standalone layout without downloading data or training a model."""

from pathlib import Path

from cp4285 import cli


def test_repository_defaults(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    assert cli.ROOT == root
    cfg = cli.config(root / "configs/pilot.toml")
    assert cfg["data"]["a"] == str(root / "data/raw/Electronics.csv.gz")
    assert cfg["data"]["b"] == str(root / "data/raw/Movies_and_TV.csv.gz")
    assert cfg["data"]["prepared"] == str(root / "data/processed/electronics-pilot.json")
    assert cfg["training"]["output"] == str(root / "runs/electronics-pilot")

    saved = []
    monkeypatch.setattr("sys.argv", ["cp4285", "audit"])
    monkeypatch.setattr(cli, "audit", lambda *args: {"inputs": args[:2]})
    monkeypatch.setattr(cli, "save_json", lambda path, value: saved.append((path, value)))
    cli.main()
    assert saved == [
        (
            str(root / "reports/dataset-audit.json"),
            {"inputs": (cfg["data"]["a"], cfg["data"]["b"])},
        )
    ]
