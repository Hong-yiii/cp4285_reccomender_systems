"""Protect the standalone layout without downloading data or training a model."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

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


def test_reviewer_reference_is_self_contained_and_links_resolve():
    root = Path(__file__).resolve().parents[1]
    html = (root / "reference.html").read_text(encoding="utf-8")
    ids, links = [], []

    class ReferenceParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            assert tag not in {"script", "link", "iframe", "object", "embed", "base"}
            assert not any(key.startswith("on") or key in {"src", "srcset"} for key in attrs)
            if "id" in attrs:
                ids.append(attrs["id"])
            if "href" in attrs:
                links.append(attrs["href"])

    parser = ReferenceParser()
    parser.feed(html)
    parser.close()
    assert len(ids) == len(set(ids))
    assert {"review-status", "reviewer-checkpoint"} <= set(ids)
    for href in links:
        target = urlsplit(href)
        assert target.scheme in {"", "https"}
        if target.scheme:
            continue
        assert not target.netloc
        if target.path:
            path = (root / unquote(target.path)).resolve()
            assert path.is_relative_to(root) and path.is_file()
        else:
            assert unquote(target.fragment) in ids
    assert "@import" not in html.lower() and "url(" not in html.lower()
    for private in ("/Users/", "file://", "localhost:", "Delete this page when finished"):
        assert private not in html


@pytest.mark.parametrize("malformed", [False, True])
def test_cli_config_errors_are_reported_without_traceback(tmp_path, monkeypatch, capsys, malformed):
    path = tmp_path / "invalid.toml"
    if malformed:
        path.write_text("[")
    monkeypatch.setattr("sys.argv", ["cp4285", "--config", str(path), "doctor"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert capsys.readouterr().err.startswith("Error: ")
