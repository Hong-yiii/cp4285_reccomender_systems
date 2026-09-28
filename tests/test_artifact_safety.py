"""Offline regressions for download promotion and immutable JSON artifacts."""

import gzip
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import httpx
import pytest

from cp4285 import cli, data
from cp4285.common import utils


def test_common_file_hash_and_utc_dates(tmp_path):
    payload = b"common helpers\n" * 100_000  # Exercise more than one hash chunk.
    path = tmp_path / "data.bin"
    path.write_bytes(payload)
    assert utils.sha256(path) == hashlib.sha256(payload).hexdigest()
    assert utils.millis("2021-01-01") == 1609459200000
    with pytest.raises(FileNotFoundError):
        utils.sha256(tmp_path / "missing.bin")
    with pytest.raises(ValueError):
        utils.millis("not-a-date")


def test_json_preserves_existing_artifacts_and_rejects_invalid_payloads(tmp_path):
    path = tmp_path / "report.json"
    utils.save_json(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        utils.save_json(path, {"replacement": True})
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        utils.save_json(tmp_path / "invalid.json", {"score": float("nan")})
    assert list(tmp_path.iterdir()) == [path]


def test_json_publication_is_complete_and_does_not_replace_a_racing_writer(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    real_link = os.link

    def race(source, destination):
        assert json.loads(Path(source).read_text()) == {"new": True}
        path.write_text("other writer")
        real_link(source, destination)

    monkeypatch.setattr("os.link", race)
    with pytest.raises(FileExistsError):
        utils.save_json(path, {"new": True})
    assert path.read_text() == "other writer"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("damage", [None, "crc", "truncated", "header", "empty", "http"])
def test_download_validates_entire_gzip_before_publication(tmp_path, monkeypatch, damage):
    # Exceed gzip's header-read buffer: a valid first row must not hide a corrupt trailer.
    csv = b"user_id,parent_asin,rating,timestamp\n" + b"u,i,5,1577836800000\n" * 2000
    payload = gzip.compress(csv)
    if damage == "crc":
        payload = payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:]
    elif damage == "truncated":
        payload = payload[:-4]
    elif damage == "header":
        payload = gzip.compress(b"wrong,columns\n")
    elif damage == "empty":
        payload = gzip.compress(b"")
    response = httpx.Response(
        503 if damage == "http" else 200,
        content=payload,
        request=httpx.Request("GET", f"{data.BASE}/Electronics.csv.gz"),
    )
    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: nullcontext(response))
    dest = tmp_path / "Electronics.csv.gz"
    manifest = tmp_path / "Electronics.csv.gz.manifest.json"
    if damage:
        with pytest.raises(ValueError):
            data.download("Electronics", tmp_path)
        assert list(tmp_path.iterdir()) == []
    else:
        record = data.download("Electronics", tmp_path)
        assert gzip.decompress(dest.read_bytes()) == csv
        assert record == json.loads(manifest.read_text())
        assert record == {
            "source_url": f"{data.BASE}/Electronics.csv.gz",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        assert set(tmp_path.iterdir()) == {dest, manifest}


@pytest.mark.parametrize("existing", ["Electronics.csv.gz", "Electronics.csv.gz.manifest.json"])
def test_download_preserves_existing_data_or_manifest_without_network(
    tmp_path, monkeypatch, existing
):
    path = tmp_path / existing
    path.write_bytes(b"original")
    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: pytest.fail("must not request network"))
    with pytest.raises(FileExistsError):
        data.download("Electronics", tmp_path)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("command", ["audit", "prepare"])
def test_cli_refuses_to_replace_audit_and_prepared_snapshots(
    tmp_path, monkeypatch, capsys, command
):
    path = tmp_path / "snapshot.json"
    path.write_text('{"original": true}\n')
    original = path.read_bytes()
    cfg = cli.config(cli.ROOT / "configs/pilot.toml")
    cfg["data"]["prepared"] = str(path)
    monkeypatch.setattr(cli, "config", lambda _: cfg)
    monkeypatch.setattr(cli, command, lambda *a: {"metadata": {"replacement": True}})
    args = ["cp4285", command]
    if command == "audit":
        args.extend(["--output", str(path)])
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "overwrite" in capsys.readouterr().err
    assert path.read_bytes() == original
