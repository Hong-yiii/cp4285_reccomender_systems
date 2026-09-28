"""Shared file and timestamp helpers, using only the standard library."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


def save_json(path, value):
    """Publish a complete JSON artifact without replacing any existing file or symlink."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite artifact: {path}; choose a new path")
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp") as f:
        f.write(payload)
        f.flush()
        # Same-filesystem link publishes atomically and refuses a racing destination.
        os.link(f.name, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def millis(date):
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() * 1000)
