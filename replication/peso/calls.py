"""Start or watch pipeline stages on the deployed Modal app, independent of this client.

A detached `modal run` keeps only its last triggered call alive when the local client
disconnects; a brief network drop on 19 October cancelled three runs that way. Deploy once,
then spawn server-side calls:

    modal deploy replication/peso/modal_app.py
    python replication/peso/calls.py spawn replicate 42 43 44 > runs/peso/calls.json
    python replication/peso/calls.py watch runs/peso/calls.json

Run with the Python environment that provides `modal` (for example its pipx venv).
"""

import json
import sys
import time
from pathlib import Path

import modal

APP = "cp4285-peso-replication"


def spawn(stage, seeds):
    pipeline = modal.Function.from_name(APP, "pipeline")
    calls = {f"seed{s}": pipeline.spawn(stage, f"seed{s}", int(s)).object_id for s in seeds}
    print(json.dumps({"stage": stage, "calls": calls}))


def watch(path, interval=120):
    """Print one line per finished or failed call; network errors are retried."""
    pending = dict(json.loads(Path(path).read_text())["calls"])
    while pending:
        for name, call_id in list(pending.items()):
            try:
                result = modal.FunctionCall.from_id(call_id).get(timeout=0)
                print(f"{name} DONE {json.dumps(result)}", flush=True)
                del pending[name]
            except TimeoutError:
                pass
            except (OSError, modal.exception.ConnectionError) as exc:
                print(f"{name} poll error, retrying: {exc!r}", file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001 - any remote failure ends that call
                print(f"{name} FAILED {exc!r}", flush=True)
                del pending[name]
        if pending:
            time.sleep(interval)


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    spawn(rest[0], rest[1:]) if command == "spawn" else watch(rest[0])
