"""Shared validation selection, isolated continuation branches and fixed evaluation."""

import json
import math
import platform
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from .artifacts import save_json
from .data import PreparedExperiment
from .evaluation import evaluate
from .model import Recommender


def run(
    model: Recommender,
    data: PreparedExperiment,
    *,
    output: str | Path,
    initial_epochs: int,
    k: int,
    observe: Callable[[dict, Recommender], None] | None = None,
) -> dict:
    """Run an explicit small protocol; never mutate the caller's model or overwrite output.

    One initial epoch is one train(initial) call. One continuation block is one
    train(block) call. Selection uses validation NDCG only, retaining the first
    best epoch. Steps/examples are recorded, not claimed to be matched FLOPs.
    Optional observe(event, learner) reports completed epochs/evaluations and selected
    state for logging/create-only checkpoints. It must not mutate the learner or tune
    on test records. Observer errors fail the run, leaving partial artifacts intact.
    """
    data.validate()
    if type(initial_epochs) is not int or initial_epochs <= 0 or type(k) is not int or k <= 0:
        raise ValueError("initial_epochs and k must be positive integers")
    snapshot = asdict(data)
    data_hash = sha256(json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()).hexdigest()
    package = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(path.relative_to(package)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    manifest = {
        "schema_version": 1,
        "provenance": data.provenance,
        "data_sha256": data_hash,
        "model": model.describe(),
        "initial_epochs": initial_epochs,
        "k": k,
        "selection": "first best validation NDCG",
        "evaluation": "single target; fixed candidates; lexical item ties; no seen masking",
        "budget": "one train call per block; one supervised target per example; no hidden replay",
        "python": platform.python_version(),
        "package_version": version("cp4285-recommender-study"),
        "source_sha256": code_hashes,
    }
    # Fail on non-serializable/nonfinite model metadata before creating any artifacts.
    json.dumps(manifest, allow_nan=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "prepared.json", snapshot)
    save_json(output / "manifest.json", manifest)

    working = model.fork()
    best = None
    best_score = -math.inf
    selected_epoch = 0
    validation = []
    for epoch in range(1, initial_epochs + 1):
        working.train(data.initial)
        metrics = evaluate(working, data.validation, k=k)
        validation.append({"epoch": epoch, **metrics})
        if observe is not None:
            observe({"phase": "initial", "epoch": epoch, **metrics}, working)
        if metrics["ndcg"] > best_score:
            best, best_score, selected_epoch = working.fork(), metrics["ndcg"], epoch
    assert best is not None
    if observe is not None:
        observe({"phase": "selected", "epoch": selected_epoch}, best)

    baseline = {task.name: evaluate(best, task, k=k) for task in data.tests}
    records = []

    def record(learner: Recommender, condition: str, step: int, exposure: Counter) -> None:
        for task in data.tests:
            metrics = evaluate(learner, task, k=k)
            if (condition == "frozen" or step == 0) and metrics != baseline[task.name]:
                raise ValueError(
                    "Frozen/branch baseline changed: check fork isolation and score purity"
                )
            records.append(
                {
                    "condition": condition,
                    "step": step,
                    "task": task.name,
                    "domain": task.domain,
                    "n": len(task.examples),
                    "candidate_count": len(task.candidates),
                    "targets_seen": dict(exposure),
                    **metrics,
                    "ndcg_change_from_initial": metrics["ndcg"] - baseline[task.name]["ndcg"],
                }
            )
            if observe is not None:
                observe({"phase": "evaluation", **records[-1]}, learner)

    frozen = best.fork()
    record(frozen, "frozen", 0, Counter())
    for condition in data.conditions:
        branch = best.fork()
        branch.begin_continuation()
        exposure = Counter()
        record(branch, condition.name, 0, exposure)
        for step, block in enumerate(condition.blocks, 1):
            branch.train(block)
            exposure.update(example.target.domain for example in block)
            record(branch, condition.name, step, exposure)
    for step in range(1, max(len(c.blocks) for c in data.conditions) + 1):
        record(frozen, "frozen", step, Counter())

    result = {
        "manifest": manifest,
        "selected_epoch": selected_epoch,
        "validation": validation,
        "records": records,
    }
    save_json(output / "results.json", result)
    return result
