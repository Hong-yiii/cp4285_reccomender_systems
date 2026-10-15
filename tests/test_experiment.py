"""Executable shared-interface checks; all examples are synthetic, no downloads."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import pytest

from cp4285.experiment import (
    Condition,
    DataSource,
    Event,
    Example,
    Recommender,
    SyntheticSource,
    evaluate,
    interleave,
    plot_results,
    run,
)
from cp4285.experiment.artifacts import save_json
from cp4285.models import CountingRecommender


def test_shared_lifecycle_and_saved_graph(tmp_path):
    data = SyntheticSource().load_data()
    model = CountingRecommender()
    output = tmp_path / "run"
    result = run(model, data, output=output, initial_epochs=3, k=1)
    assert model.counts == {}  # The caller's model was not trained in place.
    assert result["selected_epoch"] == 1  # First validation tie, not the final epoch.
    assert json.loads((output / "results.json").read_text()) == result
    assert json.loads((output / "prepared.json").read_text())["provenance"].startswith("SYNTHETIC")
    rows = result["records"]
    frozen = [r for r in rows if r["condition"] == "frozen" and r["domain"] == "A"]
    assert [r["ndcg"] for r in frozen] == [1.0, 1.0, 1.0]
    b_only = [r for r in rows if r["condition"] == "B-only" and r["domain"] == "A"]
    assert all(r["ndcg"] == 1.0 for r in b_only)  # Independent domains are an isolation control.
    assert b_only[-1]["targets_seen"] == {"B": 8}
    mixed = [r for r in rows if r["condition"] == "mixed" and r["domain"] == "A"]
    assert mixed[0]["targets_seen"] == {}
    assert mixed[-1]["targets_seen"] == {"A": 4, "B": 4}
    assert mixed[-1]["ndcg"] == 0.0
    a_only = [r for r in rows if r["condition"] == "A-only" and r["domain"] == "A"]
    assert a_only[-1]["ndcg"] == 0.0  # Would differ if all three initial epochs were kept.
    svg = plot_results(output / "results.json", task="retention-A")
    ElementTree.fromstring(svg)
    assert "SYNTHETIC" in svg and "NDCG@1" in svg
    with pytest.raises(FileExistsError):
        run(model, data, output=output, initial_epochs=1, k=1)
    with pytest.raises(ValueError, match="No saved results"):
        plot_results(output / "results.json", task="absent")


def test_abc_and_model_branch_contract():
    with pytest.raises(TypeError):
        DataSource()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Recommender()  # type: ignore[abstract]
    data = SyntheticSource().load_data()
    model = CountingRecommender()
    model.train(data.initial)
    before = model.counts.copy()
    task = data.tests[0]
    scores = model.score(task.examples[0].query, task.candidates)
    assert scores == model.score(task.examples[0].query, task.candidates)
    assert model.counts == before
    left, right = model.fork(), model.fork()
    left.begin_continuation()
    left.train(data.conditions[0].blocks[0])
    assert right.score(task.examples[0].query, task.candidates) == scores
    assert model.counts == before


def test_metric_candidate_order_and_failures():
    task = SyntheticSource().load_data().tests[1]  # b2 loses an all-zero lexical tie to b1.
    model = CountingRecommender()
    assert evaluate(model, task, k=1) == {"ndcg": 0.0, "hit": 0.0, "recall": 0.0}
    assert evaluate(model, replace(task, candidates=tuple(reversed(task.candidates))), k=1) == (
        evaluate(model, task, k=1)
    )
    assert evaluate(model, task, k=2)["ndcg"] == pytest.approx(0.6309297535714575)

    class InvalidScores(CountingRecommender):
        def score(self, query, candidates):
            return (float("nan"),) * len(candidates)

    with pytest.raises(ValueError, match="finite"):
        evaluate(InvalidScores(), task, k=1)
    with pytest.raises(ValueError, match="positive"):
        evaluate(model, task, k=0)


def test_schedule_rounding_independent_cursors_and_exhaustion():
    data = SyntheticSource().load_data()
    a = tuple(e for block in data.conditions[0].blocks for e in block)
    b = tuple(e for block in data.conditions[2].blocks for e in block)
    plan = interleave(a, b, b_shares=(0.5, 0.0, 1.0), block_size=3, name="schedule")
    assert [[e.target.domain for e in block] for block in plan.blocks] == [
        ["A", "B", "B"],
        ["A", "A", "A"],
        ["B", "B", "B"],
    ]
    assert plan == interleave(a, b, b_shares=(0.5, 0.0, 1.0), block_size=3, name="schedule")
    with pytest.raises(ValueError, match="Insufficient"):
        interleave(a, b, b_shares=(0.0,) * 3, block_size=3, name="too-long")
    with pytest.raises(ValueError, match="fractions"):
        interleave(a, b, b_shares=(float("nan"),), block_size=3, name="bad")


def test_data_contract_rejects_leakage_replay_and_invalid_order():
    data = SyntheticSource().load_data()
    held = data.validation.examples[0].target
    later = Event("A", held.user, "later", "a2", 100)
    leaking = Example((held,), later)
    with pytest.raises(ValueError, match="Held-out"):
        replace(data, conditions=(Condition("leak", ((leaking,),)),)).validate()
    with pytest.raises(ValueError, match="fresh"):
        replace(data, conditions=(Condition("replay", (data.initial,)),)).validate()
    block = data.conditions[0].blocks[0]
    with pytest.raises(ValueError, match="chronological"):
        replace(data, conditions=(Condition("reverse", (tuple(reversed(block)),)),)).validate()
    with pytest.raises(ValueError, match="missing"):
        replace(
            data, validation=replace(data.validation, candidates=("not-the-target",))
        ).validate()
    bad_history = Example((replace(held, user="unrelated"),), later)
    with pytest.raises(ValueError, match="splice"):
        replace(data, conditions=(Condition("splice", ((bad_history,),)),)).validate()
    conflicting = replace(data.initial[0], target=replace(data.initial[0].target, item="changed"))
    with pytest.raises(ValueError, match="conflicting"):
        replace(data, initial=(*data.initial, conflicting)).validate()


def test_notebook_runs_without_hidden_state(tmp_path):
    """Run trusted repository notebook cells in a fresh Python process, in order."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/01_shared_experiment.ipynb").read_text())
    code = "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    subprocess.run(
        [sys.executable, "-"], input=code, text=True, cwd=tmp_path, check=True, timeout=30
    )
    assert len(list((tmp_path / "runs").glob("*/results.json"))) == 1


def test_test_target_cannot_influence_validation_selection():
    data = SyntheticSource().load_data()
    test_target = data.tests[0].examples[0].target
    validation_target = replace(data.validation.examples[0].target, timestamp=6)
    validation = replace(data.validation, examples=(Example((test_target,), validation_target),))
    with pytest.raises(ValueError, match="Test target leaked"):
        replace(data, validation=validation).validate()


def test_global_cutoff_belongs_to_concrete_loader_not_shared_contract():
    data = SyntheticSource().load_data()
    other_user = Example((), Event("A", "other-user", "initial", "a1", 100))
    # The fixture user's fresh updates at 10..17 still follow their own initial 1..3.
    replace(data, initial=(*data.initial, other_user)).validate()


def test_json_publication_preserves_existing_and_racing_artifacts(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    save_json(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_json(path, {"replacement": True})
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        save_json(tmp_path / "invalid.json", {"score": float("nan")})
    race_path = tmp_path / "race.json"
    real_link = os.link

    def race(source, destination):
        assert json.loads(Path(source).read_text()) == {"new": True}
        race_path.write_text("other writer")
        real_link(source, destination)

    monkeypatch.setattr("os.link", race)
    with pytest.raises(FileExistsError):
        save_json(race_path, {"new": True})
    assert race_path.read_text() == "other writer"
    assert set(tmp_path.iterdir()) == {path, race_path}
