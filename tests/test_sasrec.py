"""Neural lifecycle checks; optional CPU Torch, never a real-data result."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_amazon import files

from cp4285.experiment import Event, Example, Query, run
from cp4285.experiment.amazon import AmazonSource
from cp4285.models.sasrec import SASRecRecommender


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def state(model):
    return copy.deepcopy(
        (
            model.network.state_dict(),
            model.optimizer.state_dict(),
            model.generator.get_state(),
            model.targets_seen,
            model.last_loss,
        )
    )


def example(domain="A", target="y"):
    return Example(
        (Event(domain, "u", "context", "x", 0),), Event(domain, "u", "target", target, 1)
    )


def test_score_purity_rng_forks_checkpoint_next_update(tmp_path):
    torch.set_num_threads(1)
    global_rng = torch.get_rng_state().clone()
    model = SASRecRecommender({"A": ("x", "y"), "B": ("x", "y")}, dropout=0.3)
    model.train((example(), example("B")))
    equal(torch.get_rng_state(), global_rng)
    before = state(model)
    query = example().query
    scores = model.score(query, ("x", "y"))
    assert model.score(query, ("y", "x")) == tuple(reversed(scores))
    equal(state(model), before)
    left, right = model.fork(), model.fork()
    assert isinstance(left, SASRecRecommender) and isinstance(right, SASRecRecommender)
    left.train((example("B"),))
    equal(state(right), before)
    equal(state(model), before)
    right.train((example("B"),))
    equal(state(left), state(right))  # independent optimizer AND RNG, not just equal rankings
    path = tmp_path / "learner.pt"
    left.begin_continuation()
    left.train((example(),))
    left.save(path)
    restored = SASRecRecommender.load(path)
    equal(state(left), state(restored))
    left.train((example("B"),))
    restored.train((example("B"),))
    equal(state(left), state(restored))
    with pytest.raises(FileExistsError):
        left.save(path)
    with pytest.raises(ValueError, match="once"):
        restored.begin_continuation()
    with pytest.raises(ValueError, match="Unknown"):
        model.score(Query("A", "u", ("unknown",)), ("x",))
    equal(torch.get_rng_state(), global_rng)


def test_causal_history_actual_learning_and_shared_encoder():
    torch.set_num_threads(1)
    model = SASRecRecommender({"A": ("x", "y"), "B": ("x", "y")}, dropout=0.0)
    with torch.no_grad():
        long = model.network.encode("A", torch.tensor([[1, 2]]))
        short = model.network.encode("A", torch.tensor([[1]]))
    assert torch.allclose(long[:, 0], short[:, 0], atol=1e-6)
    initial = model.score(example().query, ("x", "y"))
    model.train((example(),) * 8)
    final = model.score(example().query, ("x", "y"))
    assert final[1] - final[0] > initial[1] - initial[0]
    assert model.targets_seen == 8
    a_table = model.network.item_table("A").weight.detach().clone()
    shared = model.network.positions.weight.detach().clone()
    model.begin_continuation()  # remove A optimizer momentum before B-only updates
    before = model.score(example().query, ("x", "y"))
    model.train((example("B"),))
    assert torch.equal(a_table, model.network.item_table("A").weight)
    assert not torch.equal(shared, model.network.positions.weight)
    assert before != model.score(example().query, ("x", "y"))


def test_neural_shared_runner_and_observation(tmp_path):
    torch.set_num_threads(1)
    source = AmazonSource(files(tmp_path))
    data = source.load_data()
    model = SASRecRecommender(source.vocabularies, width=8)
    events = []

    def observe(event, learner):
        events.append(event)
        if event["phase"] == "selected":
            assert isinstance(learner, SASRecRecommender)
            learner.save(tmp_path / "selected.pt")

    result = run(model, data, output=tmp_path / "run", initial_epochs=2, k=1, observe=observe)
    assert model.targets_seen == 0
    assert len(result["records"]) == 36
    assert events[0]["phase"] == "initial"
    assert len([e for e in events if e["phase"] == "selected"]) == 1
    assert all(
        r["ndcg_change_from_initial"] == 0 for r in result["records"] if r["condition"] == "frozen"
    )
    assert (tmp_path / "selected.pt").is_file()


def test_neural_notebook_offline(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/02_neural_continual_learning.ipynb").read_text())
    code = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    # Even a developer's real-data environment must never select real inputs in tests.
    monkeypatch.setenv("CP4285_NEURAL_MODE", "real")
    monkeypatch.setenv("CP4285_AMAZON_A", "/unavailable-real-A.csv")
    monkeypatch.setenv("CP4285_AMAZON_B", "/unavailable-real-B.csv")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CP4285_AMAZON_A", "CP4285_AMAZON_B"}
    }
    env["CP4285_NEURAL_MODE"] = "fixture"
    subprocess.run(
        [sys.executable, "-"], input=code, text=True, cwd=tmp_path, check=True, timeout=90, env=env
    )
    assert len(list((tmp_path / "runs").glob("*/results.json"))) == 1
    assert list((tmp_path / "runs").glob("*/checkpoints/selected.pt"))
