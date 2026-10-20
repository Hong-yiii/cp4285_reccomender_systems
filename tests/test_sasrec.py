"""CPU lifecycle executes offline; CUDA cases skip explicitly without a GPU."""

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
from cp4285.models.sasrec import SASRecRecommender, cuda_preflight, device_preflight, resolve_device

devices = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA hardware unavailable; CPU tests are not CUDA evidence",
        ),
    ),
]


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cuda_required_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = torch.get_rng_state().clone()
    for action in (
        cuda_preflight,
        lambda: SASRecRecommender({"A": ("x", "y")}, device="cuda"),
        lambda: SASRecRecommender.load(tmp_path / "must-not-be-read.pt", device="cuda"),
    ):
        with pytest.raises(RuntimeError, match="NVIDIA CUDA GPU required.*uv sync --locked"):
            action()
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    default = SASRecRecommender({"A": ("x", "y")})
    automatic = SASRecRecommender({"A": ("x", "y")}, device="auto")
    assert default.device == automatic.device == torch.device("cpu")
    equal(state(default), state(automatic))
    assert device_preflight("cpu")["device"] == "cpu"
    with pytest.raises(ValueError, match="device must"):
        SASRecRecommender({"A": ("x", "y")}, device="mps")
    assert torch.equal(torch.get_rng_state(), rng)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["fixture", "audit", "real"])
def test_notebook_explicit_cuda_fails_before_data_or_artifacts(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/02_neural_continual_learning.ipynb").read_text())
    code = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        CP4285_DEVICE="cuda",
        CP4285_NEURAL_MODE=mode,
        CP4285_AMAZON_A="/must-not-open-A.csv",
        CP4285_AMAZON_B="/must-not-open-B.csv",
    )
    result = subprocess.run(
        [sys.executable, "-"],
        input=code,
        text=True,
        cwd=tmp_path,
        capture_output=True,
        check=False,
        timeout=30,
        env=env,
    )
    assert result.returncode != 0
    assert "NVIDIA CUDA GPU required" in result.stderr
    assert "uv sync --locked --extra neural-cuda" in result.stderr
    assert not list(tmp_path.iterdir())


def equal(left, right, *, approximate=False):
    if isinstance(left, torch.Tensor):
        if approximate:
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        else:
            assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key], approximate=approximate)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b, approximate=approximate)
    elif approximate and isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-6, abs=1e-7)
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


@pytest.mark.parametrize("device", devices)
def test_score_purity_rng_forks_checkpoint_next_update(tmp_path, device):
    runtime = device_preflight(device)
    assert str(runtime["device"]).startswith(device)
    global_rng = torch.get_rng_state().clone()
    global_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    model = SASRecRecommender({"A": ("x", "y"), "B": ("x", "y")}, dropout=0.3, device=device)
    if device == "cuda":
        assert resolve_device("auto") == model.device
    model.train((example(), example("B")))
    assert all(p.device == model.device for p in model.network.parameters())
    assert model.generator.device == model.device
    assert model._ids(example().query).device == model.device
    assert all(
        value.device == model.device
        for values in model.optimizer.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    )
    assert all(int(values["step"]) <= 2 for values in model.optimizer.state.values())
    equal(torch.get_rng_state(), global_rng)
    equal(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], global_cuda_rng)
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
    equal(state(left), state(right), approximate=True)  # independent optimizer AND RNG
    path = tmp_path / "learner.pt"
    left.begin_continuation()
    left.train((example(),))
    left.save(path)
    restored = SASRecRecommender.load(path, device=device)
    equal(state(left), state(restored))
    assert all(
        value.device == restored.device
        for values in restored.optimizer.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    )
    # Staging storage on CPU must preserve the recorded generator backend.
    portable = tmp_path / "portable.pt"
    torch.save(torch.load(path, map_location="cpu", weights_only=True), portable)
    remapped = SASRecRecommender.load(portable, device=device)
    equal(state(left), state(remapped))
    incompatible = tmp_path / "incompatible.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["device_type"] = "cuda" if device == "cpu" else "cpu"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="RNG backend differs"):
        SASRecRecommender.load(incompatible, device=device)
    payload["format"] = 2
    torch.save(payload, tmp_path / "old-format.pt")
    with pytest.raises(ValueError, match="checkpoint format"):
        SASRecRecommender.load(tmp_path / "old-format.pt", device=device)
    left.train((example("B"),))
    restored.train((example("B"),))
    remapped.train((example("B"),))
    equal(state(left), state(restored), approximate=True)
    equal(state(left), state(remapped), approximate=True)
    with pytest.raises(FileExistsError):
        left.save(path)
    with pytest.raises(ValueError, match="once"):
        restored.begin_continuation()
    with pytest.raises(ValueError, match="Unknown"):
        model.score(Query("A", "u", ("unknown",)), ("x",))
    equal(torch.get_rng_state(), global_rng)
    equal(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], global_cuda_rng)


@pytest.mark.parametrize("device", devices)
def test_causal_history_actual_learning_and_shared_encoder(device):
    model = SASRecRecommender({"A": ("x", "y"), "B": ("x", "y")}, dropout=0.0, device=device)
    with torch.no_grad():
        long = model.network.encode("A", torch.tensor([[1, 2]], device=model.device))
        short = model.network.encode("A", torch.tensor([[1]], device=model.device))
    assert torch.allclose(long[:, 0], short[:, 0], atol=1e-6)
    initial = model.score(example().query, ("x", "y"))
    model.train((example(),) * 8)
    final = model.score(example().query, ("x", "y"))
    assert final[1] - final[0] > initial[1] - initial[0]
    assert model.targets_seen == 8
    assert int(model.optimizer.state[model.network.positions.weight]["step"]) == 8
    a_table = model.network.item_table("A").weight.detach().clone()
    shared = model.network.positions.weight.detach().clone()
    model.begin_continuation()  # remove A optimizer momentum before B-only updates
    before = model.score(example().query, ("x", "y"))
    model.train((example("B"),))
    assert torch.equal(a_table, model.network.item_table("A").weight)
    assert not torch.equal(shared, model.network.positions.weight)
    assert before != model.score(example().query, ("x", "y"))


@pytest.mark.parametrize("device", devices)
def test_neural_shared_runner_and_observation(tmp_path, device):
    source = AmazonSource(files(tmp_path))
    data = source.load_data()
    model = SASRecRecommender(source.vocabularies, width=8, device=device)
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


@pytest.mark.parametrize("device", devices)
def test_neural_notebook_offline(tmp_path, monkeypatch, device):
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
    env["CP4285_TORCH_THREADS"] = "1"
    if device == "cpu":
        env.pop("CP4285_DEVICE", None)  # Exercise the notebook's real CPU default.
    else:
        env["CP4285_DEVICE"] = device
    subprocess.run(
        [sys.executable, "-"], input=code, text=True, cwd=tmp_path, check=True, timeout=180, env=env
    )
    assert len(list((tmp_path / "runs").glob("*/results.json"))) == 1
    assert list((tmp_path / "runs").glob("*/checkpoints/selected.pt"))
    runtime_path = next((tmp_path / "runs").glob("*/runtime.json"))
    runtime = json.loads(runtime_path.read_text())
    assert runtime["device"].startswith(device)
    if sys.platform != "win32":
        assert runtime["peak_process_rss_bytes"] > 0
    else:
        assert runtime["peak_process_rss_bytes"] is None
    if device == "cuda":
        assert runtime["peak_allocated_bytes"] > 0
    else:
        assert runtime["peak_allocated_bytes"] is None
