"""The shared ABC is used by real learners, with native protocols kept explicit."""

import copy
import random

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from cp4285.experiment.model import Recommender
from cp4285.models.classical import QRSVD, IncrementalQRSVD, MarkovQRSVD
from cp4285.models.sasrec import SASRec
from cp4285.models.sasrec_learner import SASRecLearner, update


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def snapshot(learner):
    return copy.deepcopy(
        (
            learner.model.state_dict(),
            learner.optimizer.state_dict(),
            learner.rng.getstate(),
            learner.torch_rng_state,
            learner.device_rng_state,
            learner.continuing,
            [module.training for module in learner.model.modules()],
        )
    )


def test_abc_requires_real_methods_and_refit_models_do_not_claim_continuation():
    incomplete = type("Incomplete", (Recommender,), {})
    with pytest.raises(TypeError, match="abstract"):
        incomplete()
    assert issubclass(IncrementalQRSVD, Recommender)
    assert issubclass(SASRecLearner, Recommender)
    assert not issubclass(QRSVD, Recommender)
    assert not issubclass(MarkovQRSVD, Recommender)


def test_neural_lifecycle_matches_native_updates_and_forks_all_learning_state():
    torch.manual_seed(9)
    learner = SASRecLearner(
        SASRec(9, 4, hidden=8, layers=1, dropout=0.3),
        0.001,
        0.002,
        random.Random(9),
    )
    initial = [{"history": [1, 2, 3], "target": 4, "seen": {1, 2, 3, 4}}]
    fresh = [{"history": [1, 2, 3, 4], "target": 5}]
    queries = torch.tensor([[0, 1, 2, 3]])
    raw = copy.deepcopy(learner.model)
    optimizer = torch.optim.Adam(raw.parameters(), lr=0.001, betas=(0.9, 0.98))
    expected = update(raw, optimizer, initial, 9, "cpu", random.Random(9), all_positions=True)
    outside_rng = torch.get_rng_state().clone()
    assert learner.train(initial) == expected
    assert_state_equal(learner.model.state_dict(), raw.state_dict())
    assert_state_equal(learner.optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.get_rng_state(), outside_rng)

    # Pure evaluation preserves even mixed submodule modes and optimizer/RNG state.
    learner.model.blocks[0].eval()
    before = snapshot(learner)
    full = learner.score(queries)
    torch.testing.assert_close(learner.score(queries, (7, 0, 2)), full[:, (7, 0, 2)])
    assert_state_equal(snapshot(learner), before)
    assert torch.equal(torch.get_rng_state(), outside_rng)
    for invalid in ((), (-1,), (9,), (1, 1), (1.5,)):
        with pytest.raises(ValueError, match="Candidates"):
            learner.score(queries, invalid)  # pyright: ignore[reportArgumentType] -- invalid-input test
    assert_state_equal(snapshot(learner), before)

    left, right = learner.fork(), learner.fork()
    for branch in (left, right):
        branch.begin_continuation()
        assert not branch.optimizer.state
        assert_state_equal(branch.model.state_dict(), learner.model.state_dict())
    assert left.optimizer is not learner.optimizer
    left_params = set(left.model.parameters())
    assert set(left.optimizer.param_groups[0]["params"]) == left_params
    assert not left_params.intersection(learner.model.parameters())

    # The two branches receive the same dropout/negative draws despite intervening global RNG use.
    left.train(fresh)
    torch.rand(17)
    right.train(fresh)
    assert_state_equal(snapshot(left), snapshot(right))
    assert_state_equal(snapshot(learner), before)
    assert any(not torch.equal(v, left.model.state_dict()[k]) for k, v in before[0].items())
    optimizer_identity = left.optimizer
    left.train(fresh)
    assert left.optimizer is optimizer_identity
    assert all(state["step"] == 2 for state in left.optimizer.state.values())
    after = snapshot(left)
    with pytest.raises(ValueError, match="already begun"):
        left.begin_continuation()
    assert_state_equal(snapshot(left), after)
    with pytest.raises(ValueError, match="nonempty"):
        left.train([])
    with pytest.raises(ValueError, match="in-catalogue"):
        left.train([{"history": [1], "target": 10}])
    assert_state_equal(snapshot(left), after)


def test_incremental_rows_match_native_svd_and_fork_the_residual_rng():
    rng = np.random.default_rng(3)
    rows = sp.csr_matrix(rng.random((30, 8), dtype=np.float32))
    learner = IncrementalQRSVD(k=3, residual_rank=2, seed=4)
    raw = IncrementalQRSVD(k=3, residual_rank=2, seed=4)
    with pytest.raises(ValueError, match="Initial training"):
        learner.begin_continuation()
    for chunk in (rows[:10], rows[10:20]):
        learner.train(chunk)
        raw.partial_fit(chunk)
    np.testing.assert_array_equal(learner.V, raw.V)
    np.testing.assert_array_equal(learner.S, raw.S)
    before = copy.deepcopy((learner.V, learner.S, learner._rng.bit_generator.state))
    left, right = learner.fork(), learner.fork()
    for branch in (left, right):
        branch.begin_continuation()
        branch.train(rows[20:])
    np.testing.assert_array_equal(left.V, right.V)
    assert_state_equal((learner.V, learner.S, learner._rng.bit_generator.state), before)
    full = learner.score(rows[:2])
    np.testing.assert_array_equal(learner.score(rows[:2], (7, 0, 2)), full[:, (7, 0, 2)])
    assert_state_equal((learner.V, learner.S, learner._rng.bit_generator.state), before)
    with pytest.raises(ValueError, match="already begun"):
        left.begin_continuation()
    with pytest.raises(ValueError, match="nonempty"):
        learner.train(rows[:0])
    with pytest.raises(ValueError, match="fixed item catalogue"):
        learner.train(rows[:, :7])
    with pytest.raises(ValueError, match="Candidates"):
        learner.score(rows[:2], (-1,))
    assert_state_equal((learner.V, learner.S, learner._rng.bit_generator.state), before)
