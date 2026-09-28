"""Small reference-equation check against kang205/SASRec's model.py/modules.py.

This is NOT a live TensorFlow comparison or a reproduction of paper metrics.
"""

import random

import numpy as np
import pytest
import torch

from cp4285.neural.model import SASRec
from cp4285.neural.pilot import initial_sequences, ranking_metrics, tensors, training_tensors


def reference_states(model, history):
    """NumPy translation using upstream's head-concatenation order, without port helpers."""
    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}

    def norm(x, name):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-8) * state[
            name + ".weight"
        ] + state[name + ".bias"]

    def linear(x, name):
        return x @ state[name + ".weight"].T + state[name + ".bias"]

    mask = (history != 0)[..., None]
    x = (state["items.weight"][history] * np.sqrt(model.hidden) + state["positions.weight"]) * mask
    for i, block in enumerate(model.blocks):
        name = f"blocks.{i}."
        queries = norm(x, name + "attention_norm")
        q, k, v = [
            np.concatenate(np.split(value, block.heads, axis=-1), axis=0)
            for value in (
                linear(queries, name + "query"),
                linear(x, name + "key"),
                linear(x, name + "value"),
            )
        ]
        logits = q @ k.transpose(0, 2, 1) / np.sqrt(k.shape[-1])
        keys = np.tile(np.abs(x).sum(-1) != 0, (block.heads, 1))[:, None, :]
        allowed = keys & np.tril(np.ones(logits.shape[-2:], dtype=bool))
        logits = np.where(allowed, logits, -(2**32) + 1)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        probs *= np.tile(np.abs(queries).sum(-1) != 0, (block.heads, 1))[..., None]
        x = np.concatenate(np.split(probs @ v, block.heads, axis=0), axis=-1) + queries
        x = norm(x, name + "forward_norm")
        x = (x + linear(np.maximum(linear(x, name + "forward_in"), 0), name + "forward_out")) * mask
    return norm(x, "norm")


@pytest.mark.parametrize("heads", [1, 2])
def test_original_sasrec_equations_and_masked_loss(heads):
    torch.manual_seed(4285)
    model = SASRec(9, 5, hidden=8, heads=heads, layers=2, dropout=0, l2_emb=0.01).double().eval()
    # Nonzero biases expose incorrect padding/query masking and residual placement.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.uniform_(-0.2, 0.2)
    history = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 3, 2, 1]])
    expected = reference_states(model, history.numpy())
    np.testing.assert_allclose(model(history).detach().numpy(), expected, rtol=1e-9, atol=1e-9)
    expected_scores = expected[:, -1] @ model.items.weight[1:].detach().numpy().T
    np.testing.assert_allclose(model.scores(history).detach().numpy(), expected_scores, atol=1e-9)

    positive = torch.tensor([[0, 0, 2, 3, 4], [0, 3, 2, 1, 5]])
    negative = torch.where(positive != 0, 8, 0)
    active = positive.numpy() != 0
    items = model.items.weight.detach().numpy()
    pos = (expected[active] * items[positive.numpy()[active]]).sum(-1)
    neg = (expected[active] * items[negative.numpy()[active]]).sum(-1)
    expected_loss = (np.logaddexp(0, -pos) + np.logaddexp(0, neg)).mean()
    expected_loss += 0.005 * (
        np.square(items).sum() + np.square(model.positions.weight.detach().numpy()).sum()
    )
    loss = model.sampled_loss(history, positive, negative)
    assert loss.item() == pytest.approx(expected_loss, abs=1e-9)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.items.weight.grad is not None
    assert torch.count_nonzero(model.items.weight.grad[0]) == 0

    changed = history.clone()
    changed[:, -1] = 9
    with torch.no_grad():
        torch.testing.assert_close(model(history)[:, :-1], model(changed)[:, :-1])


def test_sequence_training_and_fresh_event_continuation_masks():
    rows = [
        {"user": "u", "history": [1], "target": 2},
        {"user": "u", "history": [2, 3], "target": 4},
        {"user": "v", "history": [1], "target": 2},
    ]
    sequences = initial_sequences(rows)
    assert len(sequences) == 2
    assert sequences[0]["seen"] == {1, 2, 3, 4}  # includes item outside the final window
    h, p, n = training_tensors(sequences, 6, "cpu", 4, random.Random(1), all_positions=True)
    assert h.tolist() == [[0, 0, 2, 3], [0, 0, 0, 1]]
    assert p.tolist() == [[0, 0, 3, 4], [0, 0, 0, 2]]
    assert set(n[0, -2:].tolist()) <= {5, 6}
    assert torch.equal(n.eq(0), p.eq(0))
    _, fresh, _ = training_tensors(rows[-2:], 6, "cpu", 4, random.Random(1))
    assert fresh.tolist() == [[0, 0, 0, 4], [0, 0, 0, 2]]
    with pytest.raises(ValueError, match="No eligible negative"):
        training_tensors(sequences[:1], 4, "cpu", 4, random.Random(1), all_positions=True)


def test_reference_walkthrough_tensors_and_ranking():
    """Executable toy example from reference.html; no real data or optimizer run."""
    rows = [
        {"user": "u", "history": [1], "target": 2},
        {"user": "u", "history": [1, 2], "target": 3},
        {"user": "u", "history": [1, 2, 3], "target": 4},
    ]
    initial = initial_sequences(rows)
    assert len(initial) == 1 and initial[0]["seen"] == {1, 2, 3, 4}
    h, p, n = training_tensors(initial, 20, "cpu", 5, random.Random(4285), all_positions=True)
    assert h.tolist() == [[0, 0, 1, 2, 3]]
    assert p.tolist() == [[0, 0, 2, 3, 4]]
    assert set(n[0, -3:].tolist()) <= set(range(5, 21))
    assert torch.equal(n.eq(0), p.eq(0))

    later = [
        {"history": [1, 2, 3, 4], "target": 5},
        {"history": [1, 2, 3, 4, 5], "target": 6},
    ]
    ch, cp, cn = training_tensors(later, 20, "cpu", 5, random.Random(4285))
    assert ch.tolist() == [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
    assert cp.tolist() == [[0, 0, 0, 0, 5], [0, 0, 0, 0, 6]]
    assert torch.equal(cn.eq(0), cp.eq(0))
    retention, target = tensors([{"history": [1, 2, 3, 4], "target": 8}], "cpu", 5)
    assert retention.tolist() == [[0, 1, 2, 3, 4]] and target.tolist() == [8]

    torch.manual_seed(4285)
    model = SASRec(20, 5, dropout=0).eval()
    with torch.no_grad():
        states = model(retention)
        assert states.shape == (1, 5, 50)
        assert model.scores(retention).shape == (1, 20)
        torch.testing.assert_close(
            model.scores(retention), states[:, -1] @ model.items.weight[1:].T
        )
        active = p.ne(0)
        vectors = model(h)[active]
        pos = (vectors * model.items(p[active])).sum(-1).numpy()
        neg = (vectors * model.items(n[active])).sum(-1).numpy()
        expected = (np.logaddexp(0, -pos) + np.logaddexp(0, neg)).mean()
        assert model.sampled_loss(h, p, n).item() == pytest.approx(expected)

    # Item 1 is only context here, not a positive or negative, yet receives a gradient.
    model.sampled_loss(ch[:1], cp[:1], cn[:1]).backward()
    assert model.items.weight.grad is not None
    assert torch.count_nonzero(model.items.weight.grad[1]) > 0
    assert torch.count_nonzero(model.items.weight.grad[0]) == 0

    scores = torch.full((1, 20), -1.0)
    scores[0, [1, 10, 2, 7]] = torch.tensor([0.9, 0.7, 0.4, 0.4])
    metrics = ranking_metrics(scores, target, 10)
    assert metrics == pytest.approx({"ndcg": 1 / np.log2(5), "hit": 1, "count": 1})
