"""The category language model learner on a tiny random Llama (CPU, no download)."""

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from cp4285.models.category_lm import CategoryLMLearner

NAMES = ["Guitars", "Drums", "Microphones", "Strings"]


class ByteTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode()]


def tiny_model():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=259, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
    )  # fmt: skip
    return transformers.LlamaForCausalLM(config)


def learner():
    return CategoryLMLearner(tiny_model(), ByteTokenizer(), NAMES, learning_rate=1e-2,
                             continuation_learning_rate=1e-2, rank=4, score_batch=2)  # fmt: skip


def unpacked(learner, history, column):
    """log P(name | prompt) from one ordinary causal sequence."""
    prompt, name = learner.prompt(history), learner.name_ids[column]
    learner.model.eval()
    with torch.no_grad():
        logp = learner.model(input_ids=torch.tensor([prompt + name])).logits.log_softmax(-1)[0]
    return sum(logp[len(prompt) - 1 + t, name[t]].item() for t in range(len(name)))


def test_packed_scores_equal_one_sequence_per_candidate():
    model = learner()
    model.train([{"history": [0, 1], "target": 2}])  # LoRA B starts at zero; make it count
    queries = [[0, 1], [2], [3, 3, 1]]
    scores = model.score(queries)
    for query, row in zip(queries, scores):
        for column in range(len(NAMES)):
            assert row[column].item() == pytest.approx(unpacked(model, query, column), abs=1e-4)
    assert torch.allclose(model.score(queries, [2, 0]), scores[:, [2, 0]], atol=1e-6)
    assert torch.equal(model.score(queries), scores)  # scoring changes nothing


def test_training_raises_the_target():
    model = learner()
    before = model.score([[0, 1]])[0]
    losses = [model.train([{"history": [0, 1], "target": 3}]) for _ in range(25)]
    gain = model.score([[0, 1]])[0] - before
    # A frozen random base limits how far rank-4 LoRA moves it; the target must gain most.
    assert losses[-1] < losses[0] and gain.argmax() == 3 and gain[3] > 2
    assert model.describe()["steps"] == 25


def test_new_stream_freezes_old_adapters_and_rolls_back():
    model = learner()
    for _ in range(5):
        model.train([{"history": [0], "target": 1}])
    model.begin_continuation()
    with pytest.raises(RuntimeError):
        model.begin_continuation()
    reference = model.score([[0], [2]])
    old = {k: v.clone() for k, v in model.model.named_parameters() if ".a0." in k}
    model.add_stream("s1")
    for _ in range(10):
        model.train([{"history": [0], "target": 2}])
    now = dict(model.model.named_parameters())
    assert old and all(torch.equal(v, now[k]) for k, v in old.items())
    assert not torch.allclose(model.score([[0], [2]]), reference)
    model.set_active(["a0"])
    assert torch.allclose(model.score([[0], [2]]), reference, atol=1e-6)


def test_state_round_trip_restores_scores_and_optimizer():
    model = learner()
    for _ in range(3):
        model.train([{"history": [1], "target": 0}])
    model.begin_continuation()
    model.add_stream("s1")
    model.train([{"history": [1], "target": 2}])
    saved, scores = model.state(), model.score([[1], [3]])
    other = learner()
    other.load_state(saved)
    assert other.adapters == ["a0", "s1"] and other.phase == "continuation"
    assert torch.allclose(other.score([[1], [3]]), scores, atol=1e-6)
    model.train([{"history": [1], "target": 3}])
    other.train([{"history": [1], "target": 3}])
    assert torch.allclose(other.score([[1]]), model.score([[1]]), atol=1e-5)


def test_dropping_a_stream_restores_the_earlier_stack():
    model = learner()
    for _ in range(3):
        model.train([{"history": [0], "target": 1}])
    model.begin_continuation()
    reference = model.score([[0], [3]])
    model.add_stream("s1")
    for _ in range(5):
        model.train([{"history": [0], "target": 2}])
    model.drop("s1")
    assert model.adapters == ["a0"] and torch.allclose(
        model.score([[0], [3]]), reference, atol=1e-6
    )
    model.add_stream("s2")  # later streams build without the dropped adapter
    model.train([{"history": [0], "target": 3}])
    assert model.adapters == ["a0", "s2"]
    with pytest.raises(ValueError):
        model.drop("a0")
