import numpy as np
import pytest

from cp4285.experiment.metrics import from_ranks, selection_key


def test_hand_computed_ranks():
    m = from_ranks([1, 3, 7, 30], ks=(5, 10))
    assert m["hit@5"] == 0.5
    assert m["hit@10"] == 0.75
    assert m["recall@10"] == m["hit@10"]
    assert m["ndcg@5"] == pytest.approx((1 + 1 / np.log2(4)) / 4)
    assert m["ndcg@10"] == pytest.approx((1 + 1 / np.log2(4) + 1 / 3) / 4)
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3 + 1 / 7) / 4)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 7 + 1 / 30) / 4)


def test_selection_prefers_hit_then_ndcg():
    a = {"hit@10": 0.5, "ndcg@10": 0.1}
    b = {"hit@10": 0.5, "ndcg@10": 0.2}
    c = {"hit@10": 0.6, "ndcg@10": 0.0}
    assert max([a, b, c], key=selection_key) is c
    assert max([a, b], key=selection_key) is b


def test_empty_ranks_rejected():
    with pytest.raises(ValueError):
        from_ranks([])
