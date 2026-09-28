import csv
import json

import pytest
import torch

from cp4285.cli import demo
from cp4285.common.data import audit
from cp4285.common.utils import millis
from cp4285.neural.data import prepare
from cp4285.neural.model import IMPLEMENTATION, UPSTREAM_COMMIT, SASRec
from cp4285.neural.pilot import ranking_metrics, tensors


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
        writer.writerows(rows)
    return path


def test_audit_preserves_identity_and_reports_duplicates_and_ties(tmp_path):
    t = millis("2020-01-01")
    a = write_csv(
        tmp_path / "a.csv",
        [("same", "a", 5, t), ("same", "a", 5, t), ("same", "b", 4, t), ("other", "c", 1, t + 1)],
    )
    b = write_csv(tmp_path / "b.csv", [("same", "c", 5, t), ("third", "d", 5, t)])
    report = audit(a, b, "2021-01-01")
    assert report["overlap"] == {"user_id": 1, "parent_asin": 1}
    assert report["domains"]["a"]["exact_duplicate_extra_rows"] == 1
    assert report["domains"]["a"]["tied_rows_after_rating_filter_and_dedup"] == 2
    assert report["domains"]["a"]["after_tie_removal"]["users"] == 1
    positive = audit(a, b, "2021-01-01", 4)
    assert positive["domains"]["a"]["after_tie_removal"]["users"] == 0


def test_seconds_are_rejected_instead_of_silent_bad_temporal_split(tmp_path):
    f = write_csv(tmp_path / "seconds.csv", [("u", "i", 5, 1577836800)])
    with pytest.raises(ValueError, match="milliseconds"):
        audit(f, f, "2021-01-01")


def test_batch_composition_does_not_change_fixed_position_scores():
    torch.manual_seed(1)
    model = SASRec(20, 4, hidden=8, heads=1, layers=1, dropout=0).eval()
    alone = torch.tensor([[0, 0, 1, 2]])
    batched = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]])
    with torch.no_grad():
        assert torch.allclose(model.scores(alone)[0], model.scores(batched)[0], atol=1e-5)
    with pytest.raises(ValueError, match="left-padded"):
        model(torch.tensor([[1, 2, 0, 0]]))
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.zeros((1, 4), dtype=torch.long))


def test_ranking_metrics_known_ranks_and_ties():
    scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    metrics = ranking_metrics(scores, torch.tensor([1, 2]), 1)
    assert metrics == {"ndcg": 1.0, "hit": 1.0, "count": 2}


def test_end_to_end_checkpoint_and_holdout_contract(tmp_path):
    output = tmp_path / "demo"
    result = demo(output)
    assert result["synthetic"] is True
    assert result["checkpoint_reload_max_difference"] == 0
    points = result["retention"]
    assert len(points) == 3
    assert all(p["frozen"] == points[0]["frozen"] for p in points)
    data = json.loads((output / "prepared.json").read_text())
    reverse = {i + 1: item for i, item in enumerate(data["item_ids"])}
    cutoff = millis(data["metadata"]["config"]["initial_cutoff"])
    for row in data["train"]:
        assert row["timestamp"] < cutoff
    for row in data["continuation"]:
        assert row["timestamp"] >= cutoff
    # This fixture has no repeats per user, so a held-out item appearing in a training
    # prefix would unambiguously identify leakage of the held-out event.
    for held in data["validation"] + data["retention"]:
        for row in data["train"] + data["continuation"]:
            if row["user"] == held["user"]:
                assert held["target"] != row["target"]
                assert held["target"] not in row["history"]
    assert len(reverse) == data["metadata"]["items"]
    initial = torch.load(output / "pilot/initial.pt", weights_only=True)
    continued = torch.load(output / "pilot/cycle-2.pt", weights_only=True)
    assert initial["implementation"] == continued["implementation"] == IMPLEMENTATION
    assert result["upstream_commit"] == UPSTREAM_COMMIT
    assert initial["model_args"]["hidden"] == 50
    assert initial["model_args"]["layers"] == 2
    assert initial["model_args"]["max_length"] == 50
    assert continued["stream_cursor"] == 32
    assert continued["optimizer"]["state"]
    assert continued["optimizer"]["param_groups"][0]["betas"] == (0.9, 0.98)
    assert any(not torch.equal(v, continued["model"][k]) for k, v in initial["model"].items())
    with pytest.raises(FileExistsError):
        demo(output)


def test_cohort_selection_does_not_require_future_activity(tmp_path):
    # Reuse the demo source, then remove every later event for one eligible user.
    base = tmp_path / "source"
    demo(base)
    data = json.loads((base / "prepared.json").read_text())
    p = base / "Electronics.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    rows = [
        r
        for r in rows
        if not (r["user_id"] == "synthetic-user-0" and int(r["timestamp"]) >= millis("2021-01-01"))
    ]
    write_csv(
        p, [tuple(r[k] for k in ("user_id", "parent_asin", "rating", "timestamp")) for r in rows]
    )
    prepared = prepare(p, data["metadata"]["config"])
    assert any(r["user"] == "synthetic-user-0" for r in prepared["retention"])


def test_tensor_prefixes_are_left_padded_to_fixed_width():
    h, t = tensors(
        [{"history": [1, 2, 3, 4], "target": 5}, {"history": [4], "target": 5}], "cpu", 3
    )
    assert h.tolist() == [[2, 3, 4], [0, 0, 4]]
    assert t.tolist() == [5, 5]
