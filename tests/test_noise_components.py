"""Experiment 4's variance components on tiny hand-checked dumps (CPU, no upstream code)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PESO = ROOT / "replication/peso"
Z = norm.ppf(0.975) + norm.ppf(0.8)
TAGS = {
    "pretrain": "QUICK_ws20.msl-1.temp0.8.sflora_pretrain.blocks5.json",
    "lora": "QUICK_ws20.msl-1.temp0.8.sflora_cl_0.0.blocks5.json",
    "peso": "QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5.json",
}


def write(folder, state, block, ranks):
    """One dump: each user's target "t" at the given 1-based rank, or absent (None)."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (user, rank) in enumerate(ranks.items()):
        ranked = [f"x{k}" for k in range(10)]
        if rank is not None:
            ranked[rank - 1] = "t"
        rows.append(json.dumps({"index": i, "user": user, "target": "t", "ranked": ranked}))
    (folder / f"state{state}_block{block}.jsonl").write_text("\n".join(rows) + "\n")


def table2(folder, peso_u2):
    """Pretrain misses u2; single LoRA hits both users at rank 1; PESO's u2 rank is peso_u2."""
    for b in (1, 2, 3):
        write(folder / "lora", 0, b, {"u1": 1, "u2": None})
        write(folder / "lora", b, b, {"u1": 1, "u2": 1})
        write(folder / "peso", b, b, {"u1": 1, "u2": peso_u2})


def upstream(results, gains):
    """Replication seeds whose PESO - single LoRA NDCG@10 is the given gain in every block."""
    for seed, gain in zip((42, 43, 44), gains):
        for method, tag in TAGS.items():
            value = 0.01 + (gain if method == "peso" else 0.0)
            blocks = {f"block_{b}": {"all_prompt_results": [{"ndcg@10": value, "hit@10": value}]}
                      for b in range(5)}  # fmt: skip
            path = results / f"seed{seed}" / "results" / tag
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"all_block_results": blocks}))


def categories(run):
    """Single LoRA's state 0 misses u2 at rank 1; later states hit both; PESO puts u1 second."""
    for b in range(5):
        for s in range(5):
            write(run / "matrix-beam20/lora", s, b, {"u1": 1, "u2": 2 if s == 0 else 1})
        for s in range(1, 5):
            write(run / "matrix-beam20/peso", s, b, {"u1": 2, "u2": 1})


@pytest.fixture
def report(tmp_path):
    release = tmp_path / "release"
    table2(release / "table2-beam-beam10", None)
    table2(release / "table2-sampling-s1-beam10", None)
    table2(release / "table2-sampling-s2-beam10", 1)
    upstream(tmp_path / "results", (0.001, 0.003, 0.002))
    for name in ("cat-a", "cat-b"):
        categories(tmp_path / name)
    out = tmp_path / "noise.json"
    args = [out, release, tmp_path / "results", "--category-runs", tmp_path / "cat-a",
            tmp_path / "cat-b"]  # fmt: skip
    result = subprocess.run(
        [sys.executable, "noise_components.py", *map(str, args)],
        cwd=PESO, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    again = subprocess.run(
        [sys.executable, "noise_components.py", *map(str, args)],
        cwd=PESO, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert again.returncode != 0 and "Refusing to overwrite" in again.stderr
    return json.loads(out.read_text())


def test_item_components_follow_the_hand_computed_variances(report):
    cell = report["items"]["table2_one_seed"]["peso-lora"]["ndcg@10"]
    # Deterministic differences [0, -1] in each block: mean -0.5, per-term variance 0.5 / 2.
    assert cell["deterministic_estimates"] == [-0.5]
    assert cell["users_bc"] == pytest.approx(3 * 0.25 / 9)
    # Draws differ only in u2 ([-1, 0]): across-draw variance 0.5 per term, over n^2 = 4.
    assert cell["sampled_estimates"] == [-0.5, 0.0]
    assert cell["eval_per_user"] == pytest.approx(3 * 0.125 / 9)
    assert cell["eval_across_draws"] == pytest.approx(0.125)
    seeds = cell["seeds_sampled"]
    assert seeds["values"] == pytest.approx([0.001, 0.003, 0.002])
    assert seeds["variance"] == pytest.approx(1e-6)
    assert cell["v_seed"] == pytest.approx(1e-6 - 3 * 0.125 / 9)
    # A negative seed estimate is raised to the interaction floor C = users_bc - B.
    for bound in ("B = 0", "B = users_bc"):
        mde = cell["design"][bound]["mde"]
        assert mde["1 seed(s), sampled"] == pytest.approx(Z * math.sqrt(0.125))
        assert mde["1 seed(s), deterministic"] == pytest.approx(Z * math.sqrt(1 / 12))
    no_shared = cell["design"]["B = 0"]
    room = (0.0012 / Z) ** 2
    assert no_shared["seeds_for_effect_sampled"] == math.ceil(0.125 / room)
    assert cell["design"]["B = users_bc"]["seeds_for_effect_sampled"] is None  # floor > effect
    assert report["items"]["table2_one_seed"]["lora-pretrain"]["ndcg@10"]["eval_per_user"] == 0


def test_mcnemar_counts_flips_between_cells_and_draws(report):
    flips = report["items"]["mcnemar_hit@10"]
    assert flips["deterministic"]["block1"] == {
        "users": 2, "lost": 1, "gained": 0, "net": -1, "mcnemar_p": 1.0}  # fmt: skip
    assert flips["draw table2-sampling-s2-beam10"]["block2"]["lost"] == 0
    assert flips["same weights, two draws (single LoRA)"]["block3"]["net"] == 0


def test_category_seeds_share_users_and_forgetting_is_scored(report):
    cats = report["categories"]
    assert cats["runs"] == ["cat-a", "cat-b"]
    cell = cats["peso-lora"]["hit@1"]
    assert cell["deterministic_estimates"] == [-0.5, -0.5]
    assert cell["users_b"] == pytest.approx(cell["users_bc"])  # identical seeds share all of it
    assert cell["seeds"]["variance"] == 0
    assert cell["design"]["B measured"]["mde_floor_any_seeds"] == pytest.approx(
        Z * math.sqrt(1 / 12)
    )
    forgetting = cats["lora-forgetting"]["hit@1"]
    # State 4 - state j on blocks 0-3: only block 0 changes ([0, 1]).
    assert forgetting["deterministic_estimates"] == [0.125, 0.125]
    assert forgetting["users_bc"] == pytest.approx(0.25 / 16)
