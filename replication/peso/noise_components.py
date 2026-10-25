"""Experiment 4: the smallest Table 2 difference the PESO protocol can detect.

A contrast averages per-user differences between two cells over its terms, each block
weighted equally as Table 2 does: PESO - single LoRA and single LoRA - Pretrain on
blocks 1-3 (state b on block b; Pretrain is single LoRA's state 0), and single LoRA's
forgetting, state 4 - state j on blocks 0-3. For user i, training seed s and evaluation
draw r the difference is d = Delta + a_s + b_i + c_is + e_isr, so the contrast over S
seeds, R draws per seed and the same test users has variance

    (V_seed + E / R) / S + B,    V_seed = var(a) + C,

with every term at the benchmark's own size (n_t users in term t, weight 1 / T each):

- E, evaluation: each user's across-draw variance of d over evaluation seeds of the
  same weights, since upstream's released generation config samples inside beam search;
- B + C, users: the across-user variance of d in one deterministic run. B, the part a
  user keeps across training seeds, is the across-seed covariance of the same users' d;
- V_seed: the between-seed variance of the contrast under deterministic decoding. For
  products only the replication's sampled Table 2 seeds exist, so E is subtracted from
  their spread. The chi-square 95% interval shows how little a few seeds pin it down.

The minimum detectable effect at 80% power and a two-sided 5% level is
2.80 * sqrt(variance). Exact McNemar tests on per-user Hit@10 flips cross-check the
product contrasts. Aggregates only.

    python noise_components.py <new report.json> <release run dir> <replication results> \
        [--item-runs <run dir> ...] [--category-runs <run dir> ...]

The release run dir holds table2-sampling*-beam10 (evaluation seeds of the same weights)
and table2-beam-beam10 (deterministic); item and category run dirs hold deterministic
matrix-beam20 dumps, one training seed each. Needs a checkout with cp4285 on the path.
"""

import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
from summarize import runs_blocks

from cp4285.experiment import flips
from cp4285.experiment.metrics import from_ranks

Z = norm.ppf(0.975) + norm.ppf(0.8)
CONTRASTS = {
    "peso-lora": [(b, ("peso", b), ("lora", b)) for b in (1, 2, 3)],
    "lora-pretrain": [(b, ("lora", b), ("lora", 0)) for b in (1, 2, 3)],
    "lora-forgetting": [(j, ("lora", 4), ("lora", j)) for j in (0, 1, 2, 3)],
}
# The first metric is the one each family's effects of interest are stated in.
FAMILIES = {
    "items": {"metrics": ("ndcg@10", "hit@10"), "effects": {"peso-lora": 0.0012, "lora-pretrain": 0.0010}},
    "categories": {"metrics": ("hit@1", "ndcg@10"), "effects": {"peso-lora": 0.003, "lora-pretrain": 0.003}},
}  # fmt: skip
DESIGNS = [(1, 1), (1, None), (3, 1), (3, None), (10, None)]  # (seeds, draws); None = deterministic


def scores(path, metric):
    """Sorted users and their per-user scores in one dump; the mean matches from_ranks."""
    ranks = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        top = row["ranked"]
        ranks[row["user"]] = top.index(row["target"]) + 1 if row["target"] in top else math.inf
    users = sorted(ranks)
    r = np.array([ranks[u] for u in users], dtype=float)
    k = int(metric.split("@")[1])
    hit = r <= k
    s = np.where(hit, 1 / np.log2(r + 1), 0.0) if metric.startswith("ndcg") else hit.astype(float)
    assert math.isclose(s.mean(), from_ranks(r, (k,))[metric], abs_tol=1e-12)
    return users, s


def differences(folder, contrast, metric):
    """Per-term (users, per-user differences), or None when a dump is missing."""
    out = []
    for block, (ma, sa), (mb, sb) in CONTRASTS[contrast]:
        paths = [Path(folder) / m / f"state{s}_block{block}.jsonl" for m, s in ((ma, sa), (mb, sb))]
        if not all(p.exists() for p in paths):
            return None
        (ua, xa), (ub, xb) = (scores(p, metric) for p in paths)
        if ua != ub:
            raise ValueError(f"{folder}: test users differ in block {block}")
        out.append((ua, xa - xb))
    return out


def estimate(d):
    return statistics.mean(float(x.mean()) for _, x in d)


def user_variance(d):
    return sum(x.var(ddof=1) / len(x) for _, x in d) / len(d) ** 2


def shared_user_variance(d1, d2):
    if [u for u, _ in d1] != [u for u, _ in d2]:
        raise ValueError("Seeds were scored on different test users")
    return (
        sum(float(np.cov(x1, x2)[0, 1]) / len(x1) for (_, x1), (_, x2) in zip(d1, d2))
        / len(d1) ** 2
    )


def eval_variance(draws):
    """Summed per-user across-draw variance, at the benchmark's size."""
    total = 0.0
    for t in range(len(draws[0])):
        if any(d[t][0] != draws[0][t][0] for d in draws):
            raise ValueError("Evaluation draws scored different test users")
        stack = np.stack([d[t][1] for d in draws])
        total += stack.var(axis=0, ddof=1).sum() / stack.shape[1] ** 2
    return total / len(draws[0]) ** 2


def spread(values):
    """Between-seed variance with its chi-square 95% interval (n - 1 degrees of freedom)."""
    v, df = statistics.variance(values), len(values) - 1
    return {"values": values, "variance": v,
            "ci95": [df * v / chi2.ppf(0.975, df), df * v / chi2.ppf(0.025, df)]}  # fmt: skip


def design(v_seed, e, b, c, effect):
    """Minimum detectable effects for DESIGNS and the seeds an effect needs, per decoding.

    V_seed = var(a) + C cannot fall below C, so a smaller (noisy) estimate is raised to C.
    """
    b = max(b, 0.0)
    v_seed = max(v_seed, c - b, 0.0)
    out = {"mde": {}}
    for seeds, draws in DESIGNS:
        var = (v_seed + (e / draws if draws else 0.0)) / seeds + b
        label = f"{seeds} seed(s), {'sampled' if draws else 'deterministic'}"
        out["mde"][label] = Z * math.sqrt(var)
    room = (effect / Z) ** 2 - b if effect else None
    for label, draws in (("sampled", 1), ("deterministic", None)):
        need = v_seed + (e / draws if draws else 0.0)
        out[f"seeds_for_effect_{label}"] = (
            None if room is None or room <= 0 else max(1, math.ceil(need / room))
        )
    out["mde_floor_any_seeds"] = Z * math.sqrt(b)
    return out


def family(name, deterministic, draws=(), sampled_seeds=None):
    """Components and designs for every contrast and metric of one family."""
    spec, out = FAMILIES[name], {}
    for contrast in CONTRASTS:
        for metric in spec["metrics"]:
            det = [d for d in (differences(f, contrast, metric) for f in deterministic) if d]
            if not det:
                continue
            cell = {"deterministic_estimates": [estimate(d) for d in det]}
            cell["users_bc"] = statistics.mean(user_variance(d) for d in det)
            pairs = [shared_user_variance(a, b) for a, b in combinations(det, 2)]
            cell["users_b"] = statistics.mean(pairs) if pairs else None
            sampled = [d for d in (differences(f, contrast, metric) for f in draws) if d]
            e = eval_variance(sampled) if len(sampled) > 1 else 0.0
            if sampled:
                cell["sampled_estimates"] = [estimate(d) for d in sampled]
                cell["eval_per_user"] = e
                cell["eval_across_draws"] = statistics.variance(cell["sampled_estimates"])
            if len(det) > 1:  # deterministic seeds
                cell["seeds"] = spread(cell["deterministic_estimates"])
                v_seed = cell["seeds"]["variance"]
            elif sampled_seeds and contrast in sampled_seeds and metric in sampled_seeds[contrast]:
                cell["seeds_sampled"] = spread(sampled_seeds[contrast][metric])
                v_seed = cell["seeds_sampled"]["variance"] - e
            else:  # one deterministic seed: the user term alone, a floor on any design
                v_seed = None
                cell["z_users_only"] = cell["deterministic_estimates"][0] / math.sqrt(
                    cell["users_bc"]
                )
                cell["mde_users_only"] = Z * math.sqrt(cell["users_bc"])
            cell["v_seed"] = v_seed
            effect = spec["effects"].get(contrast) if metric == spec["metrics"][0] else None
            if v_seed is not None:  # one seed leaves B between none and all of the user term
                bounds = {"B measured": cell["users_b"]} if pairs else {
                    "B = 0": 0.0, "B = users_bc": cell["users_bc"]}  # fmt: skip
                cell["design"] = {
                    k: design(v_seed, e, b, cell["users_bc"], effect) for k, b in bounds.items()
                }
            cell["effect_of_interest"] = effect
            out.setdefault(contrast, {})[metric] = cell
    return out


def upstream_seeds(results):
    """{contrast: {metric: per-seed Table 2 values}} from the replication's upstream runs."""
    runs = runs_blocks(sorted(Path(results).glob("seed4?")))
    pairs = {"peso-lora": ("peso", "lora"), "lora-pretrain": ("lora", "pretrain")}
    out = {}
    for contrast, (a, c) in pairs.items():
        blocks = [b for b, _, _ in CONTRASTS[contrast]]
        out[contrast] = {
            metric: [
                statistics.mean(sa[b][metric] - sc[b][metric] for b in blocks)
                for sa, sc in zip(runs[a], runs[c])
            ]
            for metric in FAMILIES["items"]["metrics"]
        }
    return out


def mcnemar(deterministic, draws):
    """Hit@10 flips: PESO against single LoRA per block, and two draws of the same cell."""
    out = {}
    for label, folder in [
        ("deterministic", deterministic),
        *((f"draw {Path(f).name}", f) for f in draws),
    ]:
        lora, peso = flips.load(Path(folder) / "lora"), flips.load(Path(folder) / "peso")
        out[label] = {f"block{b}": flips.flips(lora[b, b], peso[b, b])["all"] for b in (1, 2, 3)}
    if len(draws) > 1:
        first, second = (flips.load(Path(f) / "lora") for f in draws[:2])
        out["same weights, two draws (single LoRA)"] = {
            f"block{b}": flips.flips(first[b, b], second[b, b])["all"] for b in (1, 2, 3)
        }
    keep = ("users", "lost", "gained", "net", "mcnemar_p")
    return {k: {b: {f: v[f] for f in keep} for b, v in cells.items()} for k, cells in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("release", help="run dir with table2-sampling*-beam10, table2-beam-beam10")
    parser.add_argument("replication", help="results dir with seed4?/results (upstream runs)")
    parser.add_argument("--item-runs", nargs="*", default=[], help="deterministic matrix-beam20")
    parser.add_argument(
        "--category-runs", nargs="*", default=[], help="deterministic matrix-beam20"
    )
    args = parser.parse_args()
    output, release = Path(args.output), Path(args.release)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    draws = sorted(str(p) for p in release.glob("table2-sampling*-beam10"))
    deterministic = release / "table2-beam-beam10"
    matrices = [Path(r) / "matrix-beam20" for r in args.item_runs]
    report = {
        "z_80_power_5pct_two_sided": Z,
        "items": {
            "table2_one_seed": family("items", [deterministic], draws, upstream_seeds(args.replication)),
            "evaluation_draws": [Path(f).name for f in draws],
            "mcnemar_hit@10": mcnemar(deterministic, draws),
        },
    }  # fmt: skip
    if [m for m in matrices if m.exists()]:
        report["items"]["matrix_seeds"] = family("items", [m for m in matrices if m.exists()])
    if args.category_runs:
        folders = [Path(r) / "matrix-beam20" for r in args.category_runs]
        report["categories"] = {"runs": [Path(r).name for r in args.category_runs],
                                **family("categories", folders)}  # fmt: skip
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    out = main()
    for contrast, metrics in out["items"]["table2_one_seed"].items():
        for metric, cell in metrics.items():
            print(
                "items",
                contrast,
                metric,
                {k: cell.get(k) for k in ("users_bc", "eval_per_user", "v_seed")},
            )
