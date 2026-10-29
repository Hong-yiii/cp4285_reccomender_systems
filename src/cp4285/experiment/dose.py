"""Dose-response and critical-mass summary for the corrupted-stream study (issue #19).

Each attacked run is paired with the clean run of the same domain, seed, condition and
pushed category. Per-user ranks come from the runner's `--dumps` files, so the summary
works on paired differences rather than aggregate cells. The primary threshold endpoint
is the mean rank change of the pushed category among genuine users, because top-10 share
saturates: a category already in 77% of clean top-10 lists cannot show a threshold there.

The threshold is fixed before running: the smallest dose whose mean rank change is at
least one rank in the promoted direction and whose per-seed bootstrap interval lies
entirely below zero in at least two of three seeds. A flat or non-monotone curve is a
valid result and is reported as "no threshold identifiable" rather than smoothed away.
"""

import json
import random
from itertools import pairwise
from pathlib import Path

from .utils import save_json

DEFAULT_DOSES = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)


def load_run(path):
    """Read a run's metrics.json from its folder or from the file itself."""
    path = Path(path)
    metrics = path / "metrics.json" if path.is_dir() else path
    return json.loads(metrics.read_text()), metrics.parent


def load_rows(run_dir, cell):
    """Per-user dump rows for one scored cell."""
    path = Path(run_dir) / "dumps" / f"{cell}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dumps for {cell}: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pair(attacked_rows, clean_rows, push):
    """Join two runs on user, drop the pushed category's own targets, keep top sets."""
    attacked = {r["user"]: r for r in attacked_rows if r["target"] != push}
    clean = {r["user"]: r for r in clean_rows if r["target"] != push}
    users = sorted(set(attacked) & set(clean))
    if not users:
        raise ValueError("No genuine user is present in both runs")
    return {
        "users": users,
        "deltas": [attacked[u]["rank"] - clean[u]["rank"] for u in users],
        "attacked_rank": [attacked[u]["rank"] for u in users],
        "clean_rank": [clean[u]["rank"] for u in users],
        "attacked_top": [attacked[u].get("top") for u in users],
        "clean_top": [clean[u].get("top") for u in users],
    }


def bootstrap_mean(values, seed=42, draws=2000):
    """Percentile interval for the mean, resampling users with replacement."""
    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                   for _ in range(draws))  # fmt: skip
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def endpoints(pairing, seed=42, draws=2000):
    """Rank change, top-K shares, churn and normalised gain for one paired run."""
    deltas = pairing["deltas"]
    attacked, clean = pairing["attacked_rank"], pairing["clean_rank"]
    n = len(deltas)
    top_sets = pairing["attacked_top"], pairing["clean_top"]
    churn = None
    if all(top is not None for top in top_sets[0]) and all(top is not None for top in top_sets[1]):
        churn = sum(set(a) != set(c) for a, c in zip(*top_sets)) / n
    gains = [(c - a) / (c - 1) for a, c in zip(attacked, clean) if c > 1]
    low, high = bootstrap_mean(deltas, seed=seed, draws=draws)
    return {
        "users": n,
        "mean_rank": sum(deltas) / n,
        "ci_low": low,
        "ci_high": high,
        "top1": sum(r <= 1 for r in attacked) / n,
        "top5": sum(r <= 5 for r in attacked) / n,
        "top10": sum(r <= 10 for r in attacked) / n,
        "clean_top5": sum(r <= 5 for r in clean) / n,
        "churn": churn,
        "normalised_gain": sum(gains) / len(gains) if gains else None,
    }


def curve(records, draws=2000):
    """Per-record endpoints and per-dose aggregates; records carry dose, seed and pairing."""
    if not records:
        raise ValueError("No paired runs to summarise")
    points = []
    for record in records:
        point = {"dose": record["dose"], "seed": record["seed"], "label": record["label"],
                 **endpoints(record["pairing"], seed=record["seed"] + 17, draws=draws)}  # fmt: skip
        points.append(point)
    doses = []
    for dose in sorted({p["dose"] for p in points}):
        rows = [p for p in points if p["dose"] == dose]
        doses.append({
            "dose": dose,
            "runs": len(rows),
            "mean_rank": sum(r["mean_rank"] for r in rows) / len(rows),
            "ci_low": min(r["ci_low"] for r in rows),
            "ci_high": max(r["ci_high"] for r in rows),
            "seed_ci_below_zero": sum(r["ci_high"] < 0 for r in rows),
            "top1": sum(r["top1"] for r in rows) / len(rows),
            "top5": sum(r["top5"] for r in rows) / len(rows),
            "top10": sum(r["top10"] for r in rows) / len(rows),
            "clean_top5": sum(r["clean_top5"] for r in rows) / len(rows),
            "churn": (sum(r["churn"] for r in rows if r["churn"] is not None)
                      / max(1, sum(r["churn"] is not None for r in rows))
                      if any(r["churn"] is not None for r in rows) else None),
        })
    return {"points": points, "doses": doses}


def threshold(result, step=1.0, top5=0.5):
    """Apply the pre-registered rule to a curve result."""
    doses = sorted(result["doses"], key=lambda d: d["dose"])
    means = [d["mean_rank"] for d in doses]
    non_monotone = any(b > a for a, b in pairwise(means))
    primary = next((d for d in doses
                    if d["mean_rank"] <= -step and d["seed_ci_below_zero"] >= 2), None)  # fmt: skip
    secondary = next((d for d in doses if d["top5"] >= top5), None)
    return {
        "rule": f"mean rank change <= -{step} with at least 2 runs' CI entirely below 0",
        "dose": None if primary is None else primary["dose"],
        "identifiable": primary is not None,
        "mean_rank": None if primary is None else primary["mean_rank"],
        "top5_dose": None if secondary is None else secondary["dose"],
        "non_monotone": non_monotone,
        "tested": [d["dose"] for d in doses],
    }


def summary(attacked_paths, clean_paths, cell="state1_block1", draws=2000,
            step=1.0, top5=0.5):
    """Pair attacked runs with clean references and return the curve and threshold."""
    clean = {}
    for path in clean_paths:
        metrics, _ = load_run(path)
        cfg = metrics["config"]
        clean[(cfg.get("seed"), cfg.get("push"), cfg.get("condition"))] = path
    records = []
    for path in attacked_paths:
        metrics, run_dir = load_run(path)
        cfg = metrics["config"]
        key = (cfg.get("seed"), cfg.get("push"), cfg.get("condition"))
        if key not in clean:
            raise ValueError(f"No clean reference for {path} with seed/push/condition {key}")
        _clean_metrics, clean_dir = load_run(clean[key])
        pairing = pair(load_rows(run_dir, cell), load_rows(clean_dir, cell), cfg.get("push"))
        records.append({"dose": cfg["alpha"], "seed": cfg["seed"],
                        "label": Path(path).name, "pairing": pairing})  # fmt: skip
    result = curve(records, draws=draws)
    result["threshold"] = threshold(result, step=step, top5=top5)
    result["cell"] = cell
    result["clean_runs"] = len(clean)
    return result


def markdown(result):
    """A compact table plus the threshold verdict."""
    lines = [f"### Dose response on `{result['cell']}`", "",
             "| dose | mean rank change | 95% CI (min, max) | top-5 | clean top-5 | churn | runs |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in sorted(result["doses"], key=lambda d: d["dose"]):
        churn = "n/a" if row.get("churn") is None else f"{row['churn']:.2f}"
        lines.append(
            f"| {row['dose']:.4f} | {row['mean_rank']:+.2f} | "
            f"({row['ci_low']:+.2f}, {row['ci_high']:+.2f}) | "
            f"{row['top5']:.2f} | {row.get('clean_top5', 0):.2f} | {churn} | "
            f"{row.get('runs', 1)} |"
        )
    verdict = result["threshold"]
    if verdict["identifiable"]:
        lines += ["", (f"**Threshold:** {verdict['dose']:.4f} "
                       f"(mean rank {verdict['mean_rank']:+.2f}).")]
    else:
        tested = ", ".join(f"{d:.4f}" for d in verdict["tested"])
        lines += ["", f"**No threshold identifiable** in the tested range ({tested})."]
    if verdict["non_monotone"]:
        lines += ["", "The curve is non-monotone; the first crossing is reported, not smoothed."]
    if verdict["top5_dose"] is not None:
        lines += ["", f"Top-5 reaches 50% at dose {verdict['top5_dose']:.4f}."]
    return "\n".join(lines) + "\n"


def save(result, path):
    """Create-only JSON output."""
    save_json(path, result)
