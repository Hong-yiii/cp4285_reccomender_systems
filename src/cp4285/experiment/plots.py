"""Small notebook-ready SVG plots derived only from saved experiment records."""

import json
from html import escape
from pathlib import Path


class SVG(str):
    """Use Jupyter's native rich-display protocol without importing IPython."""

    def _repr_svg_(self) -> str:
        return str(self)


def plot_results(results_path: str | Path, *, task: str, metric: str = "ndcg") -> SVG:
    """Absolute quality versus update blocks; return SVG without a plotting dependency."""
    if metric not in {"ndcg", "hit", "recall"}:
        raise ValueError("Choose ndcg, hit or recall")
    result = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = [row for row in result["records"] if row["task"] == task]
    if not rows:
        raise ValueError(f"No saved results for task {task!r}")
    groups = sorted({row["condition"] for row in rows})
    last = max(1, max(row["step"] for row in rows))
    height = 350 + 22 * len(groups)
    title = f"{task}: {metric.upper()}@{result['manifest']['k']}"
    provenance = result["manifest"]["provenance"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {height}" role="img">',
        f"<title>{escape(title)}</title><desc>{escape(provenance)}</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="60" y="25" font-size="18">{escape(title)}</text>',
        f'<text x="60" y="46" font-size="10">{escape(provenance[:110])}</text>',
        '<path d="M60 65 V285 H710" fill="none" stroke="black"/>',
    ]
    for value in (0, 0.25, 0.5, 0.75, 1):
        y = 285 - 220 * value
        parts.append(f'<text x="20" y="{y + 4}" font-size="12">{value:g}</text>')
    for step in sorted({row["step"] for row in rows}):
        x = 60 + 650 * step / last
        parts.append(f'<text x="{x}" y="302" font-size="12">{step}</text>')
    parts.append('<text x="280" y="325" font-size="12">Update block (not matched compute)</text>')
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333", "#946700")
    for index, condition in enumerate(groups):
        color = colors[index % len(colors)]
        selected = sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["step"])
        points = " ".join(
            f"{60 + 650 * r['step'] / last},{285 - 220 * r[metric]}" for r in selected
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row in selected:
            x, y = 60 + 650 * row["step"] / last, 285 - 220 * row[metric]
            label = f"{condition}; block {row['step']}; value {row[metric]:.6f}"
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}">')
            parts.append(f"<title>{escape(label)}</title></circle>")
        parts.append(
            f'<text x="60" y="{350 + 22 * index}" fill="{color}" font-size="12">'
            f"{escape(condition)}</text>"
        )
    return SVG("\n".join([*parts, "</svg>"]))
