"""Handover tour: what the project found, where each claim comes from, and a live recheck.

Read-only and stdlib-only. It reads committed files under `replication/peso/results/` and
`docs/`, so it works on a fresh clone without `data/`, `runs/` or `local/`. It writes nothing.

    uv run python replication/peso/handover.py           # whole tour
    uv run python replication/peso/handover.py --step    # pause between sections
    uv run python replication/peso/handover.py --only 4  # one section

Each recomputed number is compared with the figure quoted on the STePS page draft. A claim that
lives only in a written table is quoted from its doc with the line number. The exit code is 1 if
any check differs or any quoted line has moved, so the tour also works as a drift check.
"""

import argparse
import json
import statistics as st
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "replication" / "peso" / "results"
WIDTH = 100
problems: list[str] = []
context = {"where": ""}


def heading(n: int, title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"{n}. {title}")
    print("=" * WIDTH)


def say(text: str) -> None:
    for para in textwrap.dedent(text).strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), WIDTH))
        print()


def check(label: str, got: float, quoted: float, tol: float = 0.006, unit: str = "") -> None:
    ok = abs(got - quoted) <= tol
    mark = "ok" if ok else "DIFFERS"
    print(f"  [{mark:>7}] {label:<58} recomputed {got:+.4g}{unit}  page says {quoted:+.4g}{unit}")
    if not ok:
        problems.append(f"{context['where']}{label.strip()}: {got:+.4g} against {quoted:+.4g}")


def quote(rel: str, needle: str) -> None:
    path = ROOT / rel
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            text = line.strip()
            print(f"  {rel}:{i}")
            print(textwrap.indent(textwrap.fill(text, WIDTH - 6), "      "))
            return
    print(f"  [MISSING] {rel}: no line contains {needle!r}")
    problems.append(f"{rel}: {needle}")


def mean(xs) -> float:
    return st.fmean(xs)


# --------------------------------------------------------------------------------------------


def s0_orientation() -> None:
    heading(0, "What this project is, and what state the claims are in")
    say(
        """
        CP4285 group project, branded Forget-Me-Not for STePS. The question: an online shop
        updates its LLM recommender with each new batch of reviews instead of retraining. Does it
        forget older behaviour, how far can a small poisoned batch move it, and can the damage be
        undone?

        Three strands of evidence, in the order they happened. (A) A replication of PESO
        (ICLR 2026), a continual LLM recommender. (B) A granularity ladder asking where forgetting
        shows up. (C) A poisoning study: inject fake or edited data into one update and track the
        target through later clean updates.

        Status matters more than the numbers. The attack catalogue and study design are recorded
        as agent proposals awaiting team sign-off (CONTEXT.md). The STePS page text is a
        user-requested draft that is not team-approved and is not yet on UVENTS. Every result is
        category-level, from one model size (Llama-3.2-1B with LoRA adapters), and three seeds.
        """
    )


def s1_data() -> None:
    heading(1, "The data everything rests on")
    say(
        """
        PESO's authors released Amazon reviews already cut into five time blocks for three
        catalogues: Instruments, Video Games and CDs & Vinyl. Block 0 trains the starting model;
        blocks 1-4 are the later updates. Upstream has no licence, so the blocks live only in
        ignored local folders and on the NUS SoC cluster. Nothing raw is in git.

        For the attack study each product is renamed to its level-2 Amazon subcategory, so the
        model recommends categories. That is why Hit@10 sits near 1.0: the target space is 30,
        43 or 59 labels. The pushed target is chosen by rule, the category nearest a 1.3% share of
        test targets. Quoted from the docs:
        """
    )
    quote("docs/CONTINUED-TRAINING.md", "| Instruments | 30 | 50,023 / 3,486 |")
    quote("docs/CONTINUED-TRAINING.md", "No pushed category is chosen by hand.")
    print()
    say("What a fresh clone has (committed, metrics only) and what it lacks:")
    for rel in ("summary-3seeds.json", "ladder/ladder-2026-10-24.json", "grid/lattice.json"):
        p = RES / rel
        print(f"  {'present' if p.exists() else 'ABSENT':>7}  replication/peso/results/{rel}")
        if not p.exists():
            problems.append(f"missing {rel}")
    print("   absent  raw blocks, model weights, per-user dumps, November run folders (cluster)")


def s2_peso() -> None:
    heading(2, "Finding A: PESO's reported gain does not reproduce")
    say(
        """
        Source: upstream PESO code at commit edbec5f on the released Instruments blocks, seeds
        42-44 (Runpod and Modal A100s, 20 October). The per-seed metric files and their summary
        are committed with SHA-256 hashes. Recomputed now from summary-3seeds.json, own-block
        NDCG@10 over blocks 1-3, PESO minus a plain evolving LoRA, paired by seed:
        """
    )
    s = json.loads((RES / "summary-3seeds.json").read_text())
    key = "blocks 1-3 (paper)"
    peso = s["peso"]["averages"][key]["ndcg@10"]["seeds"]
    lora = s["lora"]["averages"][key]["ndcg@10"]["seeds"]
    diff = [p - q for p, q in zip(peso, lora)]
    check("PESO - LoRA, mean of 3 seeds", mean(diff), -0.0001, tol=0.00005)
    check("PESO - LoRA, seed SD", st.stdev(diff), 0.0002, tol=0.00007)
    print("  paper claims +0.0012")
    say(
        """
        The likely mechanism, measured from checkpoints: the released penalty is about 1e-6
        against a training loss near 2, so PESO trains almost exactly like plain LoRA.
        """
    )
    quote("docs/PESO-REPLICATION.md", "**Value:** 0.7–1.3 × 10⁻⁶")


def s3_ladder() -> None:
    heading(3, "Finding B: forgetting shows up at the product level only")
    say(
        """
        Source: the same Instruments events labelled at 30, 75 and 299 categories and as raw
        products (24 October). Drift is the relative change in NDCG@10 on an older block after
        four more updates (plain LoRA), range over seeds. Recomputed from ladder-2026-10-24.json:
        """
    )
    d = json.loads((RES / "ladder" / "ladder-2026-10-24.json").read_text())["rungs"]
    page = {"level 2": (-0.4, -0.1), "level 3": (-0.7, -0.4), "products": (-37, -32)}
    for rung, r in d.items():
        v = [100 * x["ndcg@10"]["lora_drift_after_k"]["4"] for x in r["seeds"]]
        labels = r["data"]["labels"]
        print(f"  {rung:<9} {labels:>6} labels  drift {min(v):+.1f}% to {max(v):+.1f}%")
        if rung in page:
            check(f"  {rung} drift, low end", min(v), page[rung][0], tol=0.5, unit="%")
            check(f"  {rung} drift, high end", max(v), page[rung][1], tol=0.5, unit="%")
    js = {k: mean(r["data"]["js_to_previous_block"].values()) for k, r in d.items()}
    check("block-to-block shift, products / level 4", js["products"] / js["level 4"], 28, tol=1)
    say(
        """
        Soft spot: products also shift about 28 times more between blocks, so these runs cannot
        say whether forgetting comes from label granularity or from the shift. Products have two
        seeds.
        """
    )


def s4_attacks() -> None:
    heading(4, "Finding C: the poisoning study (committed grid, fully recheckable)")
    say(
        """
        Source: grid/lattice.json, written by replication/peso/grid_lattice.py from the
        24-27 October run folders. One row per attacked run, paired with the clean run of the same
        catalogue, seed and adapter layout. Rows below use isolated adapters (a fresh frozen
        adapter per update), 9 runs each (3 catalogues x 3 seeds). A positive number means the
        target category climbed that many places.

        Plain names: re-listing = relabel 10% of products into the target; edited histories =
        hijack real users' last item; copied fakes = clone shills; crude fakes = bandwagon shills.
        """
    )
    runs = json.loads((RES / "grid" / "lattice.json").read_text())["runs"]
    iso = [r for r in runs if r["condition"] == "isolated"]
    page = {  # family: (gain, after 3 clean, late removal, AUC)
        "relabel": ("Re-list 10% of products", 7.68, 1.18, -9.11, None),
        "hijack@3%": ("Edit real histories, 3%", 4.80, 0.80, -4.57, 0.73),
        "clone@3%": ("Copied fake shoppers, 3%", 2.23, 0.26, -1.84, 0.70),
        "bandwagon@3%": ("Crude fake shoppers, 3%", 2.14, 0.13, -1.89, 1.00),
    }
    for fam, (name, gain, resid, late, auc) in page.items():
        rows = [r for r in iso if r["family"] == fam]
        g = mean(-r["spike_push_rank"] for r in rows)
        res = mean(-r["residue_push_rank"] for r in rows)
        rb = [-r["rollback_vs_clean_mean"] for r in rows if r["rollback_vs_clean_mean"] is not None]
        below = sum(x < 0 for x in rb)
        print(f"\n  {name}  ({len(rows)} runs)")
        context["where"] = f"{name}: "
        check("places gained right after", g, gain)
        check("still gained after 3 clean updates", res, resid)
        print(f"  {'':10}washed out: {100 * (1 - res / g):.0f}% of the gain")
        check(f"after late removal ({below}/{len(rb)} runs below clean)", mean(rb), late)
        if auc is not None:
            check("fraud-check AUC", mean(r["separability_auc"] for r in rows), auc, tol=0.006)
        else:
            print(f"  {'':10}no fake accounts, so no fraud-check AUC")

    say(
        """

        Why edits beat fakes: count the target-ending examples each attack injects (seed 42,
        isolated). Every edited example ends in the target; half of a fake shopper's do.
        """
    )
    for dom in ("ins", "vg", "cd"):
        row = {r["family"]: r for r in iso if r["seed"] == 42 and r["domain"] == dom}
        bw, hj, rl = row["bandwagon@3%"], row["hijack@3%"], row["relabel"]
        print(
            f"  {dom:<3} fakes {bw['pushed_examples']:>6} of {bw['total_examples']:>4}"
            f"   edits {hj['pushed_examples']:>4} of {hj['total_examples']:>4}"
            f"   re-listing {rl['pushed_examples']:>4}"
            f" ({rl['pushed_examples'] / bw['pushed_examples']:.1f}x fakes)"
        )

    say(
        """

        The comic and animation follow one real case: re-listing on Instruments, mean of 3 seeds.
        The speech-bubble jokes and the category order drawn are invented.
        """
    )
    context["where"] = ""
    ins = [r for r in iso if r["family"] == "relabel" and r["domain"] == "ins"]
    check("comic: gain right after", mean(-r["spike_push_rank"] for r in ins), 6.36)
    check("comic: left after 3 clean updates", mean(-r["residue_push_rank"] for r in ins), 0.66)
    check("comic: after removal", mean(-r["rollback_vs_clean_mean"] for r in ins), -4.61)
    cds = [r for r in iso if r["family"] == "relabel" and r["domain"] == "cd"]
    check(
        "re-listing late removal on CDs alone",
        mean(-r["rollback_vs_clean_mean"] for r in cds),
        -14.6,
        tol=0.06,
    )


def s5_written() -> None:
    heading(5, "Results that exist only as written tables (November, raw runs on the cluster)")
    say(
        """
        These came after the committed grid. Their run folders are on the NUS SoC cluster (Slurm
        jobs 1035987-1045609), and `cp4285 dose-summary` refuses the local October dumps, so this
        checkout can only quote them. Treat them as one step less checkable than section 4.
        """
    )
    items = [
        (
            "Dose: effect clears noise near 1% in all three catalogues",
            "docs/CONTINUED-TRAINING.md",
            "**The mean-rank threshold is around 1% in all three domains.**",
        ),
        (
            "Dose: relative climb and top-5 reach at 3% (55/25/13%)",
            "docs/CONTINUED-TRAINING.md",
            "| 3% | 0.429 | 0.431 | 0.417 | 0.55 | 0.25 | 0.13 |",
        ),
        (
            "Dose: raw rank change depends on starting rank (9 to 14)",
            "docs/CONTINUED-TRAINING.md",
            "**Raw rank change is mostly baseline.**",
        ),
        (
            "Dose: top-1 never reached, even at 20%",
            "docs/CONTINUED-TRAINING.md",
            "top-1 stays at 0–0.5%",
        ),
        (
            "Activity: high- against low-activity accounts",
            "docs/CONTINUED-TRAINING.md",
            "| Instruments | −3.65 (SD 1.09) |",
        ),
        (
            "Activity: untouched shoppers move as far as edited ones",
            "docs/CONTINUED-TRAINING.md",
            "edited users' own push-rank change was −5.17 against −5.12",
        ),
        (
            "Late removal: kept +0.19, removed -1.85, the adapter's own push about 2",
            "docs/CONTINUED-TRAINING.md",
            "| 4 | +0.19 | −1.85 | 2.04 |",
        ),
        (
            "Accuracy barely moves (Hit@1 under 0.2 points on average)",
            "docs/CONTINUED-TRAINING.md",
            "with mean shifts under 0.2 points",
        ),
    ]
    for label, rel, needle in items:
        print(f"\n  - {label}")
        quote(rel, needle)
    say(
        """

        "More than 400 runs" on the page has no single source. It sums 303 (October grid and
        attack catalogue) + 54 (cross-catalogue dose) + 75 (activity) = 432:
        """
    )
    quote("docs/CONTINUED-TRAINING.md", "the study holds 303 runs")
    quote("docs/CONTINUED-TRAINING.md", "54 new runs")
    quote("docs/CONTINUED-TRAINING.md", "75 runs across jobs")


def s6_next() -> None:
    heading(6, "Where to go next, and what is still open")
    say(
        """
        Read in this order: CONTEXT.md (status of every strand), docs/PESO-REPLICATION.md
        (findings A and B, and the grid write-up), docs/CONTINUED-TRAINING.md (protocol and the
        November tables), docs/DATA-WALKTHROUGH.md (how to open the local files), then
        docs/REQUIREMENTS.md#steps-poster for the STePS rules.

        The STePS page text, field map and booth notes are in a private Claude doc, and the nine
        page images are on a private design canvas; both links are in CONTEXT.md. The images are
        public on team-repo branch docs/steps-page-images.

        Open before anything goes out: team sign-off on the attack study, which is still an agent
        proposal; a UVENTS login to fill the page; the poster redo (print dates 5, 6, 9 and 10
        November, PDF also to Canvas Files); the mentor's name and member roles, which the repo
        does not confirm; and confirming the AI-use line.
        """
    )


SECTIONS = [s0_orientation, s1_data, s2_peso, s3_ladder, s4_attacks, s5_written, s6_next]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", action="store_true", help="pause for Enter between sections")
    ap.add_argument("--only", type=int, choices=range(len(SECTIONS)), help="run one section")
    args = ap.parse_args()
    chosen = [SECTIONS[args.only]] if args.only is not None else SECTIONS
    for i, section in enumerate(chosen):
        section()
        if args.step and i < len(chosen) - 1:
            input("\n-- Enter for the next section --")
    print("\n" + "=" * WIDTH)
    if problems:
        print(f"{len(problems)} check(s) differ or quoted lines moved:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Every recomputed number matches the page, and every quoted line was found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
