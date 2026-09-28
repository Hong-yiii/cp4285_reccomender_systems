# CP4285 · Recommender Systems

Project home for **CP4285: Modern Recommendation Systems**, NUS School of Computing.

**Research question:** How does continued training on a changing domain mixture affect a recommender's performance on its original task?

**Status — 28 September 2026:** working research proposal plus a runnable **original-architecture SASRec A-only pilot (PyTorch port)**. Hongyi selected the authors' SASRec model; see [source fidelity and protocol differences](docs/SASREC.md). Dataset auditing, checkpoint reload, frozen evaluation and A-only continuation are implemented. A **classical comparator** is also implemented, with contributor-reported real Amazon exploratory single-seed results under leave-last-out splitting ([CLASSICAL.md](docs/CLASSICAL.md)). Its protocol differs from the neural cutoff pilot, so their scores are not yet comparable. Amazon Electronics / Movies & TV and the shared two-domain neural model remain provisional. Neural B interfaces, mixed neural updates and multi-seed experiments are not implemented. Neural checks remain synthetic and are **not** real-data results.

## Start here

- [Reviewer reference](reference.html): project walkthrough, diagrams and dated checkpoints. Open `reference.html` locally from the checkout; GitHub displays its source.
- [Current context and next steps](CONTEXT.md)
- [Project overview](docs/PROJECT.md) and [experiment definition](docs/EXPERIMENT.md)
- [Neural workstream](docs/MODERN-RECOMMENDER.md) and [continued-training protocol](docs/CONTINUED-TRAINING.md)
- [Classical comparator](docs/CLASSICAL.md): Markov, QR-factorised Markov and SLIST under B contamination
- [Dataset assessment](docs/DATASET-ASSESSMENT.md)
- [Runbook](docs/IMPLEMENTATION.md) and [pinned SASRec implementation](docs/SASREC.md)
- [Critique requirements](docs/REQUIREMENTS.md), [preparation](docs/DEFENSE.md), [team deck and slide principles](slides/README.md), [deck issues](slides/ISSUES.md) and [neural slide content](docs/SLIDE-FLOW.md)
- [Sources and course slides](docs/SOURCES.md) · [AI assistance log](docs/AI_LOG.md)

The next recorded milestone is the **29 September 2026 design critique**. The official template's weekday/date conflict remains unresolved; verify Canvas rather than assuming a submission deadline.

## Run locally

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is pinned. From this checkout:

```sh
uv sync --locked
uv run cp4285 doctor
uv run pytest -q
uv run ruff check src tests
uv run cp4285 demo --output runs/my-synthetic-check
```

The demo generates synthetic data locally; it does not download Amazon. Use a fresh output directory for every run. Review [the runbook](docs/IMPLEMENTATION.md) before downloading full datasets or interpreting metrics.

## Layout

| Path | Purpose |
| --- | --- |
| `src/cp4285/cli.py` | `cp4285` command |
| `src/cp4285/common/` | Shared download, validated loading and audit (`data.py`); create-only JSON, hashing and UTC dates (`utils.py`) |
| `src/cp4285/neural/` | Neural workstream: cutoff pilot split, SASRec model and A-only training |
| `src/cp4285/classical/` | Classical comparator (`cp4285 classical ...`) |
| `tests/` | Data isolation, evaluation and checkpoint lifecycle checks |
| `configs/pilot.toml` | Reviewable pilot settings; paths relative to the repo root |
| `docs/` | Research context, requirements, protocol, provenance and AI log |
| `reference.html` | Self-contained reviewer walkthrough; preserve the structure and append dated updates |
| `slides/` | `README.md`: link to the team's Google Slides deck and principles for making slides. `ISSUES.md`: known deck problems |
| `local/` | **Ignored** course PDFs, private team sources and local migration notes |
| `data/`, `runs/`, `reports/` | **Ignored** downloaded data and generated artifacts |
| `AGENTS.md`, `CONTEXT.md` | Pi/agent entry points and current project state |

## Public repository boundary

This repository publishes project code, authored research notes and slide guidance. The team deck itself stays in Google Slides. Raw meeting transcripts, prior agent chats, team PDFs, course PDFs, datasets, checkpoints and agent sessions stay local. See [source provenance](docs/SOURCES.md) for official links and local-only filenames; those local files are intentionally absent from a fresh clone. Do not force-add them without reviewing privacy, course policy and redistribution rights.

The course workspace now points here; this checkout is the project source of truth. [Migration notes](docs/MIGRATION.md) describe the split. No submission is made by this repository, and working proposals are not team-approved decisions.
