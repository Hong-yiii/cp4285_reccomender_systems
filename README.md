# CP4285 · Recommender Systems

Project home for **CP4285: Modern Recommendation Systems**, NUS School of Computing.

**Research question:** How does continued training on a changing domain mixture affect a recommender's performance on its original task?

**Status — 28 September 2026:** working research proposal plus a runnable **original-architecture SASRec A-only pilot (PyTorch port)**. Hongyi selected the authors' SASRec model; see [source fidelity and protocol differences](docs/SASREC.md). Dataset auditing, checkpoint reload, frozen evaluation and A-only continuation are implemented. Amazon Electronics / Movies & TV and the shared two-domain model remain provisional. B interfaces, mixed updates, the classical comparator and multi-seed experiments are not implemented. Synthetic checks are **not** real-data results.

## Start here

- [Current context and next steps](CONTEXT.md)
- [Project overview](docs/PROJECT.md) and [experiment definition](docs/EXPERIMENT.md)
- [Neural workstream](docs/MODERN-RECOMMENDER.md) and [continued-training protocol](docs/CONTINUED-TRAINING.md)
- [Dataset assessment](docs/DATASET-ASSESSMENT.md)
- [Runbook](docs/IMPLEMENTATION.md) and [pinned SASRec implementation](docs/SASREC.md)
- [Critique requirements](docs/REQUIREMENTS.md), [preparation](docs/DEFENSE.md) and [slide narrative](docs/SLIDE-FLOW.md)
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
| `src/cp4285/` | CLI, ingestion/audit, model and A-only training |
| `tests/` | Data isolation, evaluation and checkpoint lifecycle checks |
| `configs/pilot.toml` | Reviewable pilot settings; paths relative to the repo root |
| `docs/` | Research context, requirements, protocol, provenance and AI log |
| `slides/index.html` | Editable six-slide technical narrative; open directly in a browser |
| `local/` | **Ignored** course PDFs, private team sources and local migration notes |
| `data/`, `runs/`, `reports/` | **Ignored** downloaded data and generated artifacts |
| `AGENTS.md`, `CONTEXT.md` | Pi/agent entry points and current project state |

## Public repository boundary

This repository publishes project code, authored research notes and the editable HTML diagrams. Raw meeting transcripts, prior agent chats, team PDFs, course PDFs, datasets, checkpoints and agent sessions stay local. See [source provenance](docs/SOURCES.md) for official links and local-only filenames; those local files are intentionally absent from a fresh clone. Do not force-add them without reviewing privacy, course policy and redistribution rights.

The course workspace now points here; this checkout is the project source of truth. [Migration notes](docs/MIGRATION.md) describe the split. No submission is made by this repository, and working proposals are not team-approved decisions.
