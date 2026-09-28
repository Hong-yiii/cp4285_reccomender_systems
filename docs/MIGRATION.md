# Standalone project migration

28 September 2026. The group project was split from the CP4285 course workspace into the public `cp4285_reccomender_systems` repository. The spelling is intentional: it matches the requested repository name. Python's existing package name, `cp4285-recommender-study`, and `cp4285` command are unchanged.

| Former course-workspace path | Project-repository path |
| --- | --- |
| `project/README.md` | `docs/PROJECT.md`; root `README.md` is the new entry point |
| `project/*.md` | `docs/` |
| `project/configs/` | `configs/` |
| `project/diagrams-html/` | `slides/` |
| `src/`, `tests/`, `pyproject.toml`, `uv.lock`, `.python-version` | Same root-level locations |
| `project/sources/` | `local/sources/` (**ignored**) |
| Course `slides/`, lessons and reference notes | `local/course/` (**ignored** copies) |
| `project/data/`, `project/runs/`, `project/reports/` | `data/`, `runs/`, `reports/` (**ignored**) |

## Source of truth and preservation

- The old course folder keeps coursework and its existing Git history. Its `project` shortcut points to this checkout; its agent guide and `PROJECT.md` direct project work here.
- Original project files and pre-migration navigation files are retained in the old folder's ignored `.project-migration-backup-2026-09-28/`. They are a recovery snapshot, **not** an active second project.
- The new repository begins with a clean history: old course commits included unrelated coursework and agent transcripts and are deliberately not imported.
- Existing synthetic runs were copied locally without altering their provenance. Re-run checks with fresh output paths; old metrics contain old absolute paths.
- Local machine paths and the exact backup location are in ignored `local/WORKSPACE.md`. No machine-specific absolute path is required by published code or documentation.
- Source originals are backed up before redirecting. To undo the local redirect, remove only the old `project` symlink and restore the archived project files/navigation from the backup. Do not delete or overwrite this checkout.

## Publication boundary

Only authored code, tests, configuration, working notes and self-contained HTML slides are published. Raw meeting/chat/sketch sources, draft team PDF, official course/template PDFs or exports, paper copies, datasets, model weights and agent sessions stay local. Teammate identifiers and original attachment paths are omitted from the public provenance summary. Official course and paper URLs remain in [SOURCES.md](SOURCES.md).

No license is assigned on behalf of teammates or course authors. Review ownership and permission before adding a redistribution license or publishing more source material. GitHub Pages, CI, dataset downloads and a live service are not needed for this migration.

## Continuing in Pi

Open the project checkout in Pi and start with `AGENTS.md` → `CONTEXT.md` → the task-relevant document. No old agent session or custom extension is required. Record substantive future assistance in [AI_LOG.md](AI_LOG.md).
