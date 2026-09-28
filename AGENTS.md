# CP4285 project — agent guide

This checkout is the canonical **group-project** home, not the whole course workspace. Read [CONTEXT.md](CONTEXT.md) first, then the relevant document under `docs/`. The source implementation and [runbook](docs/IMPLEMENTATION.md) establish what actually runs; design notes and slides include unimplemented proposals.

## Navigation

- `README.md`: setup, repository map and public/private boundary.
- `docs/PROJECT.md`, `docs/UNDERSTANDING.md`: purpose, meeting synthesis and decision status.
- `docs/EXPERIMENT.md`, `docs/MODERN-RECOMMENDER.md`, `docs/CONTINUED-TRAINING.md`: research protocol.
- `docs/DATASET-ASSESSMENT.md`, `configs/pilot.toml`: provisional data choices and runnable defaults.
- `docs/REQUIREMENTS.md`, `docs/DEFENSE.md`, `docs/SLIDE-FLOW.md`, `slides/index.html`: critique preparation and editable slides.
- `docs/SOURCES.md`: primary references, official course links and local-only source locations.
- `docs/SASREC.md`: selected original architecture, pinned upstream code, license and explicit training/evaluation differences.
- `src/cp4285/`, `tests/`: Python CLI and lifecycle tests.
- `docs/AI_LOG.md`: collective AI-use documentation.

## Working rules

- Use `uv sync --locked`, `uv run cp4285 doctor`, `uv run pytest -q`, and `uv run ruff check src tests` from the root. Do not modify global Python or copy old virtual environments.
- Default to `uv run cp4285 demo --output runs/<new-name>` for an offline synthetic check. Full data downloads require an explicit decision; do not start them as part of setup or tests.
- Preserve holdout isolation, original identities, fixed candidates, validation-only selection and no-overwrite safeguards. Synthetic numbers must never be represented as Amazon results.
- Keep implemented behavior, team-confirmed choices, agent proposals and external findings distinct. Do not promise degradation or neural superiority.
- Prefer course terminology. Verify assessment constraints against official sources; do not guess Canvas deadlines or submit/contact anyone without authorization.
- Read source PDFs only as needed (`pdfinfo`, `pdftotext -layout -f N -l M`, `pdftoppm` for visual checks). Cite filenames and page/slide numbers. Do not infer diagrams or equations from broken text extraction.
- `local/`, `data/`, `runs/`, `reports/`, `.pi/` and `.pi-subagents/` are ignored. Never force-add transcripts, chats, team PDFs, personal IDs, course PDFs, credentials or datasets. Public links are preferable to republishing third-party material.
- The public repo must work without `local/` or the original course directory. Machine-specific locations and backup details belong in ignored `local/WORKSPACE.md`.
- Preserve student ownership. Update `CONTEXT.md` when decisions or implementation status change and `docs/AI_LOG.md` after substantive AI-assisted work.

No additional Pi extension or agent framework is needed: these Markdown entry points are the handoff.
