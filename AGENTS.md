# CP4285 project — agent guide

This checkout is the canonical **group-project** home, not the whole course workspace. Read [CONTEXT.md](CONTEXT.md) first, then the relevant document under `docs/`. The source implementation and [runbook](docs/IMPLEMENTATION.md) establish what actually runs; design notes and slides include unimplemented proposals.

## Navigation

- `README.md`: setup, repository map and public/private boundary.
- `reference.html`: reviewer walkthrough and diagrams; preserve its structure, append dated checkpoints and keep the top status notice current.
- `docs/PACKAGES-AND-BUILDING.md`: sample-first development guide: complete model/evaluator code, contract check, notebook recipe and standing rules. Tests enforce source/example agreement.
- `docs/MEETING-2026-10-15.md`: current meeting synthesis; distinguish agreements from exploration.
- `docs/PROJECT.md`, `docs/UNDERSTANDING.md`, `docs/EXPERIMENT.md`, `docs/MODERN-RECOMMENDER.md`, `docs/CONTINUED-TRAINING.md`, `docs/DATASET-ASSESSMENT.md`: historical reasoning/proposals, not current runtime defaults.
- `docs/REQUIREMENTS.md`, `docs/DEFENSE.md`, `docs/SLIDE-FLOW.md`: critique preparation and neural slide content.
- `slides/README.md`, `slides/ISSUES.md`: pointer to the team's Google Slides deck (canonical; edit it there, only when asked), slide principles for a class audience, and known deck issues.
- `docs/SOURCES.md`: primary references, official course links and local-only source locations.
- `docs/SASREC.md`: historical architecture/source discussion; the trial port has been removed.
- `src/cp4285/experiment/`, `src/cp4285/models/`, `tests/`: active shared contracts and model implementations.
- `notebooks/01_shared_experiment.ipynb`: synthetic executable walkthrough importing both packages.
- `docs/AI_LOG.md`: collective AI-use documentation.

## Working rules

- Use `uv sync --locked`, `uv run pytest -q`, and `uv run ruff check src tests` from the root. Do not modify global Python or copy old virtual environments. The old `cp4285` CLI is retired.
- Use the synthetic notebook/API for a small check; tests execute its cells offline. Real-data acquisition or expensive training requires an explicit decision.
- Build two responsibility packages, not separate workstream benchmarks. Models implement the experiment ABC; evaluation belongs in `experiment/evaluation.py`. Do not put private splits/metrics in models or notebooks, or force scientifically different tasks into one universal runner.
- Keep development examples runnable. When changing the sample model/evaluator, update the matching guide code and run `uv run pytest -q tests/test_documentation.py`. Prefer one complete sample over placeholder models or a plugin framework.
- Trial code is disposable and has been removed; do not restore compatibility layers without a concrete need. Preserve scientific reasoning and contract tests. Review the north-star document before extending interfaces.
- Preserve holdout isolation, original identities, fixed candidates, validation-only selection and no-overwrite safeguards. Synthetic numbers must never be represented as Amazon results.
- Keep implemented behavior, team-confirmed choices, agent proposals and external findings distinct. Do not promise degradation or neural superiority.
- Prefer course terminology. Verify assessment constraints against official sources; do not guess Canvas deadlines or submit/contact anyone without authorization.
- Read source PDFs only as needed (`pdfinfo`, `pdftotext -layout -f N -l M`, `pdftoppm` for visual checks). Cite filenames and page/slide numbers. Do not infer diagrams or equations from broken text extraction.
- `local/`, `data/`, `runs/`, `reports/`, `.pi/` and `.pi-subagents/` are ignored. Never force-add transcripts, chats, team PDFs, personal IDs, course PDFs, credentials or datasets. Public links are preferable to republishing third-party material.
- The public repo must work without `local/` or the original course directory. Machine-specific locations and backup details belong in ignored `local/WORKSPACE.md`.
- Preserve student ownership. Update `CONTEXT.md` when decisions or implementation status change and `docs/AI_LOG.md` after substantive AI-assisted work.

No additional Pi extension or agent framework is needed: these Markdown entry points are the handoff.
