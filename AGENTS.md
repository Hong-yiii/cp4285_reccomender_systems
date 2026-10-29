# CP4285 project — agent guide

This checkout is the canonical **group-project** home, not the whole course workspace. Read [CONTEXT.md](CONTEXT.md) first, then the relevant document under `docs/`. The source implementation and [runbook](docs/IMPLEMENTATION.md) establish what actually runs; design notes and slides include unimplemented proposals.

## Navigation

- `README.md`: setup, repository map and public/private boundary.
- `reference.html`: reviewer walkthrough and diagrams; preserve its structure, append dated checkpoints and keep the top status notice current.
- `docs/PROJECT.md`, `docs/UNDERSTANDING.md`: purpose, meeting synthesis and decision status.
- `docs/EXPERIMENT.md`, `docs/MODERN-RECOMMENDER.md`, `docs/CONTINUED-TRAINING.md`: research protocol.
- `docs/DATASET-ASSESSMENT.md`, `configs/pilot.toml`: provisional data choices and runnable defaults.
- `docs/REQUIREMENTS.md`, `docs/DEFENSE.md`, `docs/SLIDE-FLOW.md`: critique preparation and neural slide content.
- `slides/README.md`, `slides/ISSUES.md`: pointer to the team's Google Slides deck (canonical; edit it there, only when asked), slide principles for a class audience, and known deck issues.
- `docs/SOURCES.md`: primary references, official course links and local-only source locations.
- `docs/SASREC.md`: selected original architecture, pinned upstream code, license and explicit training/evaluation differences.
- `docs/ADAPTERS.md`: sourced survey of adapters and parameter-efficient updates for sparse, continually updated recommenders; its closing experiment is an agent proposal.
- `docs/PACKAGES-AND-BUILDING.md`, `notebooks/01_shared_lifecycle.ipynb`: ABC development guide and runnable walkthrough; `boilerplate-v1` is the separate lightweight reference tag.
- `src/cp4285/experiment/`, `src/cp4285/models/`, `tests/`: protocol ownership, concrete learners and lifecycle tests. Keep native learning units and distinct classical/neural evaluations explicit; batch refits are not continuation.
- `docs/updates/`: dated progress shares in the meeting-notes format (`[Owner] Task: description` next steps). `notebooks/02_peso_replication.ipynb` reads only committed results.
- `docs/PESO-REPLICATION.md`, `replication/peso/`, `src/cp4285/experiment/flips.py`: LLM continual-recommendation replication on Modal and Runpod, plus the flip/retention, J-lens and penalty-size analyses. Upstream PESO has no licence: clone it into ignored `local/peso/` or the run image, never into the repo. GPU runs bill the user's Modal and Runpod accounts.
- `notebooks/03_peso_colab.ipynb`: Colab re-evaluation of trained weights downloaded from a GitHub release (Llama 3.2 licence, notice and "Built with Llama" ship with it). `REPO`/`BRANCH` choose the clone and release; publishing a release is outward-facing and needs the user's go.
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
