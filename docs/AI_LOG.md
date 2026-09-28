# AI assistance log

## 25 September 2026 — Project initialization

**Tool:** OpenAI Codex, including two internal agents for independent transcript synthesis and technical checking.

**User request:** Understand the CP4285 project from the supplied meeting notes and diagram, propose what to do first for user verification, and initialize a folder of Markdown notes.

**Inputs:** Supplied transcript and sketch; repository context; public course schedule, assignments, lecture slides and critique template; narrowly selected primary technical sources listed in SOURCES.md.

**Assistance provided:** Reconstructed the discussion and tentative roles; checked public assessment details; identified methodological ambiguities; proposed an experiment definition and critique preparation sequence; created this folder and updated the repository map.

**Human decisions pending:** Verify project interpretation, final scope, data/model choices, roles, deadline, and current team progress. None of the proposed experimental decisions is represented as team-approved.

**Artifacts and limits:** Markdown working notes and preserved source copies. No training, measured results, submitted work, external messages, or created slide deck.

This is a summary record, not a full prompt transcript. Retain the Codex conversation and add later prompts/outputs used by the team when preparing its collective AI-use documentation. This initialization does not establish how other teammates have used AI.

## 25 September 2026 — Neural workstream validation

**User clarification/request:** Hongyi owns the modern recommender contribution; validate the assumptions that modern models learn continually, that initial training is followed by learning during deployment, and that mixed-domain continued training is a valid experiment. Read and critique the supplied prior agent chat, then formalize the setup.

**Assistance provided:** Distinguished model architecture, changing inference inputs and actual parameter updates; checked SASRec, ADER, Monolith and CCTL primary sources; proposed a shared neural model and explicit training/evaluation controls. Preserved the supplied chat and added MODERN-RECOMMENDER.md. No new subagents were used for this follow-up.

**Status:** Role confirmed; architecture and implementation choices remain proposals. No experiments or results generated. Mathematical reasoning and proposed controls are agent analysis, not findings attributed to papers or the team.

## 27 September 2026 — SASRec paper walkthrough

**User request:** Explain the SASRec paper and how it fits the experiment.

**Assistance:** Read the original architecture, training and evaluation sections, visually inspected Figure 1, checked the authors' model/evaluation code and the official Week 05 SASRec material. Explained sequence representations versus persistent model weights, and separated the published model from the proposed continual-training protocol. Preserved the original paper locally (now `local/sources/sasrec-2018.pdf` after migration). No model selection was finalized and no experiment was run.

## 27 September 2026 — Continued-training specification

**User request:** Formalize adding continued training to the proposed SASRec experiment.

**Assistance:** Created CONTINUED-TRAINING.md with initial and continued phases, next-item objective, mixture accounting, optimizer-state policy, pseudocode, fixed test protocol and controls. Rechecked the original SASRec and ADER sources. The proposed defaults and equations are working design assistance; no dataset, architecture or numerical training budget has been finalized. No training code or empirical result was produced.

## 28 September 2026 — Scoped critique preparation

**User request:** Review the Group 3 PDF, break down requirements and scope preparation to Hongyi's modern-recommender work.

**Assistance:** Read all deck text and visually inspected relevant pages; compared with the preserved official template; expanded DEFENSE.md with page-specific gaps, neural preparation, controls, mixture definitions and rehearsal questions. Rechecked SASRec and ADER, including the draft's inaccurate RNN/YouTube reference.

**Limits:** PDF unchanged; no submission or experiment performed. No compute access, dataset approval or implementation progress inferred. Recommendations remain AI-assisted preparation for user/team verification.

## 28 September 2026 — Slide narrative and diagrams

**Request:** Provide diagrams and work out a logical slide flow for the neural contribution.

**Assistance:** Created SLIDE-FLOW.md with four editable Mermaid diagrams for SASRec inference, proposed shared-domain architecture, continued updates and comparison conditions. Added speaker notes and placement within the existing group sections. No slide deck was modified, no experiment was run and no synthetic performance results were drawn.

## 28 September 2026 — Dataset suitability review

**Request:** Evaluate Amazon Reviews 2023 Electronics/Movies & TV and DIGINETICA/YOOCHOOSE for the continued-training study, including a teammate's reference-paper interpretation.

**Assistance:** Checked official Amazon schema/processing statistics, ADER paper and repository, CCTL experiment setup and YOOCHOOSE authors' documentation. Created DATASET-ASSESSMENT.md with a provisional Amazon recommendation, limitations and feasibility criteria. Flagged the difference between ADER's separate temporal benchmarks and the proposed cross-dataset mixture, and the need to audit parameter sharing in the classical comparison.

**Limits:** Documentation assessment only. No full interaction files downloaded, overlap measured, dataset choice finalized, training run or performance claim made. The teammate's exact intended Amazon paper remains unidentified.

## 28 September 2026 — uv initialization and HTML update

**Request:** Initialize the existing repository for the proposed audit/target/design/A-only-pilot steps using uv, install dependencies and update the supplied HTML.

**Assistance:** Added pyproject/lockfile, a local Python 3.12 environment, configurable data audit/preparation commands, an official ID-file downloader, a SASRec-style A-only checkpoint/continuation pilot, meaningful tests and implementation documentation. Updated the existing HTML to six slides with Amazon dataset details, review-event semantics and explicit implementation gates. Preserved the existing Git repository.

**Validation:** Seven tests and lint/format checks passed. A synthetic end-to-end run reproduced evaluation after checkpoint reload with zero maximum score difference and verified a fixed frozen control. Browser review covers slide navigation, speaker notes and layout.

**Limits:** No Amazon interaction downloads or real-data metrics. Dataset selection, review target and time cutoffs remain provisional. A-only code does not yet implement B interfaces, mixed updates, classical models, exact SASRec paper replication or a final multi-seed study. The target defaults to all review events and can be changed through the rating filter.

## 28 September 2026 — Standalone repository and Pi handoff

**Tool:** OpenAI assistant in Pi, with independent read-only scout and reviewer agents.

**Request:** Make a standalone public `cp4285_reccomender_systems` repository, reorganize the existing project and context/slides, and redirect the course workspace to the new project home.

**Assistance:** Migrated code/tests and locked environment metadata; separated `configs/`, `docs/` and editable `slides/`; updated runtime/documentation paths; added README, CONTEXT and AGENTS entry points plus a layout regression check. Retained source evidence, course PDFs, draft team PDF and previous synthetic runs locally behind Git ignores. Removed private attachment paths and teammate identifiers from public summaries. Backed up the source project and redirected the course workspace without importing unrelated course Git history.

**Validation:** Locked environment installed; all 8 tests, Ruff lint/format and the synthetic CLI smoke check passed. A separate export containing only the 29 publishable files installed and passed all 8 tests without local/course sources; its synthetic demo passed on a direct retry after a combined validation task timed out. Both completed smoke checks reproduced checkpoint scores with zero maximum difference. Relative Markdown links, the old workspace redirect, source-copy integrity and Git ignores were checked. Gitleaks found no secrets in the public-only export; independent migration review reported no findings. Static diagnostics were not fully clean: stale environment/import findings and a pre-existing optional-checkpoint type warning remain, with details retained locally.

**Limits:** No new research decisions, data downloads, real-data results, final submission, license grant or deployment. Published notes remain AI-assisted working material, not team approval. Historical source files and runs retain their original contents; machine-specific migration details live only in `local/WORKSPACE.md`.

## 28 September 2026 — Implementation code review

**Tool:** Cursor Grok 4.7.

**Request:** Review the implemented audit, split, SASRec-style model and A-only pilot.

**Assistance:** Read the package, tests, pilot config and runbook. Compared the code with the stated holdout, checkpoint and evaluation contract. Did not change research decisions, download data, or treat synthetic checks as Amazon results.

**Findings to verify:** Right-padded histories give the prediction token a length-dependent position id. Validation and retention share one prefix, so checkpoint selection and the retention metric are two labels on one score vector. Out-of-vocabulary drops can put different users in those two splits. Download promotion checks the CSV header and does not read the gzip checksum. Prepare/audit outputs can be overwritten; DuckDB spill uses a working-directory-relative path.

**Validation:** `uv run pytest -q` passed, 8 tests. No code changes.

## 28 September 2026 — Original SASRec architecture

**Tool:** OpenAI assistant in Pi, with read-only upstream research and independent code review.

**User decision:** Use the original [kang205/SASRec](https://github.com/kang205/SASRec) immediately rather than the generic Transformer pilot.

**Assistance:** Inspected and pinned upstream commit `e3738967fddab206d6eeb4fda433e7a7034dd8b1`. Ported its architecture from Python 2 / TensorFlow 1.12 to the existing PyTorch runtime, preserving normalized residuals, raw K/V, masking, hidden-width feedforward, tied scoring and sampled loss. Adopted fixed left-padding, original architecture defaults, sequence-wise initial supervision with user sampling and Adam beta2=0.98. Continuation still masks historical targets to avoid implicit replay. Added source/license notices, checkpoint provenance and explicit protocol differences in SASREC.md. The earlier Cursor review entry above is preserved unchanged.

**Validation:** All 11 tests, Ruff lint and formatting passed. Tests include a separate NumPy translation of upstream forward/loss equations (one and two heads), causal isolation, finite gradients, training masks, holdout exclusion and checkpoint lifecycle. The full configured model completed a synthetic demo with zero reload score difference and unchanged frozen control. Independent source-fidelity review reported no findings. Source/wheel builds passed, and the wheel was checked for the Apache license and attribution notice.

**Limits:** This is a PyTorch port, not execution of the legacy TensorFlow code. No TensorFlow numerical comparison, original-paper benchmark reproduction or real Amazon experiment was performed. The model is selected; the two-domain design, final dataset and training budget are not thereby approved. Earlier generic-model checkpoints are incompatible. Existing data-audit/download concerns recorded in the Cursor review are outside this model change.

## 28 September 2026 — Classical comparator

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Explore non-deep-learning (matrix-factorisation / QR-based) recommenders for the project, build a baseline to iterate on, implement a classical method from the literature (SLIST), and integrate it into this repository.

**Assistance:** Downloaded the official 5-core Electronics and Movies & TV ID files (plus the unfiltered Movies & TV file) and measured A/B overlap. Implemented `src/cp4285/classical/`: DuckDB loader with shared original user IDs, leave-last-out split, disjoint and shared-user contamination designs, popularity, raw and QR-factorised Markov chains, user-item PureSVD with a QR-based incremental variant, and SLIST ported from the authors' code. Added `cp4285 classical` subcommands, eight tests, SciPy as a dependency and CLASSICAL.md with the first real-data sweeps.

**Validation:** 16 tests plus Ruff lint and format passed. CLI smoke test on real data. Full-data sweeps ran on a laptop (validation split, one seed).

**Limits:** Exploratory single-seed numbers with lightly chosen hyperparameters (SLIST α and age decay compared on validation). The protocol is leave-last-out, not the pilot's cutoff design, so neural and classical numbers are not yet comparable. No Diginetica run, no team approval of model choice implied.

## 28 September 2026 — SASRec PR and workstream integration

**Tool:** OpenAI assistant in Pi, with independent read-only review.

**Request:** PR the current work into main, then continue to the next checkpoint.

**Assistance:** Reviewed and published the original-SASRec changes in PR #2. Main advanced with the teammate's classical comparator (PR #1); merged that work without dropping its CLI, dependency, tests or AI log. Reconciled status notes to distinguish the synthetic neural cutoff pilot from contributor-reported real-data classical leave-last-out sweeps. Preserved the Cursor review and existing slide formatting.

**Validation:** The combined suite passed all 21 tests, Ruff lint/format, CLI doctor/classical help and source/wheel builds. Both SASRec-only and combined public exports passed Gitleaks. Integration review identified an unguarded classical config preparse and unqualified result provenance; fixed both, first reproducing missing/malformed-config failures with two regression cases. Follow-up review confirmed the working-tree fixes. The classical real-data sweeps were not rerun during integration.

**Limits:** No data downloads, common-protocol model comparison, protocol approval or course submission. The next implementation checkpoint is download integrity and non-overwriting artifact persistence; research decisions remain explicit gates.

## 28 September 2026 — Real-run artifact safety checkpoint

**Tool:** OpenAI assistant in Pi, with read-only review.

**Request:** After merging the existing work, continue to the next checkpoint.

**Assistance:** Merged SASRec PR #2 into main alongside the classical workstream, then implemented full gzip/header/trailer verification before download publication. Each download uses a unique temporary file; existing data/manifests are preserved. Reused the shared JSON writer for audits, prepared snapshots, manifests, neural metrics and classical reports, publishing complete finite JSON without overwriting even a racing destination. Kept the current research split and training protocol unchanged. Made single-row SQL aggregate results explicitly non-optional for static checks without changing the queries.

**Validation:** First reproduced 10 failures in 12 offline safety cases (existing-artifact replacement, corrupt/truncated gzip promotion, failure cleanup, orphan manifest handling and CLI behavior). After fixing the shared boundaries, all 33 project tests passed along with Ruff lint/format and fresh primary LSP checks for the changed Python files. Source/wheel builds and a new synthetic CLI demo passed; reload score difference was 0.0. Independent read-only review found no issues. Tests mock HTTP locally and do not download datasets.

**Limits:** Atomic file visibility is not power-loss durability or a two-file transaction. A manifest write failure preserves validated data for inspection; hardlink support is required. Killed processes may leave unpromoted temporary files. No acquisition was approved, no new real-data result was generated, and validation/retention cohort semantics plus classical/neural protocol alignment still require a team decision.

## 28 September 2026 — Shared utilities housekeeping

**Tool:** OpenAI assistant in Pi, with read-only scouting.

**Request:** Pull the latest code, create a folder with `utils.py` for common functions, and do housekeeping.

**Assistance:** Fetched origin and checked the current safety branch with a fast-forward-only pull; it was already up to date. Started `refactor/common-utils` from that checkpoint without merging its still-open PR #3. Moved `save_json`, `sha256` and `millis` into the standard-library-only `src/cp4285/common/utils.py`, removed their duplicate definitions from ingestion, and updated every direct caller/test import. Updated the layout/runbook/context and retained existing AI-log entries. Kept workstream-specific data loaders and research protocols separate.

**Validation:** All 34 tests passed, including a new multi-chunk hash/UTC-date/error-propagation check and the existing download/JSON artifact-safety regressions. Ruff lint/format, eight-file primary LSP diagnostics and source/wheel builds passed. AST comparison confirmed all three helper bodies are unchanged from the safety checkpoint; an isolated import confirmed utilities do not load database, HTTP, numerical or model dependencies.

**Limits:** This is code organisation, not a new experiment or protocol decision. Existing no-overwrite semantics and error propagation are preserved. No dataset was downloaded and no real-data result was produced.

## 28 September 2026 — Public reviewer reference and utilities PR

**Tool:** OpenAI assistant in Pi, with independent read-only publication review.

**Request:** PR the utilities housekeeping together with the existing reference HTML, retaining its structure as a top-level reviewer reference.

**Assistance:** Promoted a copy of the teaching walkthrough to `reference.html`. Preserved the complete original narrative, integration/safety appendix, CSS and five diagrams; removed personal checkout/disposable-page framing and added current-status navigation plus a dated utilities/reviewer checkpoint. Clearly labeled historical Git states, test counts and unpublished local run evidence. Linked the page from README, context and the agent guide. Prepared a stacked review against `fix/real-run-artifact-safety` because PR #3 remains open; documented merging that prerequisite, rebasing and retargeting before merging this change.

**Validation:** All 35 tests passed in 3.36s, including one new offline reference check for unique/resolving anchors, repository-local links, self-containment and publication metadata. Ruff lint/format, nine-file primary diagnostics, source/wheel builds and public-only Gitleaks scanning passed. The source distribution includes the reference and the wheel includes the utilities. Compared the copied narrative/appendix and CSS against the original; confirmed the original temporary file is unchanged. Opened the file directly in a browser at 1440px and 390px, verified working checkpoint navigation, zero loaded external assets and no page-level horizontal overflow. Independent review found no remaining publication issues.

**Limits:** No PR merge, dataset acquisition, new research result, deployment or course submission is authorized by this reference. Historical local run artifacts are not included; source/runbook remain authoritative, and future checkpoints should be appended rather than silently replacing evidence.

## 28 September 2026 — SASRec architecture walkthrough follow-up

**Tool:** OpenAI assistant in Pi, with independent read-only code fact-checking and review.

**Request:** Stack another commit on PR #4 explaining SASRec in the detailed, concrete style of the supplied conceptual questions.

**Assistance:** Appended section 17 to `reference.html`: raw reviews versus model features, frozen catalogue/OOV rules, a consistent five-position reviewer example, tensor shapes, item/position embeddings, original attention/masks/residuals/feedforward, sampled logistic loss, phase-specific labels/negatives, full-catalogue retention arithmetic and changing parameters versus fixed evaluation. Added three captioned diagrams and source links. Clarified held-out events versus reusable product IDs, catalogue allocation versus actual training exposure, independent logistic loss versus pairwise ranking, and gradient flow through history despite masked historical target losses. Preserved the pre-existing HTML formatting, CSS and earlier sections 01–16. Added a link from `docs/SASREC.md`; used the questions as inspiration rather than publishing the supplied conversation.

**Validation:** Added one executable synthetic worked-example regression in `tests/test_sasrec.py`, covering initial/continuation tensors, full-score dot products, sampled loss, a context-only item's gradient and the rank-4/tie arithmetic. The full suite passed all 36 tests in 3.54s; Ruff lint/format, primary test-file diagnostics, source/wheel builds and public-only Gitleaks checks passed. Byte comparison confirmed earlier sections and formatted CSS remain intact. Browser checks at 1440px/390px confirmed eight captioned diagrams, resolving anchors, no external assets and no page-level horizontal overflow, including the expanded architecture details. Independent source review found no inaccuracies requiring changes.

**Limits:** Production model/data/training code is unchanged. Numerical examples are invented software checks, not measured Amazon performance or TensorFlow parity. The fixed-anchor interpretation, OOV coverage and classical/neural comparability remain research-contract questions. No data acquisition, deployment, submission or PR merge.

## 28 September 2026 — Neural package and shared data handling

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Move the deep-learning approach into one folder, move the data handling methods into `common`, rearrange whatever else is needed and open a PR.

**Assistance:** Created `src/cp4285/neural/` holding `model.py` (SASRec port), `pilot.py` (training/evaluation lifecycle) and a new `data.py` with the unchanged cutoff-split `prepare`. Moved protocol-independent download, validated loading, DuckDB connection settings and the A/B audit to `src/cp4285/common/data.py`. Kept `prepare` in `neural/` rather than `common/`, because placing it in common would imply the classical workstream had adopted the cutoff protocol; left the classical loader and split untouched. Updated CLI/test imports, NOTICE attribution paths, README, runbook, SASREC, context and the agent guide. In `reference.html`, repointed section 17's source links, refreshed the stale top status (PRs #1–#4 are merged) and appended section 18 describing the layout; earlier sections' dated file/line references were left as historical evidence.

**Validation:** AST comparison confirmed all 18 moved top-level definitions are unchanged. All 36 tests passed; Ruff lint/format passed; a fresh synthetic `cp4285 demo` completed with reload difference 0.0; the built wheel contains the `common`, `neural` and `classical` packages.

**Limits:** Import paths changed (`cp4285.data`, `cp4285.model`, `cp4285.pilot` no longer exist); CLI commands, config, checkpoints and behavior did not. No dataset was downloaded and no real-data result was produced. Whether classical should adopt the common validated loader or the cutoff split remains a team decision.

## 28 September 2026 — Team deck pointer and slide principles

**Tool:** Claude Code (Claude Opus 5.5), with the in-app browser for read-only viewing.

**Request:** Replace the HTML slides with a pointer to the team's Google Slides deck plus principles for making slides, and integrate the repository's work with that deck. Follow-ups: write the principles for a class audience with no assumed knowledge of SASRec, data techniques, the dataset or the papers, while keeping rigour; add a file of slide issues; open and merge a PR.

**Assistance:** Read the live deck through its public text export and preview, without editing it. Removed `slides/index.html` and added `slides/README.md`: the deck link, the intended audience and a "classmate test", 19 principles (explain before use, rigour in plain words, readable visuals, one deck with four authors), a plain-wording glossary of terms the deck uses, and a pre-export checklist. Added `slides/ISSUES.md`: deck-wide and slide-by-slide problems ranked must/should, covering contradictions, stale status, reviews described as purchases, unexplained jargon and placeholder references. Added a dated section to `docs/SLIDE-FLOW.md` placing two paste-ready neural slides written for the same audience: the empty data slide introduces the dataset, the outdated technical-plan slide becomes N1, and hiding the version history makes room for N2. Updated the implementation-status table there, and the pointers in README, CONTEXT, AGENTS, PROJECT, IMPLEMENTATION and DEFENSE. Appended section 19 to `reference.html` and updated its status notice.

**Validation:** All 36 tests pass, including the reference link/self-containment check; Ruff passes. Slide numbers and content were checked against the export and screenshots of slides 1–5 and 7–13.

**Limits:** No change to code, data, training or evaluation. The Google deck was not edited, and its slide numbers will drift as the team edits it. The neural slides describe built software and a planned experiment; there is still no neural Amazon result. The mixture denominator and classical/neural split alignment remain team decisions.
