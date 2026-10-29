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

## 29 September 2026 — Slide figures

**Tool:** Claude Code (Claude Opus 5.5), with headless Chrome to render figures and the in-app browser for read-only viewing of the team deck.

**Request:** Diagrams for the deck, built one slide at a time: the motivation (train once versus keep updating), the proposed-method pipeline, the classical models, two slides each on how the QR model and SASRec handle the data, fit and predict (with tensor shapes and outputs over product IDs), how mixing in Movies & TV works by tensor shape, and next item (Hit@10) versus next items (Recall@10). Then open and merge a PR with anything else useful.

**Assistance:** Read the classical and neural code, configs, design docs, issue #8 and the `classical-study` branch, and drew each figure from them rather than from the deck text. Added nine HTML figures with PNG exports under `slides/figures/`, plus an index and a pointer from `slides/README.md`. Matched the deck's colours by reading its public preview; the deck itself was not edited. Recorded Hongyi's statement that the full classical study keeps only the QR model in `CONTEXT.md`, noting that `MarkovQRSVD` still builds and stores the full transition table, so dropping the Markov baseline saves no training cost. Updated the figures to the issue #8 metric decision after it was made.

**Findings raised with the user:** on `main`, classical `sweep --split test` scores from the latest training review, so it predicts two steps ahead (fixed on the `classical-study` branch); SASRec's output is a score per product, not a probability, and its training loss is a yes/no loss on two products per position; for mixing, Electronics predictions can change only through the shared weights (about 28,500 numbers) as long as random products for Movies & TV examples come from Movies & TV, which today's sampler does not guarantee.

**Limits:** Figures only; no code, data, training or evaluation changed. Example products, ranks and bar heights are illustrative. The mixing figure describes a planned design. Figures that cite the QR test refit, Hit@10 checkpoint selection or the next-items split depend on PR #9 as it stood on 28 September.

## 16 October 2026 — Boilerplate reference and integration of main

**Tool:** OpenAI assistant in Pi, with a fresh-context read-only scout and independent reviewer.

**Request:** After reviewing a lightweight two-package reset and its continual-training ABC, apply that paradigm to main's actual implementations, open a PR into main and tag the boilerplate branch as a reference.

**Reference:** The earlier user-authorized reset distilled the October meeting, replaced trial code with standard-library-only contracts, a counting fixture, fixed single-target evaluation, a thin notebook and executable development examples. It was published separately as `feat/boiler-plate`, commit `c169032`; its detailed AI-use record remains on that branch. Before that reset, four pre-existing uncommitted trial files were copied byte-for-byte with checksums to ignored local storage. This follow-up tagged the untouched snapshot `boilerplate-v1` and pushed the annotated tag. No private transcript, trial report, dataset or checkpoint was imported into the integration.

**Assistance:** Started `feat/shared-package-integration` from main `8b9edd3`, without merging the destructive reset. Consolidated protocol ownership under `cp4285.experiment` and algorithms/state under `cp4285.models`. Added a generic learner ABC and an actual SASRec learner owning optimizer and Python/CPU/device Torch RNG; adapted the real neural pilot and incremental PureSVD stream to the shared lifecycle. Retained the original SASRec architecture, initial all-position loss, fresh-target continuation, reset-once Adam policy, classical numerical methods, separate evaluators, CLI/config and artifact/download safeguards. Classical batch-only kernels remain independent refits, not falsely labelled continual learners. Added candidate/state/fork regressions, a stream-caller check, executable guide/notebook and updated provenance/layout documentation. Historical HTML source links are pinned to pre-integration main; a new dated checkpoint describes the current layout.

**Validation:** Locked sync, Ruff lint/format, all **42 tests**, CLI doctor/classical help and source/wheel builds passed. The suite executes both guide examples and notebook cells in fresh Python processes and includes synthetic checkpoint/frozen-control checks. CPU tests verify unchanged native updates, candidate order, scoring purity, copied optimizer bindings, persistent optimizer steps, reset-once guards and parent/sibling RNG isolation. Fresh LSP checks found no code errors; stale pre-sync dependency-import diagnostics and test-path false positives were dispositioned separately. Independent read-only review reported no blocking findings; it inspected tests but did not independently execute them.

**Limits:** This is a common lifecycle and responsibility layout, **not** a common scientific benchmark. Native update/exposure units, splits, candidate catalogues, seen masking and tie policies remain distinct. Initial weights are selected on validation; continuation resets Adam and uses end-of-initial sampler state, not a resumed best-epoch optimizer. Owned Torch RNG and avoiding a discarded network initialization on reload change historical continuation trajectories, so new runs need code-revision provenance. Sequential in-memory forks do not provide thread-safe global RNG scopes or durable resume; accelerator lifecycle checks remain unverified. No datasets were downloaded, no real-data experiment or course submission was performed, the Google deck was untouched, and PR #9's separate classical study was not incorporated. PR publication is for review, not authorization to merge main.

## 19 October 2026 — LLM continual-recommendation direction and PESO replication

**Tool:** Claude Code (Claude Opus 5.5), with web search and fetch for papers, the GitHub CLI for reading the upstream repository, and Modal for GPU runs billed to the user's account.

**Request:** Find a way to bring LLMs into the study of scalability and continual learning / catastrophic forgetting, changing one variable at a time and with an interpretable readout such as a counter of predictions that got better or worse. After discussing LLM-analysis methods, GenRec (arXiv 2608.10257) and PESO (arXiv 2510.25093), the user chose to replicate PESO first, list its future directions, and run it on Modal with A100s and Python 3.10.

**Assistance:**
- Read the project docs and code. Summarized how outside researchers analyse LLMs: scaling sweeps, learning-vs-forgetting trade-offs, before/after flip counts, representation analysis, causal rollback and data attribution.
- Read PESO's paper and its unlicensed code at a pinned commit, in ignored `local/peso/` only.
- Verified from code and data:
  - the PESO penalty covers only the LoRA matrices;
  - there is no target leakage across blocks;
  - the item tokenizer covers items that arrive later;
  - the released datasets differ from the paper's;
  - only one seed is used.
- Wrote our Modal launcher, a retention-matrix evaluator that reuses upstream functions, a user-group labeller and a summariser. Added `cp4285 flips` (retention matrix, backward transfer, kept/lost/gained counts, McNemar tests, forgetting events, group slices) with tests. Documented everything in `docs/PESO-REPLICATION.md`.
- Diagnosed environment failures:
  - Hugging Face gated access, granted by the user;
  - an `hf_xet` log file left open on the Modal volume;
  - an upstream block-0 copy race;
  - a transformers 4.52.4 best-checkpoint race between ranks.
  
  The last two led to one GPU with gradient accumulation 4, keeping the effective batch at 32.
- Diagnosed lost Modal runs: a local network disconnect cancelled detached runs, and GPU preemption restarted two seeds. Moved launches server-side, cancelled seed 42 at the user's request, and migrated it to a non-preemptible Runpod on-demand pod with a network volume. The user approved the move; `pod_run.py` runs it.
- At the user's request, audited the workstream against `AGENTS.md`:
  - checked the Table 2 targets, averaging, λ, Table 3 groups and future directions against the arXiv PDF (pp. 8–10, 27);
  - moved account-specific IDs into ignored `local/WORKSPACE.md`;
  - added the dated `reference.html` checkpoint;
  - reran the required checks.

**Validation:** `uv sync --locked`, `cp4285 doctor`, 47 tests, Ruff lint/format and a fresh synthetic demo pass. A debug smoke run on Modal completed every stage. On block 1, our evaluator matched upstream `continual_test.py` exactly, but only on an all-zero result from a 100-example model; the non-zero comparison is pending the full run.

**Result (20 October):** three seeds completed (43 and 44 on Modal, 42 on Runpod). Pretrain and single evolving LoRA reproduce Table 2 within about 5%. PESO ties single evolving LoRA: paired NDCG@10 difference −0.0001 ± 0.0002, against the paper's +0.0012. A Runpod host-maintenance restart cost one PESO rerun and about 1 h 45 min of idle GPU time.

**Deliverables (20 October):**
- At the user's request, a progress-update format (`docs/updates/`) modelled on the 15 October meeting notes, with a first entry for the replication.
- `notebooks/02_peso_replication.ipynb` with metric files committed under `replication/peso/results/`, plus an execution test.
- `summarize.py` refactored to expose paired differences.
- The Gemini meeting notes were treated as unconfirmed: their "recall@20" conflicts with the repo's meeting synthesis.
- A Jacobian-lens probe (Gurnee et al. 2026) is proposed but not run.

**Limits:** No per-user significance test or retention matrix yet. The candidate causes of the PESO gap (environment deviations, unreleased data processing) are untested; see `docs/PESO-REPLICATION.md`. Results are on PESO's released Instruments blocks, not our Electronics/Movies & TV data, and are not comparable with SASRec or classical numbers. None of the extension directions has been run. The two environment deviations are documented. Nothing was committed, pushed or submitted, and the authors were not contacted.

## 20 October 2026 — PESO transfer analysis, penalty size, J-lens probe and Colab weights

**Tool:** Claude Code (Claude Opus 5.5), with Runpod (an RTX 4090 pod) and Modal billed to the user's account, and the GitHub CLI for releases on the user's fork and the team repository.

**Request:** Make the Colab notebook work for another user by publishing the trained weights as a GitHub release and having the notebook download them; at the user's request, push the work to a personal fork for testing, clone the fork first, and publish the fork's release; carry on with the transfer matrix and the J-lens probe on the pod.

**Assistance:**
- Built a flattened, deduplicated weights bundle (`release_bundle.py`; 13 assets, 9.71 GB, Llama 3.2 licence, notice and "Built with Llama" README).
- Re-downloaded six files that `modal volume get` had corrupted, verifying each one against the manifest.
- Uploaded the bundle to a draft release on the team repository and a release on the fork. Every asset matches by GitHub's SHA-256 digest.
  - The fork release was published at the user's request.
  - The team release stays a draft.
- Reworked `notebooks/03_peso_colab.ipynb`:
  - it clones `REPO` at `BRANCH` first and reads recorded numbers from the clone;
  - it installs `scipy`, which upstream imports;
  - it retries downloads and shows the log tail when an evaluation fails.
  
  The test now runs the comparison cell on the committed files.
- At the user's request, added a J-lens side note to notebook 03: a three-panel figure (readout quality by layer against chance, drift split by state, output vs internal forgetting) from the committed seed-42 report, followed by its implications. The figure was rendered in a throwaway matplotlib environment and checked visually; matplotlib is not added to the project's dependencies.
- Ran the seed-42 transfer matrix: every saved state on every block, 20 beams, at most 2,500 users per block. Wrote `transfer_report.py` (per-method flips plus head-to-head McNemar tests).
- Ran the J-lens probe and its drift split (`jlens.py`, `jlens_report.py`).
- Measured PESO's penalty from checkpoints (`penalty_size.py`). It is about 10⁻⁶, against a loss of about 2, because the softmax over every matrix entry divides the KL by the matrix size (finding 10). This is the measured reason PESO tracks single LoRA in the released code.
- Committed aggregate reports only, under `replication/peso/results/seed42/reports/` (no user IDs), and added notebook 02 sections for them.
- Moved the private Hugging Face backup's name into ignored `local/WORKSPACE.md`.
- Stopped the pod; the network volume keeps the checkpoints and per-user dumps.

**Validation:** 49 tests and Ruff lint/format pass. The notebook-02 test pins the new report numbers. Anonymous downloads from the public fork release return the manifest byte-identical. The first Modal run of the new-user notebook test crashed on import (`Path(__file__).parents[2]` inside the container) and was cancelled; after the fix the container imported cleanly.

**Limits:** The transfer matrix, flips and J-lens cover one seed. The J-lens reads only the first semantic-ID token. The penalty finding concerns the released code at `edbec5f`, not necessarily the authors' unreleased setup. Commits were made and pushed only to the user's fork at the user's request; nothing was pushed to the team repository's branches, and nothing was submitted or sent to the authors.

## 21 October 2026 — Item-level J-lens, penalty calibration, O8 and the general-audience talk

**Tool:** Claude Code (Claude Opus 5.5), with Runpod (RTX 4090 and A100 pods), Modal and the GitHub CLI.

**Request:**
- Broaden the J-lens (the user chose item-level, all layers).
- Run the "settle it" experiment for PESO.
- Write a talk for a college-educated non-specialist.
- Check the Colab notebook still works.
- Check progress every 15 minutes.

**Assistance:**
- Wrote `jlens_items.py` and `jlens_items_report.py`. The level-0 and state-1 drift checks reproduce the first-code run.
- Corrected an earlier inference: forgetting sits in the broad first code, not the exact item.
- Wrote `penalty_calibrate.py`, `penalty_variants.py`, `variant_report.py` and `variant_jlens_compare.py`, and added `calibrate` and `variant` stages to `pod_run.py`. The size-normalized penalty was unit-tested on the pod against a hand computation before training.
- Ran O8 (λ = 400, seed 42). It gave a small, non-significant retention gain, no Table 2 gain, a 26–50% smaller inner drift, and a fine-code learning cost on the newest block.
- Added a two-figure J-lens side note to notebook 03 (rendered and checked locally; matplotlib not added to the project), notebook 02 sections and tests.
- Wrote `docs/TALK-GENERAL-AUDIENCE.md` and the 21 Oct progress update.
- Fixed the Modal notebook test twice:
  - an import that crashed inside the container;
  - parsing of nbformat's list-or-string text, which lost one finished run (about $2).
- Infrastructure:
  - **Pods:** two pods hit Runpod's EUR-IS-1 registry DNS failure and were terminated within minutes.
  - **Full volume:** it stalled the O8 publish while the A100 idled for about 3.8 h. At the user's choice, the 19 Oct debug runs' weights (82.6 GB) were deleted and the checkpoint was published by hand with SHA-256 checks.
  - **Monitoring:** a 15-minute health check with a volume write test ran until completion.

**Validation:**
- **Tests:** 51 tests, Ruff lint and format pass. Notebook tests pin the item-level, penalty and O8 numbers to committed reports.
- **Colab:** the fresh-user notebook ran end to end on Modal (A100-40GB, Python 3.12) with no cell errors. The PESO − single LoRA gap reproduced exactly, and Pretrain was about 13% lower from environment numerics; the released weights were verified lossless.

**Limits:**
- **Single seed:** O8 and both J-lens probes cover seed 42 only, and O8 has a single λ.
- **Teacher forcing:** codes 2–4 are read with the correct earlier codes fed in.
- **Correlations:** the drift–forgetting links are correlations.
- **External actions:** no team contact or submission. Commits are pushed only to the user's fork.

## 21 October 2026 — O2 and λ-sweep tooling (seed 42, not yet run)

**Request:** the user chose O2 (penalty scope) and the λ sweep as the next directions, to relate penalty strength and scope to retention and learning.

**Assistance:**
- Added `--item_penalty_weight` to `penalty_variants.py`: weight × ||W − W_start||² / 2 over the trained item layers, anchored at each block's start. Upstream's frozen originals hold the base model, so they cannot serve as the anchor.
- Extended `pod_run.py` method names to `peso-size-w<λ>-i<μ>`. Added an `item-drift` stage running the new `item_drift.py`, which measures item-layer drift per block and the μ whose pull matches the task gradient (the O8 calibration rule).
- Drew an O2-motivation figure from the committed O8 J-lens report, outside the repo (matplotlib not added).

**Validation:**
- **CPU smoke test** under transformers 4.52.4: the penalty is zero on a block's first loss, adds exactly μ·Δ to the item-layer gradient, works in the evaluation path and re-anchors for the next block. `item_drift.py` reports the right rows and weights on synthetic checkpoints.
- **Ruff:** lint and format pass.

**Limits:**
- **Not yet run on GPU:** no O2 or λ-sweep results exist yet.

## 22 October 2026 — Item subcategories from metadata

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Use the `categories` field of Amazon 2023 item metadata to group items into subcategories across all categories, fill empty or missing paths with a text classifier, run it online and publish the result.

**Assistance:** Added `src/cp4285/experiment/metadata.py`, the `meta-download`, `meta-audit`, `meta-classify`, `meta-group` and `meta-publish` commands, the optional `meta` extra and `tests/test_metadata.py`. Factored the validated download into `fetch` in `experiment/data.py`; `download --domain all` skips the six categories without 5-core files; DuckDB limits can be raised through the environment. Sampled every metadata file and found six with no paths, so they learn from a donor file. An embedding classifier was tried first on a Runpod GPU; that pod stopped over a billing issue. A hashed TF-IDF linear model replaced it after it scored higher on Video_Games (0.859 against 0.770 held-out accuracy) and ran about 20 times faster on CPU. Ran the full pipeline on a Runpod CPU pod, fixed two DuckDB failures found there, published to the private Hugging Face dataset `randomwish/amazon23-item-categories` and terminated the pods. First developed on `item-categories` (from `classical-study`, PR #13) and ported onto main's `experiment/` layout for a focused PR.

**Validation:** Ruff lint and format and the full test suite passed. The output reconciles to 35,393,189 unique rows; published counts match the local copy.

**Limits:** Predicted labels and donor mappings are model output and assumptions. Genre-level files (Books, CDs, Movies) classify poorly. No subcategory has been chosen as an experiment domain or checked against the cutoff protocol.

## 22 October 2026 — Shared ranking metrics

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Look across the open PRs for a standard metric, then report Hit@5, Hit@10, NDCG@10 and other suitable metrics in every workstream.

**Assistance:** Compared the metric code on main and PRs 9, 12 and 14 and the decision in issue #8. Added `src/cp4285/experiment/metrics.py` (`from_ranks`, `selection_key`) and routed the classical and neural evaluators through it. Renamed the single-target `recall` sums to `hit`, keeping `recall@K` as an alias. Switched the pilot's checkpoint selection to Hit@10 with NDCG@10 as tie-break and updated the docs that named NDCG@10 as primary.

**Validation:** Ruff and the full suite passed (53 tests, 3 new on hand-computed ranks). The synthetic demo ran end to end.

**Limits:** No real-data run used these metrics yet. Selection by Hit@10 is a proposal from PR #9 pending team confirmation. Evaluators still differ in candidates, seen-item masking and splits, so their scores remain incomparable.

## 22 October 2026 — Adapter survey

**Tool:** Claude Code (Claude Opus 5.5), with a research subagent.

**Request:** Look into whether adapters help with sparsity here: LoRA, adapters on SASRec and other suitable approaches.

**Assistance:** Wrote `docs/ADAPTERS.md`, a sourced survey of LoRA and continual LoRA, bottleneck adapters, adapter transfer in sequential recommenders, prompt tuning, LLM recommenders, category-aware models and replay. It ends with a comparison table and one proposed experiment. Parameter counts come from the repo's `SASRec` class; local data figures come from the PESO category statistics.

**Validation:** The parameter counts (2,704,000 total, 98.9% in the item table, 8,000 for rank-8 LoRA), density ratios (481–1,349×) and the 13,038 items first seen after block 0 were recomputed independently. Fu et al.'s Tables 2–4 were checked against the PDF. Every other source was checked online by the subagent.

**Limits:** No adapter or category-channel code was written or run. The recommended experiment is an agent proposal, not a team decision. Paper figures use each paper's own protocol, several with sampled metrics.

## 22 October 2026 — Category-channel SASRec on PESO's blocks

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Run the adapter survey's proposed experiment, SASRec with a category channel, on the cluster's CPU nodes.

**Assistance:** Added `models/category_sasrec.py`, a second SASRec encoder over category sequences whose output joins the item encoder's and predicts the next category, inside the unchanged `SASRecLearner`. Added `experiment/blocks.py` and `cp4285 blocks`: PESO's users, blocks and targets; block-0 training with the pilot's regime; fresh-target continuation after one Adam reset; validation-best epochs; and the full transfer matrix with exposure buckets and a popularity baseline. Documented the study in the runbook.

**Validation:** Six new tests cover the model, the split and an end-to-end run of both arms on toy blocks; the suite (59 tests) and Ruff pass. A one-epoch run on the real blocks took 112 s (SASRec) and 193 s (category arm) on 4 threads.

**Results:** three seeds per arm on the NUS SoC cluster's CPU nodes, with the continuation learning rate ($10^{-4}$) chosen on block-1 validation of the plain arm. The channel leaves top-10 ranking unchanged and lifts rare targets' ranks about threefold, still far outside the top 10 (see `docs/ADAPTERS.md`).

**Limits:** One category level (PR 14's level 2) and one learning-rate sweep on one seed. The category design follows CaFe but is our own, and its ranking rule is one of several reasonable choices.

## 22 October 2026 — Category targets on the NUS SoC cluster

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** redo PESO on the SoC cluster with PR 14's item subcategories instead of items, using the standard metrics (Hit@5, Hit@10, NDCG@10 and others suitable).

**Assistance:**
- Joined PESO's items to Amazon 2023 metadata by exact title and path, and to PR 14's level-2 labels (`category_index.py`): all 53,509 items, 30 categories, no unknowns. Downloaded the Musical_Instruments metadata (155 MB) with PR 14's `cp4285 meta-download` for the join.
- Added `category_report.py` and `category_summary.py`: standard metrics through `cp4285.experiment.metrics` (copied unchanged from branch `standard-metrics`), popularity and history baselines on the same users, observed-label targets, paired PESO − LoRA and backward transfer.
- Made `pod_run.py` paths, dataset, index, venv and model configurable through `PESO_*` variables, with unchanged defaults. Added `replication/peso/slurm/`: a bootstrap, an environment build on node-local disk (cluster NFS writes about 20 files per second), the job wrapper and a per-seed job chain.
- Found that upstream evaluation samples inside beam search (finding 11). The first smoke run crashed on one-token targets; a CPU reproduction confirmed the cause. The category runs decode deterministically through a base-model copy whose generation config sets `do_sample: false`.
- At the user's go, queued 3 seeds × (pretrain, single LoRA, PESO, upstream test, full transfer matrix) on H100 NVL GPUs, estimated at 12–21 GPU-hours.

**Validation:**
- The second smoke run passed. Upstream and our evaluator agree exactly on block 1, and `category_report.py` reproduces their five metrics from the dump.
- New tests check report and summary values against hand-computed ranks; the suite (56 tests) and Ruff pass.
- The base-model copy's SHA-256 matches the documented revision checksum.

**Limits:**
- Category scores answer a 30-way question and are not comparable with item-level scores. Hit@10 saturates; Hit@1, Hit@5, NDCG@10 and MRR@10 carry the comparison.
- Category runs use H100 NVL GPUs and deterministic beam search, where the item replication used A100s and the released beam sampling.
- 6.5% of item labels come from PR 14's classifier; results are also reported for observed-label targets.

## 22 October 2026 — Item subcategories from metadata

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Use the `categories` field of Amazon 2023 item metadata to group items into subcategories across all categories, fill empty or missing paths with a text classifier, run it online and publish the result.

**Assistance:** Added `src/cp4285/experiment/metadata.py`, the `meta-download`, `meta-audit`, `meta-classify`, `meta-group` and `meta-publish` commands, the optional `meta` extra and `tests/test_metadata.py`. Factored the validated download into `fetch` in `experiment/data.py`; `download --domain all` skips the six categories without 5-core files; DuckDB limits can be raised through the environment. Sampled every metadata file and found six with no paths, so they learn from a donor file. An embedding classifier was tried first on a Runpod GPU; that pod stopped over a billing issue. A hashed TF-IDF linear model replaced it after it scored higher on Video_Games (0.859 against 0.770 held-out accuracy) and ran about 20 times faster on CPU. Ran the full pipeline on a Runpod CPU pod, fixed two DuckDB failures found there, published to the private Hugging Face dataset `randomwish/amazon23-item-categories` and terminated the pods. First developed on `item-categories` (from `classical-study`, PR #13) and ported onto main's `experiment/` layout for a focused PR.

**Validation:** Ruff lint and format and the full test suite passed. The output reconciles to 35,393,189 unique rows; published counts match the local copy.

**Limits:** Predicted labels and donor mappings are model output and assumptions. Genre-level files (Books, CDs, Movies) classify poorly. No subcategory has been chosen as an experiment domain or checked against the cutoff protocol.

## 22 October 2026 — Shared ranking metrics

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Look across the open PRs for a standard metric, then report Hit@5, Hit@10, NDCG@10 and other suitable metrics in every workstream.

**Assistance:** Compared the metric code on main and PRs 9, 12 and 14 and the decision in issue #8. Added `src/cp4285/experiment/metrics.py` (`from_ranks`, `selection_key`) and routed the classical and neural evaluators through it. Renamed the single-target `recall` sums to `hit`, keeping `recall@K` as an alias. Switched the pilot's checkpoint selection to Hit@10 with NDCG@10 as tie-break and updated the docs that named NDCG@10 as primary.

**Validation:** Ruff and the full suite passed (53 tests, 3 new on hand-computed ranks). The synthetic demo ran end to end.

**Limits:** No real-data run used these metrics yet. Selection by Hit@10 is a proposal from PR #9 pending team confirmation. Evaluators still differ in candidates, seen-item masking and splits, so their scores remain incomparable.

## 22 October 2026 — Adapter survey

**Tool:** Claude Code (Claude Opus 5.5), with a research subagent.

**Request:** Look into whether adapters help with sparsity here: LoRA, adapters on SASRec and other suitable approaches.

**Assistance:** Wrote `docs/ADAPTERS.md`, a sourced survey of LoRA and continual LoRA, bottleneck adapters, adapter transfer in sequential recommenders, prompt tuning, LLM recommenders, category-aware models and replay. It ends with a comparison table and one proposed experiment. Parameter counts come from the repo's `SASRec` class; local data figures come from the PESO category statistics.

**Validation:** The parameter counts (2,704,000 total, 98.9% in the item table, 8,000 for rank-8 LoRA), density ratios (481–1,349×) and the 13,038 items first seen after block 0 were recomputed independently. Fu et al.'s Tables 2–4 were checked against the PDF. Every other source was checked online by the subagent.

**Limits:** No adapter or category-channel code was written or run. The recommended experiment is an agent proposal, not a team decision. Paper figures use each paper's own protocol, several with sampled metrics.

## 22 October 2026 — Category-channel SASRec on PESO's blocks

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Run the adapter survey's proposed experiment, SASRec with a category channel, on the cluster's CPU nodes.

**Assistance:** Added `models/category_sasrec.py`, a second SASRec encoder over category sequences whose output joins the item encoder's and predicts the next category, inside the unchanged `SASRecLearner`. Added `experiment/blocks.py` and `cp4285 blocks`: PESO's users, blocks and targets; block-0 training with the pilot's regime; fresh-target continuation after one Adam reset; validation-best epochs; and the full transfer matrix with exposure buckets and a popularity baseline. Documented the study in the runbook.

**Validation:** Six new tests cover the model, the split and an end-to-end run of both arms on toy blocks; the suite (59 tests) and Ruff pass. A one-epoch run on the real blocks took 112 s (SASRec) and 193 s (category arm) on 4 threads.

**Results:** three seeds per arm on the NUS SoC cluster's CPU nodes, with the continuation learning rate ($10^{-4}$) chosen on block-1 validation of the plain arm. The channel leaves top-10 ranking unchanged and lifts rare targets' ranks about threefold, still far outside the top 10 (see `docs/ADAPTERS.md`).

**Limits:** One category level (PR 14's level 2) and one learning-rate sweep on one seed. The category design follows CaFe but is our own, and its ranking rule is one of several reasonable choices.

## 23 October 2026 — Per-user dumps for the PESO-block study

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Experiment 3 of the category study asks whether SASRec and PESO can recommend products they never trained on, which needs SASRec's per-user rankings.

**Assistance:** Added `--dumps` to `cp4285 blocks`. Each cell then writes one row per test user (PESO item IDs, the exact target rank and the top 20 by score) inside the ignored run folder. Reran both arms for three seeds on the cluster's CPU nodes with the flag.

**Validation:** A new test checks that the dumps agree with each cell's metrics; the suite (61 tests) and Ruff pass. All six reruns reproduce the 22 October metrics exactly on the original CPU type. One extra category-arm rerun on a different CPU type drifted to own-state Hit@10 0.0181 from 0.0190, recorded as a hardware sensitivity.

**Limits:** Dumps hold user IDs, so they are never committed; only aggregate reports from the PESO branch are.

## 23 October 2026 — Decoding check, category results and experiments 1–4

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** extend the category study. Check upstream's decoding, compare item and category models, try category-prefixed item IDs, then run four designed experiments: a granularity ladder, exploration against repeats, new items and detectable effects. Report every earlier block after each update.

**Assistance:**
- Rebuilt the released seed-43 checkpoints from the GitHub release (`release_ckpt.py`). Re-evaluated them with sampling and deterministic decoding (`pod_run.py table2`), then with five more sampling draws for experiment 4.
- Added `category_projection.py` (item predictions read as categories), `category_prefix.py` (a category token before the item codes), `category_explore.py` (experiment 2), `new_items.py` (experiment 3), `noise_components.py` (experiment 4) and `ladder.py` (experiment 1). Built level-3 and level-4 category data with PR 14's pipeline for experiment 1.
- At the user's go, queued the ladder (two levels, three seeds), the prefix run (two arms) and the seed-43 deterministic transfer matrix on the cluster, estimated at about 34 GPU-hours.

**Validation:**
- Tests check the category scripts, the variance components, the exposure groups and the ladder summary against hand-computed values; the suite (66 tests) and Ruff pass.
- The deterministic product matrix reproduces the earlier deterministic Table 2 lists exactly for all 17,646 user–cell pairs.
- Cluster incidents and their fixes (port clashes on shared nodes, out-of-memory on 46 GB GPU shares, aarch64 nodes) are recorded in ignored `local/WORKSPACE.md`.

**Limits:**
- Experiment 2, 3 and 4 results are preliminary. Category seed 44, the prefix run's deterministic item seed and the ladder are still running.
- Product seed variance rests on three sampled seeds, so its 95% interval is wide.

## 23 October 2026 — Per-user dumps for the PESO-block study

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** Experiment 3 of the category study asks whether SASRec and PESO can recommend products they never trained on, which needs SASRec's per-user rankings.

**Assistance:** Added `--dumps` to `cp4285 blocks`. Each cell then writes one row per test user (PESO item IDs, the exact target rank and the top 20 by score) inside the ignored run folder. Reran both arms for three seeds on the cluster's CPU nodes with the flag.

**Validation:** A new test checks that the dumps agree with each cell's metrics; the suite (61 tests) and Ruff pass. All six reruns reproduce the 22 October metrics exactly on the original CPU type. One extra category-arm rerun on a different CPU type drifted to own-state Hit@10 0.0181 from 0.0190, recorded as a hardware sensitivity.

**Limits:** Dumps hold user IDs, so they are never committed; only aggregate reports from the PESO branch are.

## 24 October 2026 — Corrupted streams and a language model with adapters

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** drop PESO, try a language model with an adapter, and align the study with the course's robustness framing (shill manipulation) while keeping the original continued-training protocol.

**Assistance:**
- Wrote the variant as a dated agent proposal in `docs/CONTINUED-TRAINING.md`, with a pointer in `docs/EXPERIMENT.md`. B is the protocol's third option, corrupted interactions; the mitigation is adapter isolation.
- Added `models/category_lm.py` (frozen language model, LoRA adapters, exact category ranking in one packed pass per user), `experiment/shilling.py` (random, average, bandwagon, segment and clone push attacks, with realism checks) and `experiment/adapter_blocks.py` (`cp4285 adapter-blocks`, `cp4285 adapter-summary`). Added the optional `llm` extra (transformers 5.20, peft 0.21).
- Built a cluster environment with the extra and ran a one-seed pilot: the shared start state and six branches (shared or isolated adapters; clean, bandwagon or clone shills at 3%).

**Validation:** Packed scores match one-sequence-per-candidate scoring to 1e-5 on SmolLM2-135M and on a tiny random Llama in the tests. Tests cover training, adapter freezing and rollback, state round trips, shill shares and sources, injected users' exclusion from evaluation, the runner end to end and the summary; the suite (74 tests) and Ruff pass.

**Limits:** One seed and one pushed category. Shills are synthetic, and bandwagon profiles are trivially detectable (AUC 1.00). The A task is PESO's released Instruments blocks.

## 24 October 2026 — Corrupted-stream grid across three domains

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** run the full grid so the result is not tied to one dataset.

**Assistance:**
- Added `experiment/block_categories.py` (`cp4285 block-categories`): PR 14's level-N category rule for any PESO-format domain, from its own item paths, with PR 14's text classifier for items without one. Built categories for Instruments (30), Video Games (43) and CDs & Vinyl (59).
- Added `--push-share` (the pushed category by test-target share, so no target is picked by hand), `--rollback-after` with `CategoryLMLearner.drop` (remove the poisoned adapter right after its block) and top-5 push rates. The summary now matches clean references on condition and pushed category.
- Wrote `grid_plans.py` and `adapter_grid.sbatch` in the PESO branch checkout (sequential runs per plan, resumable after a timeout) and submitted 12 plans, 114 runs, on the SoC cluster.

**Validation:** On Instruments the new labels agree with PR 14's on every observed item and on 94% of predicted ones. Tests cover the category builder, adapter dropping, immediate rollback, rule-chosen targets and push-matched summaries; the suite (78 tests) and Ruff pass. The three domains' cluster copies match the local ones by checksum.

**Limits:** Video Games and CDs & Vinyl test sets are smaller (866–1,949 users per block). Video Games labels rely on a classifier with 0.81 holdout accuracy for 4% of items. Yelp is excluded because its tags have no hierarchy.

**Result, 25 October:** all 114 runs completed (22.6 GPU-hours; one 3-hour timeout resumed by its follow-up job). Added `grid_aggregate.py` and `grid_checks.py` to the PESO branch checkout and wrote the results into `docs/CONTINUED-TRAINING.md`. The pilot's five predictions hold across the three domains, with seed counts reported for each; the rare-target arm has one seed, and Video Games' rare category was already in every clean top 10.

## 24 October 2026 — Cluster recovery and final results for experiments 2–4

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** recover the cluster runs after the scratch quota filled (the user approved clearing space), then finish what the results allow.

**Assistance:**
- Traced the 23 October job failures to the scratch quota. Each LoRA and PESO block kept a full 3.8 GB model, because upstream's `extract_modules_subset.py` declares its delete flag with `store_false` and `pod_run.py` passes it deliberately. With the user's approval, removed the smoke runs' checkpoints and the redundant full models (the evaluator reads only the final model and per-block subsets, as the released checkpoint's layout shows). Scratch went from 501 to 169 GB. Resubmitted 31 jobs resuming from published stages.
- Made the job watcher stop instead of retrying when the stujump connection drops, after repeated key-only logins appear to have triggered a temporary block.
- Added `prefix_report.py` (category-prefixed against plain item IDs) and the repeat share to `ladder.py`, both with tests. Scored category seed 44, the three-seed category summary, experiment 2 on three seeds, experiment 3 with a second PESO seed, the final experiment 4 report and the prefix run.

**Validation:** The nested suite (67 tests) and Ruff pass. Matrix jobs on the cleaned checkpoints load them as the released checkpoint's matrix did.

**Limits:** Experiment 1 waits for levels 3 and 4. Product results rest on two training seeds; the seed term of single LoRA − Pretrain stays unsettled.

## 24 October 2026 — Cluster job status check and ladder status notes

**Tool:** OpenCode (DeepSeek V4.1 Flash).

**Request:** check how the recommender jobs on the cluster were doing, then make the documentation state clear.

**Assistance:**
- Read the queue from `xlogin2` with `status.sh`, `squeue`, `scontrol show job` and `sacct`, plus `df`, `quota` and the runs' `done/` markers and logs. Nothing on the cluster was changed and no job was submitted, cancelled or resubmitted.
- Recorded the result: level 3 complete for seeds 42–44, level 4 complete for seed 43, level 4's matrices for seeds 42 and 44 running on H100 NVL nodes. Updated `CONTEXT.md` and added the ladder's status subsection to `docs/PESO-REPLICATION.md`. Job ids, the failure counts and the access state went to ignored `local/WORKSPACE.md`.

**Validation:** The status comes from Slurm's job records and the runs' `done/` markers, not from log tails alone. The new subsection states explicitly that a finished stage says nothing about its scores.

**Limits:** No level-3 or level-4 number is claimed; `ladder.py` needs the running matrices. The finish estimate (roughly 45 minutes to two hours per seed) comes from per-block timings in the current logs and can move with queue waits or a slow co-tenant on a shared H100.

## 24 October 2026 — Levels 3 and 4, the granularity ladder and experiment 2 at finer labels

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** check the results once the cluster runs finished.

**Assistance:**
- Moved level 4 seed 42 off node xgpi17, where PESO trained at about a third of the normal speed and timed out twice, and excluded that node from the pending jobs.
- Scored levels 3 and 4 (`category_report.py`, `category_summary.py`), ran experiment 1 across all rungs (`ladder.py`) and experiment 2 at levels 3 and 4 (`category_explore.py`). Replaced the in-progress ladder notes in `docs/PESO-REPLICATION.md` and `CONTEXT.md` with the results.
- Removed the finished runs' redundant per-block full models, as approved earlier for clearing scratch.

**Validation:** The suite (67 tests) and Ruff pass for the files this work added. The reports hold aggregates only. Scoring observed labels only leaves each level's update gain unchanged.

**Limits:** Product results rest on two training seeds. The ladder cannot separate product identity from the block-to-block shift as the cause of forgetting, since both change at the same rung.

## 24 October 2026 — Corrupted streams and a language model with adapters

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** drop PESO, try a language model with an adapter, and align the study with the course's robustness framing (shill manipulation) while keeping the original continued-training protocol.

**Assistance:**
- Wrote the variant as a dated agent proposal in `docs/CONTINUED-TRAINING.md`, with a pointer in `docs/EXPERIMENT.md`. B is the protocol's third option, corrupted interactions; the mitigation is adapter isolation.
- Added `models/category_lm.py` (frozen language model, LoRA adapters, exact category ranking in one packed pass per user), `experiment/shilling.py` (random, average, bandwagon, segment and clone push attacks, with realism checks) and `experiment/adapter_blocks.py` (`cp4285 adapter-blocks`, `cp4285 adapter-summary`). Added the optional `llm` extra (transformers 5.20, peft 0.21).
- Built a cluster environment with the extra and ran a one-seed pilot: the shared start state and six branches (shared or isolated adapters; clean, bandwagon or clone shills at 3%).

**Validation:** Packed scores match one-sequence-per-candidate scoring to 1e-5 on SmolLM2-135M and on a tiny random Llama in the tests. Tests cover training, adapter freezing and rollback, state round trips, shill shares and sources, injected users' exclusion from evaluation, the runner end to end and the summary; the suite (74 tests) and Ruff pass.

**Limits:** One seed and one pushed category. Shills are synthetic, and bandwagon profiles are trivially detectable (AUC 1.00). The A task is PESO's released Instruments blocks.

## 24 October 2026 — Corrupted-stream grid across three domains

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** run the full grid so the result is not tied to one dataset.

**Assistance:**
- Added `experiment/block_categories.py` (`cp4285 block-categories`): PR 14's level-N category rule for any PESO-format domain, from its own item paths, with PR 14's text classifier for items without one. Built categories for Instruments (30), Video Games (43) and CDs & Vinyl (59).
- Added `--push-share` (the pushed category by test-target share, so no target is picked by hand), `--rollback-after` with `CategoryLMLearner.drop` (remove the poisoned adapter right after its block) and top-5 push rates. The summary now matches clean references on condition and pushed category.
- Wrote `grid_plans.py` and `adapter_grid.sbatch` in the PESO branch checkout (sequential runs per plan, resumable after a timeout) and submitted 12 plans, 114 runs, on the SoC cluster.

**Validation:** On Instruments the new labels agree with PR 14's on every observed item and on 94% of predicted ones. Tests cover the category builder, adapter dropping, immediate rollback, rule-chosen targets and push-matched summaries; the suite (78 tests) and Ruff pass. The three domains' cluster copies match the local ones by checksum.

**Limits:** Video Games and CDs & Vinyl test sets are smaller (866–1,949 users per block). Video Games labels rely on a classifier with 0.81 holdout accuracy for 4% of items. Yelp is excluded because its tags have no hierarchy.

**Result, 25 October:** all 114 runs completed (22.6 GPU-hours; one 3-hour timeout resumed by its follow-up job). Added `grid_aggregate.py` and `grid_checks.py` to the PESO branch checkout and wrote the results into `docs/CONTINUED-TRAINING.md`. The pilot's five predictions hold across the three domains, with seed counts reported for each; the rare-target arm has one seed, and Video Games' rare category was already in every clean top 10.

## 26 October 2026 — Attack catalogue and rollback timing

**Tool:** OpenCode (GLM 5.3 Flash).

**Request:** build on the md files' attack and defence suggestions — the seven corruption ideas and the rollback-as-weapon reading — treating the learner abstractly as a language model with a LoRA adapter rather than extending PESO itself.

**Assistance:**
- Added `experiment/corruption.py`: one interface for the shill corruption (delegating to `shilling.py`) plus four corruptions of real activity — `real-history` (real shoppers' training events gain target items; accounts and every user's validation/test events untouched), `demotion` (shills copied from a rival category's customers, buying the promoted category), `vandalism` (uniform-noise profiles, no target) and `relabel` (a fraction of a block's items re-listed under the promoted category; the remapped map builds training rows only, evaluation keeps the true map). Every report records the category-mix shift and, where profiles exist, a separability AUC.
- Extended `experiment/adapter_blocks.py`: `--corruption`, `--placement` (one block, every block, or block 0 of the init stage — a poisoned base refuses rollback), `--rollback-block N` (delayed rollback; the summary's new `drop_effect` compares the dropped cell with the clean state at the same block, turning the late-rollback over-correction into a timing sweep), `--rival`/`--rival-share` and `--trigger`/`--trigger-share` (share rules resolve per domain, so no watched category is picked by hand; a trigger's measurement is split into `push_trigger`/`push_other`).
- Added `experiment/robustness.py` (`cp4285 robustness-plan`): the run matrix as JSON lines with ready `adapter-blocks` argv, stage A (rollback timing, low-and-slow, hijacked accounts) and stage B (demotion, hidden trigger, relabel, vandalism, poisoned starting data); it launches nothing.
- Wrote the catalogue, its modelling choices and the rollback-timing reading into `docs/CONTINUED-TRAINING.md`, with pointers in `docs/EXPERIMENT.md` and `docs/IMPLEMENTATION.md`.

**Validation:** 17 new tests: corruption shares and accounting, untouched evaluation events, demotion donors, relabel's true-map separation, delayed-rollback and placement validation, poisoned-init branches, rival/trigger cells and the planner's argv parseability and uniqueness. The suite (95 tests) and Ruff pass.

**Limits:** The catalogue's 63 seed-42 runs (three domains, stages A and B) completed on the SoC cluster on 26 October (~13 GPU-hours; run folders in that checkout's `lm-runs/`). One seed per cell: the headline comparisons — corrupting real accounts pushes 2–3× harder than fake ones at the same dose, a 10% relabel is the strongest and behaviourally undetectable attack, sustained 1% persists where a single block heals, the hidden trigger concentrates in its segment while the aggregate hides it, demotion promotes without burying the rival, vandalism at 3% harms nothing measurably, and the rollback-timing over-correction (+0.85 to +2.11 ranks as the undo is delayed) holds on Instruments and Video Games but not on CDs & Vinyl — all need seeds 43–44 before any claim. The modelling choices (training-map versus evaluation-map split for relabel, validation/test events untouched under real-history) are agent proposals, open to team revision. A first submission attempt failed because the planner omitted `--init` on branch lines; fixed in `experiment/robustness.py` with a regression test, and the jobs were resubmitted.

## 26 October 2026 — Three-seed check of the attack catalogue

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** write up what changed in the corrupted-stream study, its results and the reasoning between steps as an artifact; then update the docs and redo the write-up with more illustrations.

**Assistance:**
- Copied the 312 `metrics.json` files (303 grid runs, 7 pilot and 2 smoke runs) read-only from the SoC cluster's `lm-runs/` into a session scratch folder; no checkpoints, dumps or cluster state were touched.
- Recomputed every catalogue contrast for seeds 42–44: block-by-block push, top-5 share, rollback timing, a with/without-A1 decomposition of the rollback over-correction, the trigger split, the rival rank, the poisoned base, costs and noise floors.
- Replaced the seed-42 results in `docs/CONTINUED-TRAINING.md` with the three-seed results and listed which seed-42 readings changed. Updated `CONTEXT.md` and `reference.html` section 23.
- Corrected the PESO branch checkout's `docs/PESO-REPLICATION.md` (category counts per domain, clean baselines, the low-and-slow reading, rollback numbers).
- Published a private illustrated write-up as a claude.ai artifact.

**Validation:** the recomputed bandwagon, clone, low-and-slow, relabel and real-history tables match `docs/PESO-REPLICATION.md` within 0.01 ranks. The suite (95 tests) and Ruff pass on the working tree.

**Limits:** the decomposition compares separate drop runs with the attacked run, so it carries run-to-run noise (at most 0.20 ranks for identical clean configurations). Reasons given for the mechanisms are interpretations of these measurements.

## 26 October 2026 — Corrupted-stream grid analysis and write-up

**Tool:** OpenCode (DeepSeek V4.1 Flash).

**Request:** analyse the finished corrupted-stream grid, turn it into presentable narratives, and explain what the shared-vs-isolated comparison actually tests.

**Assistance:**
- Read the grid's 303 `metrics.json` files and computed block-by-block promotion, dose, rare-target, trigger, demotion, vandalism and rollback contrasts, each against the clean run of the same domain, seed and condition. Confirmed in `adapter_blocks.py` and `category_lm.py` that `condition` is adapter topology — `shared` updates one adapter across blocks; `isolated` calls `add_stream`, which freezes the adapters so far and sums them at scoring — so the shared-vs-isolated number is a paired difference of attacked-minus-clean effects, not two adapters inside one model.
- Added the corrupted-stream section to `docs/PESO-REPLICATION.md` with the design, the block-1-to-block-4 table, side-effect numbers and caveats; added a status bullet to `CONTEXT.md`. Cluster operations and job ids went to ignored `local/WORKSPACE.md`.

**Validation:** Every quoted number was recomputed from the run metrics, with sign counts over the nine domain–seed pairs. The section states the comparison unit and that the rank is over 30 categories.

**Limits:** Nine pairs for the core grid; dose only on Instruments; rare targets only seed 42. Effects below about 0.3 ranks are inside seed noise. The grid uses the adapter-blocks harness on category data, so it says nothing about PESO's penalty or the replication's null result.

## 26 October 2026 — Layperson write-up and shareable artifact

**Tool:** OpenCode (DeepSeek V4.1 Flash).

**Request:** turn the corrupted-stream analysis into a layperson write-up and upload an artifact.

**Assistance:**
- Wrote a self-contained HTML feature in ignored `reports/corrupted-stream-grid.html` and rendered `reports/corrupted-stream-grid.pdf` (8 A4 pages) with headless Chrome. It uses a shop-assistant analogy, six findings and inline CSS charts; no external assets or fonts.
- Corrected the sharp-attack framing while writing it. At block 1, relabelling reaches 88.6% top-5 (mean rank 3.7 against 11.4 clean) and history hijack 31.2%; both return to about 4% by block 4. The earlier "buys rank, not top-5" sentence described the block-4 residue and was misleading. `docs/PESO-REPLICATION.md` now separates the immediate spike from the residue and records the −0.85 pp NDCG@10 cost in the tampered block.
- Added a "The six bars, decoded" table beneath the top-five chart. It gives the exact mechanics and dose for each label: clean control; bandwagon bots (about half target-category items, the rest from the 50 most popular items; 3% of training examples); cloned real histories with about half replaced by target items (3%); the bandwagon recipe at 1% every period; target items inserted into real histories before the last two events (3%); and 10% of other listings relabelled in training only. The table stacks on narrow screens; print lets it flow across pages with a repeated header.
- Added a "Four things the 303 runs said" conclusions section with a green "Grok it" intuition box on each card (the lie fades; the drip beats the splash; forgetting is the defence; a nudge, not a break-in), each tied to the measured numbers.
- Uploaded the HTML and PDF to a secret GitHub gist; the rendered HTML is served through the statically CDN and htmlpreview. The gist URL and local paths are in ignored `local/WORKSPACE.md` and the chat. No repository or cluster state changed.

**Validation:** Every quoted number was read from the 303 grid metrics. The HTML was checked with headless-Chrome screenshots at desktop and phone widths, and the PDF page-checked (8 pages). The artifact is a communication piece, not a new result.

**Limits:** Layperson rounding; the page repeats the technical section's caveats. The gist is unlisted, not access-controlled, and the statically link is a third-party render of the raw gist file.

## 26 October 2026 — Corrections to the corrupted-stream section

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** update the docs after a three-seed recomputation of the corrupted-stream grid from its `metrics.json` files.

**Assistance:** In `docs/PESO-REPLICATION.md`, gave each domain's category count (30, 43, 59) and clean baseline range, stated that low-and-slow's block-4 value includes that block's own dose and holds only under isolated adapters, and replaced the rollback sentence with the measured timing (−0.10, 1.16, 1.45 and 1.85 ranks below clean for drops after blocks 1, 2, 3 and at the end). Matched `CONTEXT.md`.

**Validation:** the section's block-by-block table for bandwagon, clone, low-and-slow, relabel and history hijack matches the recomputation within 0.01 ranks.

**Limits:** documentation only; no runs or cluster state changed.

## 26 October 2026 — Attack catalogue and rollback timing

**Tool:** OpenCode (GLM 5.3 Flash).

**Request:** build on the md files' attack and defence suggestions — the seven corruption ideas and the rollback-as-weapon reading — treating the learner abstractly as a language model with a LoRA adapter rather than extending PESO itself.

**Assistance:**
- Added `experiment/corruption.py`: one interface for the shill corruption (delegating to `shilling.py`) plus four corruptions of real activity — `real-history` (real shoppers' training events gain target items; accounts and every user's validation/test events untouched), `demotion` (shills copied from a rival category's customers, buying the promoted category), `vandalism` (uniform-noise profiles, no target) and `relabel` (a fraction of a block's items re-listed under the promoted category; the remapped map builds training rows only, evaluation keeps the true map). Every report records the category-mix shift and, where profiles exist, a separability AUC.
- Extended `experiment/adapter_blocks.py`: `--corruption`, `--placement` (one block, every block, or block 0 of the init stage — a poisoned base refuses rollback), `--rollback-block N` (delayed rollback; the summary's new `drop_effect` compares the dropped cell with the clean state at the same block, turning the late-rollback over-correction into a timing sweep), `--rival`/`--rival-share` and `--trigger`/`--trigger-share` (share rules resolve per domain, so no watched category is picked by hand; a trigger's measurement is split into `push_trigger`/`push_other`).
- Added `experiment/robustness.py` (`cp4285 robustness-plan`): the run matrix as JSON lines with ready `adapter-blocks` argv, stage A (rollback timing, low-and-slow, hijacked accounts) and stage B (demotion, hidden trigger, relabel, vandalism, poisoned starting data); it launches nothing.
- Wrote the catalogue, its modelling choices and the rollback-timing reading into `docs/CONTINUED-TRAINING.md`, with pointers in `docs/EXPERIMENT.md` and `docs/IMPLEMENTATION.md`.

**Validation:** 17 new tests: corruption shares and accounting, untouched evaluation events, demotion donors, relabel's true-map separation, delayed-rollback and placement validation, poisoned-init branches, rival/trigger cells and the planner's argv parseability and uniqueness. The suite (95 tests) and Ruff pass.

**Limits:** The catalogue's 63 seed-42 runs (three domains, stages A and B) completed on the SoC cluster on 26 October (~13 GPU-hours; run folders in that checkout's `lm-runs/`). One seed per cell: the headline comparisons — corrupting real accounts pushes 2–3× harder than fake ones at the same dose, a 10% relabel is the strongest and behaviourally undetectable attack, sustained 1% persists where a single block heals, the hidden trigger concentrates in its segment while the aggregate hides it, demotion promotes without burying the rival, vandalism at 3% harms nothing measurably, and the rollback-timing over-correction (+0.85 to +2.11 ranks as the undo is delayed) holds on Instruments and Video Games but not on CDs & Vinyl — all need seeds 43–44 before any claim. The modelling choices (training-map versus evaluation-map split for relabel, validation/test events untouched under real-history) are agent proposals, open to team revision. A first submission attempt failed because the planner omitted `--init` on branch lines; fixed in `experiment/robustness.py` with a regression test, and the jobs were resubmitted.

## 26 October 2026 — Three-seed check of the attack catalogue

**Tool:** Claude Code (Claude Opus 5.5).

**Request:** write up what changed in the corrupted-stream study, its results and the reasoning between steps as an artifact; then update the docs and redo the write-up with more illustrations.

**Assistance:**
- Copied the 312 `metrics.json` files (303 grid runs, 7 pilot and 2 smoke runs) read-only from the SoC cluster's `lm-runs/` into a session scratch folder; no checkpoints, dumps or cluster state were touched.
- Recomputed every catalogue contrast for seeds 42–44: block-by-block push, top-5 share, rollback timing, a with/without-A1 decomposition of the rollback over-correction, the trigger split, the rival rank, the poisoned base, costs and noise floors.
- Replaced the seed-42 results in `docs/CONTINUED-TRAINING.md` with the three-seed results and listed which seed-42 readings changed. Updated `CONTEXT.md` and `reference.html` section 23.
- Corrected the PESO branch checkout's `docs/PESO-REPLICATION.md` (category counts per domain, clean baselines, the low-and-slow reading, rollback numbers).
- Published a private illustrated write-up as a claude.ai artifact.

**Validation:** the recomputed bandwagon, clone, low-and-slow, relabel and real-history tables match `docs/PESO-REPLICATION.md` within 0.01 ranks. The suite (95 tests) and Ruff pass on the working tree.

**Limits:** the decomposition compares separate drop runs with the attacked run, so it carries run-to-run noise (at most 0.20 ranks for identical clean configurations). Reasons given for the mechanisms are interpretations of these measurements.

## 27 October 2026 — Focused follow-up prep (metric lattice + wave-1 plans)

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** evaluate the proposed follow-up experiments (A–I), then prep wave 0/1 with a tighter focus, centred on metric-to-metric analysis of the existing 303 runs.

**Assistance:**
- Assessed the A–I list: the dose-matching arithmetic in proposal B is internally inconsistent (hijack 218 vs 262 lessons; bandwagon 135 vs 263); "the original task is untouched" overstated finding 8; and the reported trigger block-4 segment residue (0.62 ranks, 9/9 pairs) should be re-derived from existing metrics before any new runs.
- Added `replication/peso/grid_lattice.py`: a read-only metric-to-metric pass over the run folders. Each attacked run is paired against the clean run of the same domain, seed, condition and push target; the pass then relates dose (pushed lessons) -> spike -> decay slope -> residue (own-block and same-users) -> late NDCG cost -> retention cost -> separability (AUC/JS). `--inspect` prints one run's key tree so the cluster schema can be confirmed before the full pass; unknown keys are skipped loudly, never guessed.
- Added `replication/peso/slurm/focus_plans.py`: create-only plan TSVs for a narrowed wave 1 — dose-matched real-vs-fake (bandwagon 3% at target-share 0.8; hijack at 1.85%) and the rare-target replication for seeds 43–44, isolated layout only. 36 new runs, about 7 GPU-hours. Attack names and the hijack dose are placeholders pending the wave-0 inspect; regenerate before submitting.
- Deferred C (dose until it breaks) and the wider A/D/E/I set until the lattice shows which relation deserves an intervention.

**Validation:** Smoke-tested both scripts against a synthetic 10-run fixture in `/tmp` (paired pass, `--inspect`, create-only refusals on second invocation). `uv run ruff check` passes on the new files; `uv run pytest -q` (67 tests) passes. No cluster contact: the SSH session remains closed after the failed-password incident, so nothing has run against the real metrics and the schema assumptions are deliberately unverified (`--inspect` is the check).

**Limits:** The lattice's report/stealth key names are guesses to be confirmed by `--inspect`; correlations over 9 domain–seed pairs (fewer within a family) are descriptive, not inferential; wave-1 doses must be recomputed from real injection reports before submission.

## 27 October 2026 — Follow-up refocused on continual learning / forgetting

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** reconsider whether the prepped wave-1 runs help focus the investigation on continual learning and category forgetting.

**Assistance:**
- Judged the earlier wave-1 set against the project question: the dose-matched real-vs-fake arms (B) measure attack potency, not forgetting, and were cut; the dose escalation (C) stays deferred. Two arms remain because they measure forgetting directly: the rare-target replication (frequency-dependent forgetting, tying to the ladder's finding that forgetting appears at fine granularity) and low-and-slow that stops after block 2 (the forgetting curve of a sustained update against the one-off arms).
- Extended `replication/peso/grid_lattice.py` with the continual-learning payload: drop/rollback runs now become intervention records (rebound after deleting the attacked adapter, against clean and against the intact run — the counter-adaptation measurement), an optional `grid_aggregate.py` DRIFT_JSON adds natural drift as a covariate for the washout relations, and the summary prints rare-vs-common and intervention tables. The trigger segment split is flagged as needing the per-user dumps, not metrics.json.
- `replication/peso/slurm/focus_plans.py` regenerated for the refocused arms: 27 isolated-layout runs, about 5 GPU-hours.

**Validation:** Smoke-tested on the synthetic fixture including an intervention run and a drift JSON: relations, rare/intervention tables and create-only refusals all behave. `uv run ruff check` passes on both files; `uv run pytest -q` (67 tests) passes. Still no cluster contact.

**Limits:** Same as the earlier entry, plus: the drift relation has n = domains (3) and is descriptive; the drop-timing parser guesses from branch names and config keys until `--inspect` confirms the schema.

## 27 October 2026 — Wave 0 run: the metric-to-metric lattice over the 303 grid runs

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** run the prepped wave 0 (user's go) and feed the results back into the focused investigation.

**Assistance:**
- Copied the 312 run folders' `metrics.json` files (7.1 MB, metrics only) from the cluster to a local temp dir over the reopened SSH session; no cluster state changed. Debugged the schema against the real files: attack type lives in a per-run `attack` dict (`kind`: shill / real-history / relabel / demotion / vandalism) while `config.attack` is 'none' for most kinds; `slow1` keeps per-block reports; isolated attacked runs carry a `rollback` dict (scoring-time deletion of the block-1 adapter); newer-vintage branches pair with `clean-watch`; rare branches need their own clean at the same push target.
- Extended `replication/peso/grid_lattice.py` with what the schema revealed: kind-specific doses, the runner's own `push_trigger`/`push_other` segment split (no dumps needed), the rollback counterfactual, condition-split residue relations and an optional drift covariate. Added dependency-free `replication/peso/grid_drift.py` (mirrors the cluster's `load`/`training_examples`/`js`) and ran it on the login node for the drift covariate.
- Ran the full pass: 192 attacked runs paired, 54 interventions, 0 unmatched. Results committed under `replication/peso/results/grid/` and summarised in the new "Metric-to-metric lattice" subsection of `docs/PESO-REPLICATION.md`. Headlines: spike → residue is strong only under isolated adapters (ρ +0.66 vs shared +0.27); the segment trigger's block-4 residue is group-targeted (−0.62 ranks, 9/9) and hidden by the all-shoppers average; stealth and persistence line up only weakly; natural drift does not explain the washout; scoring-time deletion unmasks the counter-push (ρ −0.68); the delete arms' over-correction is magnitude-true but direction-mixed by domain.
- The dose question the earlier proposal B targeted is answered at the metric level: within-bandwagon dose → spike is real (ρ −0.57), but per-lesson efficiency differs by kind (~0.024 ranks/lesson for hijack and bandwagon against ~0.012 for relabel), so B as designed is unnecessary; the rare-target replication and low-and-slow-that-stop arms remain the wave-1 candidates.

**Validation:** Reproducibility check — a second full pass over the same metrics produces an identical JSON (`results/grid/lattice.json` == fresh run). Family means reproduce the doc's published numbers (relabel spike −7.56 pooled against −7.44/−7.68; rare-target residue −0.87/−0.89; drop-after-block-1 ≈ clean). `uv run ruff check` on the three scripts and `uv run pytest -q` (67 tests) pass. Cluster access was read-only: two `metrics.json` rsync/scp reads, two grep/sed reads of the cluster-side runner source, one stdout-only drift computation, one 2 KB script staged in the node's `/tmp`.

**Limits:** Correlations are descriptive (9 domain–seed pairs; fewer within a family; drift n = 3 domains). Clean pairing is vintage-based (rival_name → clean-watch); the doc's earlier drop-arm numbers averaged |Δ| over blocks 0–3, so the lattice's signed push-rank view is a refinement, not a contradiction. The wave-1 runs still need the user's go and the cluster runner's flag check.

## 27 October 2026 — Wave 1 launched: rare-target replication (18 runs)

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** launch the focused wave 1 on the cluster (user's go).

**Assistance:**
- Checked the cluster runner before submitting: the grid's low-and-slow used `--placement all`, and `--placement` accepts block|all|init only — so "attack blocks 1–2 then stop" has no existing flag. Deferred that arm rather than hot-patching the runner; the dose-matched arms had already been cut after the metric lattice answered the dose question.
- Reduced `replication/peso/slurm/focus_plans.py` to the rare-target replication (18 runs, ~3.5 GPU-h): clean-rare, bandwagon-3%-rare and clone-3%-rare for seeds 43–44 across the three domains, isolated layout. Dropped the plan-file init lines (all nine inits exist; a fresh-workspace init would train at the wrong push-share). Verified attack names against the cluster's own plan files.
- Copied the six `focus-*.tsv` plans to `$W/plans/` (new files, no overwrites) and submitted six `adapter_grid.sbatch` jobs: **1023419–1023424** (SRC=$W/cp4285-main/src). The cluster sbatch script is byte-identical to the repo's (md5 match).

**Validation:** Plans generated create-only and inspected line-by-line before copying; sbatch scripts hash-compared; two jobs reached RUNNING within a minute, four queued. Results land in `$W/lm-runs/lm-*-iso-*-rare` and pair against each run's own `clean-rare`.

**Limits:** The rare sweep stays seed-42-plus-two-seeds by design; frequency-dependent conclusions will rest on 3 domains × 3 seeds for the attacked arms. The low-and-slow-that-stop arm needs a placement change with tests before it can run.

## 27 October 2026 — Wave 1 results: rare-target persistence is a CDs effect

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** check the wave-1 results once the rare-target runs finished.

**Assistance:**
- All 18 new rare-target runs (plus seed 42's 9) completed; the queue is empty. Copied the new metrics locally and re-ran the lattice (now 204 attacked runs paired, 0 unmatched); refreshed `replication/peso/results/grid/lattice.json`.
- Result: the seed-42 "rare targets are easier and stick more" reading localises to CDs & Vinyl. There, rare pushes spike −4.3 to −7.4 ranks and keep a −2.71 residue at block 4 against +0.03 for standard targets (3/3 seeds, both attacks; 6/6 cd pairs still promoted). On Instruments and Video Games, rare targets behave like standard ones (all 12 residues within ±0.2). Retention stays near zero everywhere (worst −0.39 pp NDCG@10, one seed, mixed signs otherwise). Finding 7's one-seed claim is now domain-scoped; the 59-category space is the plausible mechanism but stays a hypothesis at n = 1 domain.
- Added the result as a "Rare-target replication" paragraph in the lattice section of `docs/PESO-REPLICATION.md`; updated `CONTEXT.md`.

**Validation:** Rare runs paired against their own `clean-rare` at the same push target (0 unmatched). Numbers cross-checked per domain and per seed before writing. No new cluster writes; metrics copied read-only.

**Limits:** n = 3 seeds in 1 domain for the strong effect; the mechanism (category-space size) is untested — a fourth domain or a matched-rarity sweep would settle it. The low-and-slow-that-stop arm is still deferred on the missing placement flag.

## 27 October 2026 — Discriminator result: label semantics, not tail geometry

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** run the prepped rare-target discriminator on the cluster (user's go).

**Assistance:**
- Verified candidates against the data before choosing: Alligator Records has 1 test target (degenerate), Rhino 0.92% (too frequent); selected Verve Music Group (jazz label, 0.29% share, 131 items) and Reggae (non-label genre, 0.17%, 57 items). Combined design using the runner's rival tracking: one clean run scores both targets, each attack run pushes one while tracking the other — 3 runs instead of 6 (`focus-cd-s42.tsv`, job 1024243).
- Read the results from the runs' push/rival cells: Verve spikes −10.4 ranks from its clean baseline of 24.7 and still holds −2.91 at block 4 (the Blue Note pattern); Reggae, buried deeper (34.0), spikes only −2.2, washes out, and over-corrects to +1.46 below its own clean baseline. The unpushed target moves ≤ 0.10 ranks throughout, so the pushes are cleanly isolated.
- Conclusion recorded in `docs/PESO-REPLICATION.md`: persistence follows label semantics — record-label categories co-purchase with popular music, so the poisoned association is consistent with genuine structure — not buried-tail geometry. Caveat noted: item-count is not fully separated from label identity at one seed and two categories.

**Validation:** Both runs' metrics present and paired against the same clean run; the unpushed-target deltas (≤ 0.10 ranks) serve as the isolation check. No cluster writes beyond the run outputs; the analysis ran on the cluster login node over the metrics files.

**Limits:** One seed, one attack (bandwagon 3%), two categories; clone confirmation and a second label pair would strengthen the claim. The rival cells were scored at the same states as the push cells, so the deltas share the clean baseline by construction.

## 27 October 2026 — E1 + E2: matched control refutes label semantics; two-condition account

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** run E1 (matched control) and E2 (segment-cut analysis) for the cohesive persistence story.

**Assistance:**
- E1 (2 runs, job 1024787): New Age as the matched control for Blue Note — test share 0.521% vs 0.546%, 245 vs 210 items, clean rank 24.3 vs 24.9, genre vs label. New Age spikes −11.5 and holds −3.32 at block 4; Blue Note, tracked unpushed in the same runs, moves ≤ 0.5 ranks. Label semantics refuted; the item-count confound from the Verve/Reggae pair is dead.
- E2 (zero GPU): computed the clean natural differential for the segment attack from the logged `push_trigger`/`push_other` cells — the trigger group (36% of users) naturally ranks the push target ~1 rank WORSE (+0.97), so the attack manufactured a group-specific preference against the group's lean; it persists at block 4 under isolated adapters (trigger −0.62, 9/9; other +0.24, 4/9) and washes under shared (−0.16, 4/9). The all-shoppers average is noise (−0.07).
- Synthesis across all eight pushed-category observations: persistence = no clean anchor (baseline rank ≳ 24 of 59) AND learnable presence (≥ ~130 items / ~0.3% share). All eight fit; written into the lattice section of `docs/PESO-REPLICATION.md` as a falsifiable account.

**Validation:** E1 paired against the same clean run for both targets; E1's Blue Note rival cells act as the isolation control (≤ 0.5 ranks). E2 computed from local metrics with per-pair cross-checks (within-run vs between-run differentials reconciled). Ruff/pytest not affected (no source changes); docs and plans committed.

**Limits:** Eight observations with correlated features (rank, share, items) — the two-condition account is suggestive, not causal; the cleanest next test is a buried-and-learnable push on Instruments or Video Games (prediction: it should now persist). E1 is one seed, one attack.

## 27 October 2026 — Cross-domain test closes the two-condition account

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** submit the ins cross-domain test of the persistence account (user's go) and report.

**Assistance:**
- Identified Instruments' only learnable tail candidate from the category stats: Drum & Percussion Accessories (149 items, about Verve's 131); Video Games has none (largest tail category, 36 items). Added the 2-run test to `focus_plans.py` (`rare4`: clean tracks Drum & Perc Acc + Keyboards & MIDI as reference) and submitted job 1024984.
- Result: the candidate is anchored (clean rank 12.3 of 30) and the account holds negatively — the push does not land (+1.30 in the attacked period, the only failure to promote in ten pushed-category runs) and sits at clean by block 4 (−0.01). Side-observation: the unpushed popular category (Keyboards & MIDI, source of the bandwagon filler) sits −0.42 to −0.80 against clean in every block — a missed attack still promotes popularity. Video Games' tail is below the landing floor throughout.
- Recorded in the lattice section of `docs/PESO-REPLICATION.md`: the two-condition account covers ten pushed-category observations without a miss, and its persistent quadrant is empty outside CDs.

**Validation:** Paired against the run's own clean tracking the same two categories; the filler side-effect was checked at all four blocks before being claimed. One SSH session expired during polling and was re-authenticated by the user; no cluster state beyond the two run outputs.

**Limits:** One seed, one attack; the persistent quadrant is empty outside CDs, so the account's cross-domain confirmation is necessarily negative (anchored candidates wash) rather than a positive replication. The filler side-effect is a single run and unreplicated.

## 27 October 2026 — Artifact updated with the persistence story

**Tool:** OpenCode (GLM 5.3 Flash test).

**Request:** fold the persistence findings into the layperson artifact.

**Assistance:**
- Added Finding 07 ("Some lies have something to hold onto") to `reports/corrupted-stream-grid.html`: prose plus a five-bar chart of block-4 residue for the pushed CDs categories (New Age +3.3, Verve +2.9, Blue Note +2.7, Reggae −1.5, Soundtracks washed), the two-condition account in plain words, the ins/vg boundary, and the trigger's group-hidden persistence.
- Added a fifth conclusion card ("A lie needs something to hold onto") with a grok-it box; retitled the conclusions to "Five things the 328 runs said"; updated run counts (303 → 328) and dates throughout; extended the caveats (discriminators are one seed each; the hold-onto account is tested in one catalogue and checked in the negative on the others).
- Rendered the PDF (10 A4 pages), updated the gist (revision b802807) and verified the htmlpreview render shows the new section.

**Validation:** Headless-Chrome screenshots of the new section and conclusions at desktop width; PDF page-checked (10 pages, no broken cards); the gist's raw HTML grep-confirms the new content and a preview screenshot matches the local render. Repo files committed.

**Limits:** The artifact is a communication piece; the persistence account's limits (one seed, one shop for the strong effect, correlated features) live in the caveats list rather than the body.
=======

## 29 October 2026 — Account strata and critical-mass dose sweep (issues #18, #19)

**Tool:** OpenCode (DeepSeek V4.1 Flash).

**Request:** implement the account-quality and critical-mass actionables from the late-October feedback (GitHub issues #18 and #19) on top of the landed attack catalogue.

**Assistance:**
- Added `experiment/user_strata.py`: training-only activity and reach strata, deterministic head/tail selection with user-id tie-breaks, even and capacity-proportional insertion budgets, the value-invariant account scaling, and a Spearman/overlap confound report.
- Extended `real_history` with selection and per-account allocation while keeping the legacy uniform path unchanged. Reports add the selection block and a same-stratum separability AUC.
- Added `adapter-blocks` flags `--user-stratum`, `--user-side`, `--user-share`, `--user-accounts`, `--user-budget`, `--value-k`, `--max-insertions` and `--eval-strata`, plus `push_activity_*` and `push_reach_*` cells.
- Added `robustness-plan --stage strata` and `--stage dose`; the dose plans cover 0.1% to 5% by default. Dumps now carry the top-10 set for churn.
- Added `experiment/dose.py` and `cp4285 dose-summary`: paired user joining, mean rank change with a paired bootstrap, top-1/5/10 shares, churn, normalised gain, and the pre-registered threshold verdict.
- Documented the design, first-pass run matrices, costs and open definitions in `CONTINUED-TRAINING.md` and `IMPLEMENTATION.md`.

**Validation:** 114 tests pass with the `llm` and `meta` extras; Ruff is clean. New tests cover stratum determinism, training-only features, budget arithmetic, the value invariant, selected real-history, per-half evaluation cells, planner parsing and the dose curve/threshold.

**Limits:** No run has been launched. The run matrices (22 and 37 runs, about 5.3 and 7.4 GPU-hours) need approval. The primary metric, k grid, matched-event interpretation and threshold constants remain proposals for team confirmation.

## 29 October 2026 — Catalogue provenance and tag

**Tool:** OpenCode (DeepSeek V4.1 Flash).

**Request:** close the provenance part of #16 once the user opened the cluster SSH session.

**Assistance:**
- Hashed the five catalogue sources in the NUS SoC checkout `/mnt/scratch/w/wyenzhe/peso/cp4285-main/src` and compared them with commit `d38d27c`; all five SHA-256 values matched.
- Created and pushed annotated tag `attack-catalogue-2026-10-26` on `d38d27c`.
- Added the provenance paragraph to `docs/CONTINUED-TRAINING.md`, the tag to `CONTEXT.md`, the code-revision line to `docs/IMPLEMENTATION.md`, and updated the reference's status notice.

**Validation:** the five cluster hashes equal the tagged files; `uv run pytest -q` (114 tests) and Ruff pass.

**Limits:** this is a source-hash match, not a VCS revision recorded inside the run artifacts. The 29 October account-strata and dose tooling (`e3a19a3`) is a later commit on top of the tag, so it is not part of the tagged catalogue.

## 29 October 2026 — Account strata and critical-mass dose sweep (issues #18, #19)

**Tool:** OpenCode (DeepSeek V4.1 Flash).

**Request:** implement the account-quality and critical-mass actionables from the late-October feedback (GitHub issues #18 and #19) on top of the landed attack catalogue.

**Assistance:**
- Added `experiment/user_strata.py`: training-only activity and reach strata, deterministic head/tail selection with user-id tie-breaks, even and capacity-proportional insertion budgets, the value-invariant account scaling, and a Spearman/overlap confound report.
- Extended `real_history` with selection and per-account allocation while keeping the legacy uniform path unchanged. Reports add the selection block and a same-stratum separability AUC.
- Added `adapter-blocks` flags `--user-stratum`, `--user-side`, `--user-share`, `--user-accounts`, `--user-budget`, `--value-k`, `--max-insertions` and `--eval-strata`, plus `push_activity_*` and `push_reach_*` cells.
- Added `robustness-plan --stage strata` and `--stage dose`; the dose plans cover 0.1% to 5% by default. Dumps now carry the top-10 set for churn.
- Added `experiment/dose.py` and `cp4285 dose-summary`: paired user joining, mean rank change with a paired bootstrap, top-1/5/10 shares, churn, normalised gain, and the pre-registered threshold verdict.
- Documented the design, first-pass run matrices, costs and open definitions in `CONTINUED-TRAINING.md` and `IMPLEMENTATION.md`.

**Validation:** 114 tests pass with the `llm` and `meta` extras; Ruff is clean. New tests cover stratum determinism, training-only features, budget arithmetic, the value invariant, selected real-history, per-half evaluation cells, planner parsing and the dose curve/threshold.

**Limits:** No run has been launched. The run matrices (22 and 37 runs, about 5.3 and 7.4 GPU-hours) need approval. The primary metric, k grid, matched-event interpretation and threshold constants remain proposals for team confirmation.

## 29 October 2026 — Catalogue provenance and tag

**Tool:** OpenCode (DeepSeek V4.1 Flash).

**Request:** close the provenance part of #16 once the user opened the cluster SSH session.

**Assistance:**
- Hashed the five catalogue sources in the NUS SoC checkout `/mnt/scratch/w/wyenzhe/peso/cp4285-main/src` and compared them with commit `d38d27c`; all five SHA-256 values matched.
- Created and pushed annotated tag `attack-catalogue-2026-10-26` on `d38d27c`.
- Added the provenance paragraph to `docs/CONTINUED-TRAINING.md`, the tag to `CONTEXT.md`, the code-revision line to `docs/IMPLEMENTATION.md`, and updated the reference's status notice.

**Validation:** the five cluster hashes equal the tagged files; `uv run pytest -q` (114 tests) and Ruff pass.

**Limits:** this is a source-hash match, not a VCS revision recorded inside the run artifacts. The 29 October account-strata and dose tooling (`e3a19a3`) is a later commit on top of the tag, so it is not part of the tagged catalogue.
>>>>>>> origin/main
