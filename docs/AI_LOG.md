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
