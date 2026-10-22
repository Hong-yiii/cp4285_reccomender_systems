# Project context

Snapshot: **16 October 2026**. Read [Packages and building](docs/PACKAGES-AND-BUILDING.md), the [October meeting synthesis](docs/MEETING-2026-10-15.md) and the [runbook](docs/IMPLEMENTATION.md) before extending the code.

## Current direction and ownership

Study recommendation quality on previously learned tasks during continued updates, potentially varying model capacity and incoming domain composition. The team wants independent model exploration converging on shared, inspectable evaluation. Decline, stability and improvement are all valid outcomes. Do not promise forgetting or neural superiority.

**User-requested integration:** apply the two-package/ABC paradigm to the actual implementations on `main`, preserve them, and open a PR into `main`. Keep the earlier lightweight reset separately as a reference: tag **`boilerplate-v1`**, commit `c169032`, branch `feat/boiler-plate`. Do not merge the reset's deletion of the research implementations. The integration branch is `feat/shared-package-integration`, based on `main` commit `8b9edd3`. The separate open classical-study PR #9 is not incorporated.

One distribution contains two responsibility packages:

- `cp4285.experiment`: data acquisition/preparation, source identities, learner ABC, schedules, evaluation/selection, artifacts and reporting.
- `cp4285.models`: model mathematics, update rules, optimizer/RNG state and model-side ablations.

The CLI remains `cp4285`; a thin notebook imports the packages. Old `common/`, `neural/` and `classical/` import paths are removed, not shimmed. This is a development contract, not team approval of a final research protocol.

## Implemented now

- Existing uv-managed Python 3.12 runtime and locked dependencies; no new framework or dependency.
- Shared generic `Recommender` ABC: `train`, `begin_continuation`, `score`, `describe`, and default full-state in-memory `fork`.
- `SASRecLearner` wraps the unchanged original-architecture PyTorch network. The actual pilot uses the ABC for updates and scoring. Initial training retains uniform user resampling and all-position sampled loss; continuation retains fresh-target-only loss, with Adam reset once and then preserved.
- Owned Python and CPU/device Torch RNG state, pure scoring, sequential fork isolation and copied optimizer parameter bindings. Tests exercise CPU; accelerator backends require their own validation.
- `IncrementalQRSVD` implements the ABC; the real classical stream calls it for fresh user-row chunks and one phase transition. It retains its basis, singular values and owned NumPy RNG.
- All batch classical kernels and their independent refit sweeps remain available. `MarkovQRSVD` is **not** incremental and still retains the full sparse transition table. No fake continuation adapter is provided.
- Both protocol-specific evaluators live in `experiment/`. **They remain scientifically different**, including splits, vocabularies, seen masking, ties and learning units. A shared ABC does not make scores comparable.
- Existing validated downloader/audit, holdout isolation, validation-selected SASRec weights, checkpoint reload, frozen control and create-only artifacts are retained.
- Runnable development guide and `notebooks/01_shared_lifecycle.ipynb`, with executable documentation and model-lifecycle checks in addition to the existing tests.
- Item subcategories, 22 October ([ITEM-CATEGORIES.md](docs/ITEM-CATEGORIES.md)): `cp4285 meta-*` commands group Amazon 2023 items by the first two levels of their metadata `categories` path and predict a subcategory for items without one with a TF-IDF classifier. Each row records whether its label is observed or predicted. The full run labelled 35.4M items (89.0% observed, 8.3% predicted, 2.7% unknown) and is published as the private Hugging Face dataset `randomwish/amazon23-item-categories`. The optional `meta` extra adds scikit-learn and huggingface-hub. No subcategory has been chosen as an experiment domain.

The tagged boilerplate's single-target `Example` contract and generic runner are **not** copied into this integration: enforcing them unchanged would silently alter main's all-position SASRec training and classical row-chunk semantics. Neither is presented as the final common benchmark.

**Not implemented:** neural B interfaces/mixed updates, common classical/neural evaluation, multi-seed aggregation, interrupted-run resume or a finalized experiment matrix. Checkpoints save weights/optimizer/cursor and owned RNG, but the CLI cannot resume them. Sequential in-memory forks are not thread-safe global-RNG isolation. Owned RNG and avoiding a throwaway network initialization during reload change historical continuation random trajectories; use new run artifacts and record the code revision.

## Scientific invariants and unresolved decisions

- Preserve source identities and domain namespaces; never equate unrelated integer IDs or splice unrelated users' histories.
- Keep holdouts out of training targets and prefixes. The inherited neural cutoff pilot still uses a shared initial prefix for validation and retention, so its retention anchor is not automatically ordinary one-step next-event testing.
- Freeze each task's queries, targets, candidate catalogue and ranking policy across checkpoints. Select/tune on validation, not retention curves.
- Declare update/exposure units: a SASRec supervised target is not a complete user row for incremental SVD. Batch refits are not continual training.
- The provisional Amazon target is a reviewed item, not a purchase timestamp or ideal-recommendation ground truth. Review the feedback/rating filter.
- A larger B share at fixed total targets also reduces A exposure; that alone does not isolate harmful B gradients. Agree the mixture denominator, controls and shared-parameter mechanism.
- Historical SASRec, QR-Markov, category pair, capacities, cutoffs, @10 and budgets are implementation defaults/proposals, not newly confirmed October choices.
- Choose a common task/horizon, split/cohort, vocabulary/cold-start policy, candidates, metric/cutoff, model/capacity, schedules, seeds and affordable budgets before comparing models.
- Acquisition and expensive runs require explicit approval. Synthetic software checks are never Amazon results. Existing artifacts must not be overwritten.

## Evidence, provenance and handoff

September source/design notes, slide figures and dated `reference.html` sections remain historical evidence. The Google Slides deck is canonical and was not edited. The architecture is the original pinned SASRec PyTorch port ([SASREC.md](docs/SASREC.md)); neither TensorFlow runtime parity nor paper benchmark reproduction is claimed. Contributor-reported classical exploratory Amazon results remain in [CLASSICAL.md](docs/CLASSICAL.md), under their own leave-last-out protocol.

Before the reference reset, four pre-existing uncommitted files were preserved byte-for-byte with checksums in ignored `local/trials-before-reset-2026-10-16/`: real-neural config, baseline code/test and a report of real A-only runs. Their underlying data/checkpoints were not revalidated here. Older statements that all neural evidence is synthetic refer to the earlier main checkpoint, not everything now present locally. This integration adds no real-data result and does not import those private trials.

Private transcripts, chats, team/course PDFs, personal IDs, datasets, checkpoints and agent sessions remain ignored. The public checkout must run without `local/` or historical artifacts. The meeting synthesis contains interpreted public notes, not the raw transcript. [AI_LOG.md](docs/AI_LOG.md) records substantive assistance.

Next: review the PR and learner contracts, settle the common research protocol, and independently implement/test model adapters against that agreed protocol. Do not silently merge PR #9, treat old deadlines as current, download datasets, edit the team deck or submit coursework.
