# Project context

Snapshot: **16 October 2026**. Read this first, then [Packages and building rules](docs/PACKAGES-AND-BUILDING.md) and the [October meeting synthesis](docs/MEETING-2026-10-15.md).

## Current direction and ownership

Study old-task recommendation quality during continued learning, potentially varying model capacity and incoming domain composition. The team wants to explore independently and converge on a common, inspectable benchmark. A decline, no change or improvement is a valid outcome; do not promise catastrophic forgetting or neural superiority.

**User-confirmed reset:** treat previous implementation as disposable trial work. Preserve scientific understanding and the principles of testing and writing two extensible packages. The user explicitly permits deleting old code. The package/building document is a **draft for human review**, not an approved final protocol.

The two responsibility packages are:

- `cp4285.experiment`: source preparation/contracts, schedules, model ABC, shared evaluation/selection, artifacts and graphs.
- `cp4285.models`: concrete neural/non-neural implementations and model-side ablations.

The notebook imports both and explains the experiment. The package implements computation and safeguards; the notebook exposes assumptions and evidence; the eventual report owns the argument. Scientifically different tasks must not be forced through a universal runner.

## Implemented now

- Standard-library-only Python 3.12 boilerplate, in the existing uv-managed distribution.
- `DataSource` and `Recommender` ABCs; typed event/query/example/prepared-experiment records.
- Fresh-target update blocks, constant/changing A/B schedules, no silent stream recycling.
- One concrete **single-target** runner: initial selection on validation NDCG, independent continuation branches, frozen control, fixed old/new task evaluation.
- A standalone shared evaluator in `experiment/evaluation.py`: NDCG/Hit/Recall with explicit cutoff, stable original-item-ID ties and fixed candidates.
- Create-only prepared snapshot, manifest and raw result JSON; SVG plots read saved results.
- `CountingRecommender` and `SyntheticSource` as a plumbing/isolation fixture, **not research models/data**.
- `notebooks/01_shared_experiment.ipynb`; offline contract checks include executing its cells in a fresh Python process.
- A sample-first development guide with complete maintained model/evaluator definitions, a nonempty-history/candidate-alignment/state check and notebook usage. `tests/test_documentation.py` verifies source/example agreement and executes all guide Python blocks. No new model or metric framework was added.

The old `common/`, `neural/`, `classical/`, CLI, pilot config and trial-only tests have been removed. The create-only JSON writer's behavior was retained inside `experiment/artifacts.py`. No old model adapter or compatibility layer is maintained. Historical source is recoverable at [pre-reset commit 8b9edd3](https://github.com/Hong-yiii/cp4285_reccomender_systems/tree/8b9edd3).

**Not implemented:** a real-data loader, real neural/classical model under the new interface, durable model checkpoint/reload/resume, large-data streaming/batched evaluation, multi-positive tasks, multi-seed aggregation or formal transfer metrics. In-memory full-state forks and one-query scoring are deliberate small-pilot limits. A passing synthetic check is not a real-data result or scientific validation.

## Scientific invariants

- Preserve real source identities and domain namespaces; never equate unrelated integer IDs or splice users' histories.
- Split before learning preprocessing. Holdouts enter neither training targets nor prefixes. Test events must not leak into validation-based selection.
- Freeze each task's queries, labels, candidates and ranking policy across checkpoints.
- Select/tune on development data, not retention curves. Keep independent state and declared optimizer/RNG policies across branches.
- Count supervised targets, not padded positions, users or files. Larger B share at fixed total targets means less A exposure; this alone does not isolate harmful B gradients.
- Aggregate score changes are outcomes under a protocol, not proof that a particular weight or memory was erased.
- Source events determine terminology: Amazon reviews are not purchase timestamps or ideal-recommendation ground truth.
- Artifacts are create-only and private by default. Acquisition/expensive computation needs an explicit decision.

## Decisions before a real experiment

1. Review the runnable package/building recipes and single-target baseline interface.
2. Choose model family, meaningful capacity variation, training/state policy and parameter accounting. October did not reaffirm SASRec or QR-Markov as mandatory models.
3. Choose dataset/category pair, observed feedback/filter, split/cohort, history horizon and cold-start/candidate policy.
4. Specify original-task versus past-task matrix, metric/cutoff, constant/changing/sequential schedules, mixture denominator, budget and seeds.
5. Implement one real loader and one defensible model, verify its lifecycle, then compare additional implementations on the same prepared contract.

No paper/model names or appointment deadlines should be inferred from the machine-generated meeting transcript. External scientific claims still require primary-source checks.

## Historical evidence and privacy

September design notes, source discussions, figures and `reference.html` preserve reasoning and dated findings. They do not describe the current runtime; [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) does. The Google Slides deck remains canonical and was not edited.

Before the reset, four pre-existing uncommitted files were copied byte-for-byte to ignored `local/trials-before-reset-2026-10-16/`, with checksums: the real-neural config, baseline code/test and result report. That report describes real A-only runs, but this session did not revalidate their datasets/checkpoints. Preserve this distinction from earlier notes claiming only synthetic evidence. Existing ignored datasets/runs/reports were not modified.

Private transcripts, chats, team/course PDFs, personal IDs and datasets never belong in public commits. [AI_LOG.md](docs/AI_LOG.md) records substantive assistance. The repo must run without `local/` or historical artifacts.
