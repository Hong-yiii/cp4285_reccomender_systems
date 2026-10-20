# Project context

Snapshot: **19 October 2026**. Read this first, then [Packages and building rules](docs/PACKAGES-AND-BUILDING.md) and the [October meeting synthesis](docs/MEETING-2026-10-15.md).

## Current direction and ownership

Study old-task recommendation quality during continued learning, potentially varying model capacity and incoming domain composition. The team wants to explore independently and converge on a common, inspectable benchmark. A decline, no change or improvement is a valid outcome; do not promise catastrophic forgetting or neural superiority.

**User-confirmed reset:** treat previous implementation as disposable trial work. Preserve scientific understanding and the principles of testing and writing two extensible packages. The user explicitly permits deleting old code. The package/building document is a **draft for human review**, not an approved final protocol.

The two responsibility packages are:

- `cp4285.experiment`: source preparation/contracts, schedules, model ABC, shared evaluation/selection, artifacts and graphs.
- `cp4285.models`: concrete neural/non-neural implementations and model-side ablations.

The notebook imports both and explains the experiment. The package implements computation and safeguards; the notebook exposes assumptions and evidence; the eventual report owns the argument. Scientifically different tasks must not be forced through a universal runner.

## Implemented now

- Python 3.12 foundation with standard-library-only core. **User supersedes GPU-only requirement: CPU and CUDA are supported, CPU default.** Explicit `cpu`/`cuda`/`auto` selection; requested CUDA fails rather than falling back. Mutually exclusive optional `neural` / `neural-cuda` extras pin official Torch 2.14.1+cpu / +cu126 through explicit uv indexes on Linux/Windows, project-local only. Core/counting/standalone loader remain GPU-independent. [RTX handoff](docs/RTX2070-HANDOFF.md) retains official SM75/wheel/driver sources and pending CUDA hardware gates.
- `DataSource` and `Recommender` ABCs; typed event/query/example/prepared-experiment records.
- Fresh-target update blocks, constant/changing A/B schedules, no silent stream recycling.
- One concrete **single-target** runner: initial selection on validation NDCG, independent continuation branches, frozen control, fixed old/new task evaluation.
- A standalone shared evaluator in `experiment/evaluation.py`: NDCG/Hit/Recall with explicit cutoff, stable original-item-ID ties and fixed candidates.
- Create-only prepared snapshot, manifest and raw result JSON; SVG plots read saved results.
- `CountingRecommender` and `SyntheticSource` as a plumbing/isolation fixture, **not research models/data**.
- `notebooks/01_shared_experiment.ipynb`; offline contract checks include executing its cells in a fresh Python process.
- A sample-first development guide with complete maintained model/evaluator definitions, a nonempty-history/candidate-alignment/state check and notebook usage. `tests/test_documentation.py` verifies source/example agreement and executes all guide Python blocks. No new model or metric framework was added.

- `experiment/amazon.py`: bounded local official ID CSV/review JSONL loader (gzip supported), original IDs, training-only frozen catalogues, explicit ordinal/cohort/timestamp-tie rules and exclusion audit. No hidden acquisition in source/notebook. Parent separately acquired bounded official category prefixes with explicit authorization and provenance; data remain ignored.
- `models/sasrec.py`: CPU/CUDA SASRec-style shared causal encoder/positions with separate domain item tables, full-softmax final-target CE (not original reproduction), one Adam step/target in order (no batching), owned device dropout RNG, process RNG-preserving scoped initialization, independent full-state forks and reset-once Adam continuation. CPU uses non-capturable Adam, CUDA capturable device-local state. Create-only format-3 trusted weights-only checkpoints restore on the **same backend** (CPU/CUDA RNG algorithms differ), with CUDA ordinal remapping; old formats 1/2 rejected.
- `notebooks/02_neural_continual_learning.ipynb`: CPU default with explicit device selection and early tensor/backward preflight before data/artifacts; labelled schema fixture, bounded real audit/training, sidecar hash/size verification and acquisition provenance, progress/loss/completed timing, cumulative process peak RSS and optional CUDA allocations. Fixed retention/adaptation and distinct B development diagnostics; shared runner/evaluator remain owners of training/selection/metrics. [Shared research scratchpad](docs/RESEARCH-SCRATCHPAD.md) records decisions, evidence and independent review status.

The old `common/`, `neural/`, `classical/`, CLI, pilot config and trial-only tests have been removed. The create-only JSON writer's behavior was retained inside `experiment/artifacts.py`. No old model adapter or compatibility layer is maintained. Historical source is recoverable at [pre-reset commit 8b9edd3](https://github.com/Hong-yiii/cp4285_reccomender_systems/tree/8b9edd3).

**Not implemented:** a classical research model, automatic interrupted-run resume, large-data streaming/batched evaluation, multi-positive tasks, multi-seed aggregation or formal transfer metrics. In-memory full-state forks and one-query/target processing are deliberate small-pilot limits. The proposed loader's bounded file prefix, activity-filtered cohort and known incoming training inventory are not representative Amazon sampling or strict calendar-time deployment. An authorized bounded real CPU run now completed on this 16-vCPU/15-GiB VM: 16 MiB total official Electronics/Movies & TV prefixes acquired, 50k rows/category scanned. Original 32-user protocol failed (zero test targets); parent approved one pre-metric change to 256 lexical eligible users, all splits/catalogues/epochs/continuation budgets unchanged. A dev/test 15/8, B 2/4; candidates 1,216/1,230; 512 initial A targets. Initial CE fell 9.5242→6.9119, but A dev NDCG@1 and **all 36 A/B evaluation records remained zero**. Generalization, meaningful retention and forgetting are not established; no test-based tuning followed. Runner 1.528 s, peak process RSS ~342 MiB, reload/fork check passed. Private executed notebook `runs/neural-real-cpu-executed-20261019.ipynb`, run `runs/neural-real-1b6e83b494e1/`; initially headless sequential Python cells. A same-protocol actual Jupyter-kernel repeat also completed: private `runs/neural-real-cpu-jupyter-8dc5844cc7ca.ipynb`, run `runs/neural-real-e97f8b3da603/`, identical inputs/results, five executed cells/two SVG outputs, 1.160 s runner and ~366 MiB peak process RSS. Browser UI remains untested. Independent final and follow-up audits found no engineering blockers and accepted CPU pipeline feasibility, not recommendation learning. CPU lifecycle and fixture notebook tests execute; **four CUDA cases explicitly skip without GPU and do not validate CUDA**. Target RTX forward/backward, state/reload and notebook acceptance remain unvalidated.

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
5. Review the proposed notebook/loader protocol and establish actual A/B development learning before model comparisons. The user explicitly authorized bounded acquisition and this real CPU pilot, superseding CUDA-only execution. Pipeline completed but development learning did **not** emerge at this budget; agree a separate development-only learning criterion/budget before extending it. No automatic sweeps, further cohort changes or test-curve tuning. Independent actual-source/notebook/results and follow-up kernel/test-transcript audits found no findings; engineering/CPU feasibility accepted. Parent owns publication to draft PR #12. See the scratchpad for exact evidence and residual hardware/scientific limits.

No paper/model names or appointment deadlines should be inferred from the machine-generated meeting transcript. External scientific claims still require primary-source checks.

## Historical evidence and privacy

September design notes, source discussions, figures and `reference.html` preserve reasoning and dated findings. They do not describe the current runtime; [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) does. The Google Slides deck remains canonical and was not edited.

Before the reset, four pre-existing uncommitted files were copied byte-for-byte to ignored `local/trials-before-reset-2026-10-16/`, with checksums: the real-neural config, baseline code/test and result report. That report describes real A-only runs, but this session did not revalidate their datasets/checkpoints. Preserve this distinction from earlier notes claiming only synthetic evidence. Existing ignored datasets/runs/reports were not modified.

Private transcripts, chats, team/course PDFs, personal IDs and datasets never belong in public commits. [AI_LOG.md](docs/AI_LOG.md) records substantive assistance. The repo must run without `local/` or historical artifacts.
