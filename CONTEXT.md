# Project context

Snapshot: **28 September 2026**. Start here when continuing work in Pi; then read only the task-relevant documents linked below. Historical notes retain their original dates.

## Question and ownership

Study original-task retention during continued training: train a recommender on A, continue with A-only or A/B mixtures, and repeatedly measure a fixed A task. Compare a classical baseline with a neural/advanced model under a common protocol. Hongyi's confirmed contribution is the **neural recommender**. Other allocations and final experimental choices need team confirmation.

This is an offline study, not a deployed service. Improvement or no measurable decline is a valid result. The [project reconstruction](docs/UNDERSTANDING.md) separates meeting evidence from interpretation; the [project overview](docs/PROJECT.md) explains scope.

## Implemented versus proposed

| Implemented | Still proposed / not measured |
| --- | --- |
| uv-managed Python 3.12 package and CLI | Final dataset and classical-model selection |
| Explicit official Amazon ID-file downloads with full gzip validation | Real Amazon audit/training results for the neural cutoff pilot |
| Original-ID overlap, duplicate and timestamp audits | Neural B embeddings/scoring and mixed-domain updates |
| Bounded familiar-item A preparation with isolated holdouts | Classical/neural comparison under one protocol |
| Original SASRec architecture ported to PyTorch, validation-selected checkpoint | Original TensorFlow runtime / paper benchmark reproduction |
| Checkpoint reload, frozen control, fresh A-only continuation | B adaptation evaluation, multi-seed results and mitigations |
| Synthetic lifecycle tests and six-slide HTML narrative | Final team deck / submission |
| Classical comparator on real Amazon A/B: Markov, QR-factorised Markov, SLIST, contamination sweeps ([CLASSICAL.md](docs/CLASSICAL.md)) | Classical results under the pilot's cutoff protocol; the multi-seed classical study (`cp4285 classical study`, validation-tuned, paired CIs) is implemented and partly run, see section 19 of `reference.html` |

**Model decision, 28 September:** Hongyi requested the original [kang205/SASRec](https://github.com/kang205/SASRec) architecture, replacing the generic Transformer. See [SASREC.md](docs/SASREC.md) for the pinned source, faithful block structure, original sequence-wise initial loss and explicit experiment differences. The old model's checkpoints are incompatible.

**Shared utilities, 28 September:** `src/cp4285/common/utils.py` owns create-only JSON publication, file SHA-256 and UTC date conversion. Both workstreams import these helpers directly.

**Package layout, 28 September:** the neural workstream lives in `src/cp4285/neural/` (`data.py` cutoff split, `model.py`, `pilot.py`). Protocol-independent download, validated loading and the A/B audit moved to `src/cp4285/common/data.py`. Each workstream keeps its own split and evaluation; the classical loader is unchanged. Import paths changed; CLI, config and behavior did not.

The [runbook](docs/IMPLEMENTATION.md) describes actual code. The [continued-training specification](docs/CONTINUED-TRAINING.md) describes the broader proposed experiment. Do not confuse the two. The classical workstream reports exploratory real-data results under leave-last-out splitting; the neural cutoff pilot still has only synthetic checks. Those scores are not yet comparable.

## Provisional defaults and invariants

- Amazon Reviews 2023 Electronics (A), Movies & TV (B); official deduplicated 5-core ID files.
- Target is the **next reviewed item**, all ratings by default; review times are not purchase/viewing times and low ratings are not endorsements.
- Config: 2021-01-01 initial cutoff, 2022-01-01 stream end, at most 500 reviewers, history 50, hidden size 50, two blocks, one head, dropout 0.5. These are exploratory, not tuned settings.
- Omit ambiguous timestamp ties. Build vocabulary from initial training only; count omitted unknown items. Exclude validation and retention events from all training targets **and prefixes**.
- Freeze A histories, targets and full initial catalogue across evaluation checkpoints. Rank with stable ties; tune on validation, never retention curves.
- Left-pad to fixed width, keeping the latest item at position 49 by default. Initial training supervises all nonpadding positions in each sampled user's final training window; continuation supervises only the fresh final target to avoid implicit replay.
- Preserve weights across continuation; reset Adam (betas 0.9/0.98) once at the phase boundary, then preserve its state. Existing run directories must not be overwritten. Audit/prepared/manifest/metric/classical-report JSON is create-only; use new output paths for repeat runs.
- Planned mixtures use B's share of supervised examples. The draft group deck instead uses B/A volume: resolve the denominator. Fixed total updates also reduce A exposure as B increases.
- Separate catalogues require shared parameters before B can affect A. Do not equate unrelated integer IDs or fabricate cross-user histories. Disjoint matrix-factorisation factors can be an isolation control.

## Next actions

**Checkpoint reached:** original SASRec is merged in [PR #2](https://github.com/Hong-yiii/cp4285_reccomender_systems/pull/2), preserving the teammate's classical comparator. The subsequent artifact-safety checkpoint adds full gzip verification and non-overwriting JSON publication. It does not approve acquisition or change the experiment split. See the [runbook](docs/IMPLEMENTATION.md#artifact-safety-checkpoint) for retry/recovery limits.

1. Confirm the **29 September design critique** submission time in Canvas; the template's weekday/date conflict is unresolved. Review [requirements](docs/REQUIREMENTS.md) and [deck preparation](docs/DEFENSE.md).
2. Agree task, category pair, feedback filter, mixture denominator and classical-model sharing with the team. Explicitly resolve the neural validation/retention shared-prefix interpretation and align classical/neural protocols before comparing scores.
3. Review storage/compute before explicitly downloading full categories; run the audit and inspect feasibility/coverage before changing config defaults.
4. Run a real A-only pilot and validate its split, reload and frozen-control behavior.
5. Only then implement B interfaces, independent mixture branches and B evaluation; leave adapters/replay until a defensible baseline exists.

## Context and slides

- [Reviewer reference](reference.html): persistent, self-contained project walkthrough at the repository root. Original narrative and diagrams are preserved as dated evidence; read its latest checkpoint first. Append updates rather than silently rewriting historical results. Source and runbook remain authoritative.
- [Neural model rationale](docs/MODERN-RECOMMENDER.md), [experiment controls](docs/EXPERIMENT.md), [dataset assessment](docs/DATASET-ASSESSMENT.md).
- [Classical comparator](docs/CLASSICAL.md): the contributor reports single-seed validation sweeps showing disjoint-B effects under shared rank/item budgets and 10.3% reviewer overlap. These originating-workspace results were not rerun during the SASRec merge and are not a common-protocol neural comparison.
- [Slide narrative and Mermaid diagrams](docs/SLIDE-FLOW.md); [editable HTML slides](slides/index.html).
- [Primary references, course slides and local-only source index](docs/SOURCES.md).
- `local/course/` holds copied Week 01–02 PDFs, learning notes and references on the migrated machine. `local/sources/` holds supplied project evidence and the draft team PDF. Neither is published.
- [AI log](docs/AI_LOG.md) records prior Codex assistance and the move to Pi. Add subsequent substantive assistance for the team's disclosure.
