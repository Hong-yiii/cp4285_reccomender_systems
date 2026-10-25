# Running the recommender study

Initialized 28 September 2026. Python 3.12, uv, DuckDB, NumPy and PyTorch. Exact dependency versions are in the repository's `uv.lock`. This is a runnable dataset audit and bounded **A-only** lifecycle pilot. The shared-domain design in the diagrams remains a proposal; B interfaces and mixed updates are not implemented yet.

**Package integration, 16 October 2026:** the existing CLI and protocols are retained under `experiment/` and `models/`. SASRec and incremental PureSVD use the shared learner ABC; batch classical sweeps remain refits. Read [Packages and building](PACKAGES-AND-BUILDING.md) for extension recipes, native batch types and the separate `boilerplate-v1` reference tag.

## Setup

From the repository root:

```sh
uv sync --locked
uv run cp4285 doctor
uv run pytest -q
uv run ruff check src tests
```

The environment is local to `.venv`. Do not use pip or modify a global Python environment. Apple MPS availability is reported by `doctor`; the pilot defaults to CPU for a reproducible first check. Change the TOML device to `mps` or `cuda` only when available and verify results on that backend.

## First run without downloading Amazon

```sh
uv run cp4285 demo --output runs/my-synthetic-check
```

This creates visibly synthetic CSVs, audits them, prepares the split, trains the configured original-architecture SASRec model on a small synthetic cohort, reloads its checkpoint, and performs two continued-training cycles. Inspect `pilot/metrics.json` under the chosen output directory. Its scope is marked synthetic. These numbers must never appear as Amazon results. Existing run directories are never overwritten; use a new output path for a repeat.

Earlier synthetic runs are retained locally under `runs/` on the migrated machine; they are not included in a clone. Their recorded absolute paths describe the original workspace and are historical provenance, not reusable configuration. Tests cover holdout isolation, original-ID overlap, timestamp units, padding, stable metric ties, frozen evaluation and preservation of learned state. These checks do not establish research validity or convergence on real data.

## Amazon workflow

The downloader retrieves the **full official deduplicated 5-core ID-only category files**, not review text, images or metadata. It does not silently sample or download data when running other commands. Full categories are substantial; ensure storage and compute suit the audit. Compressed downloads and generated artifacts stay Git-ignored.

```sh
uv run cp4285 download --domain both
uv run cp4285 audit
```

Or download one category with `--domain Electronics` or `--domain Movies_and_TV`. Existing data **or manifest** files (including symlinks) are refused before requesting the network. Each attempt uses its own temporary `.part` file, validates the expected CSV header and reads the gzip through its checksum/trailer in bounded chunks before publishing. HTTP failures, truncated/corrupt gzip files and invalid headers do not produce a final data file or manifest; temporary files are cleaned on ordinary failure. A killed process may leave an unused temporary file; retries use a new one rather than overwriting it.

Successful downloads record source URL, size and SHA-256 in a manifest. The checksum records provenance, not independent verification against a publisher checksum. Data and manifest are published separately: if writing the manifest fails after data publication, the validated data is preserved. Inspect that file/manifest situation before retrying; do not automatically delete or replace either artifact.

Default inputs are `data/raw/Electronics.csv.gz` and `data/raw/Movies_and_TV.csv.gz`. Existing local CSV or CSV.gz files can be used by changing the config paths. The required schema is `user_id,parent_asin,rating,timestamp`, with Unix milliseconds.

The audit writes `reports/dataset-audit.json`. Inspect:

- Raw user/item/event counts and original-ID intersections.
- Exact duplicate rows, repeated user-item observations and ambiguous timestamp groups.
- Sequence-length quantiles after rating filtering, exact deduplication and tie removal.
- Users with enough pre-cutoff history, and how many also have later events.

Audit counts are eligibility indicators. Preparation adds cohort and vocabulary constraints, so the final report must use its retained counts too. The upstream globally filtered 5-core population uses full-history activity; disclose this benchmark limitation. See [DATASET-ASSESSMENT.md](DATASET-ASSESSMENT.md) for the feasibility criteria and the stricter temporal alternative.

## Review config, prepare, then train

Edit `configs/pilot.toml` after the audit. Current values are exploratory defaults, not inferred optimal settings:

- Target: next reviewed item using every rating (`min_rating=1.0`). For a positive-rating-only sequence, set `min_rating=4.0` and rerun both audit and preparation.
- Initial cutoff: 1 January 2021 UTC; update stream ends 1 January 2022 UTC.
- At most 500 initial-period eligible reviewers, selected by a deterministic hash of the original ID and seed. Selection does not depend on later activity. Each selected history is retained before sequence truncation.
- Original SASRec defaults: maximum history length 50; 50 hidden units; one attention head; two blocks; dropout 0.5; embedding-only L2 0.
- Three initial feasibility epochs (not upstream's 201), followed by three cycles of five updates, with batch size 128. If insufficient fresh continuation examples exist, the pilot fails with an explicit count instead of recycling data.

```sh
uv run cp4285 prepare
uv run cp4285 pilot
```

A custom config can be supplied before the subcommand:

```sh
uv run cp4285 --config configs/pilot.toml audit
uv run cp4285 --config configs/pilot.toml pilot --output runs/electronics-pilot-02
```

Paths within the config resolve from the repository root. CLI output paths resolve normally from the current working directory, so run the documented commands at the root.

### Artifact safety checkpoint

Audit reports, prepared snapshots, download manifests, neural metrics and classical JSON reports use the same **create-only** writer, `save_json` in `src/cp4285/experiment/utils.py`. It serializes finite JSON before publication, writes a same-directory temporary file, then links the completed file into place without replacing an existing file or symlink—even if another writer wins the destination race. Failed publication cleans its temporary file. The filesystem must support hardlinks; unsupported filesystems fail rather than fall back to overwriting. Atomic visibility is not a guarantee of power-loss durability or a multi-file transaction.

For repeat audits, choose a new `audit --output reports/audit-02.json`. For a new split, change `data.prepared` to a new snapshot path in the config; `prepare` has no `--output` or `--overwrite` flag. Pilot runs still require a new run directory. Existing research artifacts are never silently replaced by these JSON writers. Classical caches and model-checkpoint restart semantics are unchanged.

Offline regressions in `tests/test_artifact_safety.py` cover corrupt/truncated downloads beyond the header buffer, HTTP/header failures, preservation of existing files/manifests, JSON serialization and destination races, and audit/prepare CLI errors. No network downloads are needed for those checks. The current validation/retention shared-prefix construction is unchanged; resolve that research-contract choice before a real neural pilot.

### Split and evaluation contract

- Remove exact duplicate rows. Omit **all** events at tied timestamps within a user rather than inventing an order. Count the omissions.
- Require at least five initial-period events per eligible reviewer. Reserve the final two for validation and the fixed retention anchor; all earlier events form initial training examples.
- Exclude both held-out events from all training targets and prefixes, including later continuation prefixes. The retention target uses only the initial training prefix, without the validation event.
- Build the item vocabulary from initial training examples only. This first pilot studies familiar items; out-of-vocabulary held-out targets and later events are omitted and counted. Omitted unknown items do not enter histories. This reduces coverage and is not a new-item adaptation experiment.
- Left-pad every history to the configured fixed width. Initial training samples users uniformly with replacement and supervises all next-item positions in each user's final training window. Continuing examples remain chronological with only their fresh final target supervised; historical targets are masked to prevent unintended replay.
- Select the initial checkpoint using validation Hit@k (`training.k`, default 10), with NDCG@k breaking ties. Evaluate retention after selection, reload the saved checkpoint and verify the same scores. Reset Adam once for continued training, then preserve its state across cycles.
- Evaluate the same retention examples against the full fixed initial item catalogue, excluding padding. Repeated items remain eligible, because this task predicts review events. Use a stable item-ID tie rule. Metrics come from `experiment/metrics.py`; at one relevant target, Recall@10 is Hit@10.
- Use one sampled negative per active target: outside the user's full initial training item set for initial training, or outside the supplied prefix and fresh target for continuation. It is an unobserved alternative, not a known dislike. Sampled binary loss averages equally over active target positions. Adam uses betas (0.9, 0.98), without blanket weight decay or gradient clipping.

`metrics.json` records source/prepared checksums, config, retained counts, omitted-event counts, software/device, validation history and fixed-A curves. Model/optimizer checkpoints record cycle and stream cursor. The CLI does not yet implement restarting an interrupted continuation run, even though state is saved. Do not tune the model on its retention curves.

### Model scope

`src/cp4285/models/sasrec.py` ports the **original kang205/SASRec architecture** to PyTorch, replacing the generic encoder. The source commit, architecture checklist, license, tests and deliberate protocol differences are documented in [SASREC.md](SASREC.md). Upstream uses Python 2 / TensorFlow 1.12; we are not running that legacy runtime or claiming its benchmark scores. The demo uses the full configured architecture, with only its data/update budget reduced.

Initial training uses upstream-style sequence-wise loss and user sampling. Our fixed-A evaluator and fresh-event continuation are separate experimental choices. Checkpoints/metrics record implementation identity and upstream commit; old generic-model checkpoints are incompatible and must not be resumed as SASRec.

The frozen control and A-only continued training are implemented. Before adding B, review the real-data audit, rating policy, time windows and classical-model sharing assumptions. Then add B embeddings/scoring, preserve real domain identities, and implement controlled mixture branches. The [classical comparator](CLASSICAL.md) is implemented under a different leave-last-out protocol. Repeated seeds, neural B adaptation, the A-exposure-matched control, a common-protocol classical/neural comparison and mitigations remain further work. The defaults are small feasibility settings, not a final experiment budget.

### PESO-block SASRec study

`cp4285 blocks` trains SASRec on PESO's chronological Instruments blocks, with or without a category channel (`--arm sasrec|category`). It is the experiment proposed at the end of [ADAPTERS.md](ADAPTERS.md), and it needs PESO's released block files plus the category index from the PESO branch's `category_index.py`. Both stay in ignored folders, because upstream PESO has no licence.

```sh
uv run cp4285 blocks --data <dir>/Instruments_cl_ori \
    --categories <dir>/Instruments_cl_cat/Instruments_cl_cat.index.cat2.json \
    --arm category --seed 42 --output runs/blocks-category-seed42
```

- **Data:** PESO's users, blocks and targets. In each block, users with at least three new events train on all but the last two, validate on the second-to-last and test on the last. A target with no history is skipped and counted. The catalogue is all PESO items from the start, as in PESO's semantic-ID index.
- **Training:** block 0 uses the pilot's initial regime. Blocks 1–4 continue on fresh targets after one Adam reset. Each block keeps its validation-best epoch (Hit@10, NDCG@10 tie-break), with patience 5. The history window is 20, as in PESO.
- **Category channel** (`models/category_sasrec.py`): a second SASRec encoder over the history's category sequence. Its output is added to the item encoder's, and alone it predicts the next category with a softmax. Items rank by $\log\sigma(s_v) + \log P(c_v)$. Training adds the category cross-entropy to SASRec's sampled loss. The design follows CaFe (Li et al., SIGIR 2022) but is not a port of its code.
- **Evaluation:** every saved state on every block's test users. It ranks the full catalogue with no seen-item masking, and lower item ID wins ties. Metrics come from `experiment/metrics.py`, overall and split by how often the target was a training target up to the scored state (unseen, 1–4, 5+), beside a popularity baseline.
- **Per-user dumps** (`--dumps`): each cell also writes one row per test user under `<output>/dumps/`: PESO item IDs, the exact target rank and the top 20 by score. They contain user IDs, so they stay in the ignored run folder. The PESO branch's `new_items.py` reads them.

### Category language model with adapters (corrupted-stream variant)

`cp4285 adapter-blocks` runs the proposed variant in [CONTINUED-TRAINING.md](CONTINUED-TRAINING.md#proposed-variant-corrupted-streams-and-adapter-isolation-24-october-2026): a frozen language model with LoRA adapters recommends the next category on any of PESO's released domains (Instruments, Video Games, CDs & Vinyl), and a block's update stream can carry shill profiles. The cluster grid runs from `replication/peso/slurm/grid_plans.py` and `adapter_grid.sbatch` in the PESO branch checkout. It needs the optional extra (`uv sync --locked --extra llm`, which adds `transformers` and `peft`) and a local model directory.

```sh
uv run --extra meta cp4285 block-categories --data <dir>/Video_Games_cl_0.6_0_5_3_0 \
    --output runs/categories/Video_Games_cl_0.6_0_5_3_0-cat2
uv run cp4285 adapter-blocks --stage init --data <dir>/Video_Games_cl_0.6_0_5_3_0 \
    --categories runs/categories/Video_Games_cl_0.6_0_5_3_0-cat2/Video_Games_cl_0.6_0_5_3_0.index.cat2.json \
    --names runs/categories/Video_Games_cl_0.6_0_5_3_0-cat2/category_stats.json \
    --model <Llama-3.2-1B dir> --push-share 0.013 --output runs/lm-vg-s42-init
uv run cp4285 adapter-blocks --stage branch ... --init runs/lm-vg-s42-init --push-share 0.013 \
    --condition isolated --attack clone --alpha 0.03 --rollback-after --output runs/lm-vg-s42-iso-clone3-drop
uv run cp4285 adapter-summary runs/lm-vg-s42-init/metrics.json runs/lm-vg-s42-{shared,iso}-*/metrics.json
```

- **Categories** (`experiment/block_categories.py`, `cp4285 block-categories`): level-N labels for any PESO-format domain by PR 14's rule, from the domain's own `item.json` paths; items without a path get PR 14's text classifier, whose 10% holdout accuracy is recorded. It writes the index, label sources and `category_stats.json` (names and per-block statistics) that `adapter-blocks` reads. Needs the `meta` extra.
- **Learner** (`models/category_lm.py`): the prompt lists the history's category names; every category name is scored by its exact log-probability in one packed forward pass per user (a 4D attention mask keeps candidates apart), so ranking covers the whole closed category set. LoRA adapters (rank 8, dropout 0) hold every trainable parameter; the base weights never change. `add_stream` freezes the active adapters and stacks a new one; `set_active` switches one off to measure rollback, and `drop` deletes it so later streams train without it.
- **Stages**: `init` trains Theta_0 on block 0, saves `theta0.pt` and scores state 0 on every block (the frozen condition). `branch` loads it, resets AdamW once, and updates on blocks 1–4 with one shared adapter or one new adapter per block.
- **Pushed category**: `--push` names it, or `--push-share` picks the category whose share of block 1–4 test targets is nearest the given value, so each domain's target follows one rule. It is measured in every stage, clean branches included, to give the reference rank.
- **Rollback**: isolated attacked runs score the final state without the poisoned adapter. `--rollback-after` instead deletes it right after its block, scores that state (`dropped<b>_block<b>`) and trains the later adapters without it.
- **Shills** (`experiment/shilling.py`): random, average, bandwagon, segment or clone push profiles for one target category, added to one block until they reach `--alpha` of its training examples. Their lengths follow real users'; their `inj-` IDs keep them out of every evaluation set. The report includes the Jensen–Shannon shift of the training-category mix and a held-out classifier AUC for telling shills from real users.
- **Outputs**: `metrics.json` with every scored cell (shared metrics over all categories, ties to the lower index) and the pushed category's mean rank and top-1, top-5 and top-10 rates among genuine users whose own target differs; isolated attacked runs add rollback cells. `--dumps` writes per-user ranks inside the run folder.

## Layout

| Path | Purpose |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Managed environment and console command |
| `src/cp4285/experiment/utils.py`, `data.py` | Create-only JSON, hashing, dates, official download, validated loading and audit |
| `src/cp4285/experiment/metadata.py` | Item-metadata download, category audit, subcategory classifier and grouping; see [ITEM-CATEGORIES.md](ITEM-CATEGORIES.md) |
| `src/cp4285/experiment/model.py` | Shared generic `Recommender` lifecycle ABC |
| `src/cp4285/experiment/neural_data.py`, `pilot.py`, `neural_evaluation.py` | Neural cutoff split, sampling/selection/checkpoints and fixed evaluation |
| `src/cp4285/models/sasrec.py`, `sasrec_learner.py` | Unchanged network architecture; stateful optimizer/update/scoring adapter |
| `src/cp4285/models/category_sasrec.py`, `experiment/blocks.py` | Category-channel SASRec and the PESO-block study (`cp4285 blocks`) |
| `src/cp4285/experiment/metrics.py` | Shared Hit@K, NDCG@K and MRR@K from target ranks |
| `src/cp4285/models/category_lm.py`, `experiment/adapter_blocks.py`, `experiment/shilling.py` | Category language model with LoRA adapters, the corrupted-stream study (`cp4285 adapter-blocks`) and shill injection |
| `src/cp4285/experiment/block_categories.py` | PR 14's category rule for any PESO-format domain (`cp4285 block-categories`) |
| `src/cp4285/cli.py` | `cp4285` command |
| `src/cp4285/models/classical.py` | Classical kernels; incremental QR-SVD implements the lifecycle ABC |
| `src/cp4285/experiment/classical*.py`, `contaminate.py` | Classical data, leave-last-out split, sweeps/stream and evaluation |
| `tests/test_pipeline.py` | Meaningful data/evaluator/lifecycle checks |
| `tests/test_classical.py` | Classical loader, split, contamination, isolation and evaluator checks |
| `configs/pilot.toml` | Reviewable experiment defaults |
| `slides/README.md` | Link to the team Google Slides deck and slide principles (no code) |
| `data/`, `runs/`, `reports/` | Ignored generated inputs and outputs |

Import shared helpers from `cp4285.experiment.utils` and dataset acquisition/audit from `cp4285.experiment.data`. Both workstreams' splits and evaluators belong to `experiment/`, but remain separate modules because their protocols differ. `experiment.utils` and the learner ABC import only the standard library; numerical libraries are loaded only by the modules that need them. Old `common`, `neural` and `classical` package imports are removed; there are no compatibility aliases.

SASRec's adapter isolates Python and CPU/device Torch RNG across sequential in-memory forks and preserves module modes during scoring. The optimizer resets once at continuation and is retained afterward. Checkpoints save the learner-owned RNG rather than unrelated process-global state; durable resume remains unimplemented. CPU lifecycle checks pass; CUDA/MPS need backend validation before use. Since reload no longer consumes random draws initializing a discarded network, exact continuation numbers can differ from historical runs even though the objective/sampling policies are preserved.

The project now lives in its own repository, separate from the course archive. See [MIGRATION.md](MIGRATION.md) for the layout change. No deployment or course submission is configured.
