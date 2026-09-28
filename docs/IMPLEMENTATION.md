# Running the recommender study

Initialized 28 September 2026. Python 3.12, uv, DuckDB, NumPy and PyTorch. Exact dependency versions are in the repository's `uv.lock`. This is a runnable dataset audit and bounded **A-only** lifecycle pilot. The shared-domain design in the diagrams remains a proposal; B interfaces and mixed updates are not implemented yet.

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

Or download one category with `--domain Electronics` or `--domain Movies_and_TV`. Existing files are preserved. A `.part` file marks an incomplete download; rerunning that domain restarts it. Successful downloads record source URL, size and SHA-256 in a manifest. The checksum records provenance, not independent verification against a publisher checksum.

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

### Split and evaluation contract

- Remove exact duplicate rows. Omit **all** events at tied timestamps within a user rather than inventing an order. Count the omissions.
- Require at least five initial-period events per eligible reviewer. Reserve the final two for validation and the fixed retention anchor; all earlier events form initial training examples.
- Exclude both held-out events from all training targets and prefixes, including later continuation prefixes. The retention target uses only the initial training prefix, without the validation event.
- Build the item vocabulary from initial training examples only. This first pilot studies familiar items; out-of-vocabulary held-out targets and later events are omitted and counted. Omitted unknown items do not enter histories. This reduces coverage and is not a new-item adaptation experiment.
- Left-pad every history to the configured fixed width. Initial training samples users uniformly with replacement and supervises all next-item positions in each user's final training window. Continuing examples remain chronological with only their fresh final target supervised; historical targets are masked to prevent unintended replay.
- Select the initial checkpoint using validation NDCG only. Evaluate retention after selection, reload the saved checkpoint and verify the same scores. Reset Adam once for continued training, then preserve its state across cycles.
- Evaluate the same retention examples against the full fixed initial item catalogue, excluding padding. Repeated items remain eligible, because this task predicts review events. Use a stable item-ID tie rule. At one relevant target, Recall@10 is Hit@10.
- Use one sampled negative per active target: outside the user's full initial training item set for initial training, or outside the supplied prefix and fresh target for continuation. It is an unobserved alternative, not a known dislike. Sampled binary loss averages equally over active target positions. Adam uses betas (0.9, 0.98), without blanket weight decay or gradient clipping.

`metrics.json` records source/prepared checksums, config, retained counts, omitted-event counts, software/device, validation history and fixed-A curves. Model/optimizer checkpoints record cycle and stream cursor. The CLI does not yet implement restarting an interrupted continuation run, even though state is saved. Do not tune the model on its retention curves.

### Model scope

`src/cp4285/model.py` now ports the **original kang205/SASRec architecture** to PyTorch, replacing the generic encoder. The source commit, architecture checklist, license, tests and deliberate protocol differences are documented in [SASREC.md](SASREC.md). Upstream uses Python 2 / TensorFlow 1.12; we are not running that legacy runtime or claiming its benchmark scores. The demo uses the full configured architecture, with only its data/update budget reduced.

Initial training uses upstream-style sequence-wise loss and user sampling. Our fixed-A evaluator and fresh-event continuation are separate experimental choices. Checkpoints/metrics record implementation identity and upstream commit; old generic-model checkpoints are incompatible and must not be resumed as SASRec.

The frozen control and A-only continued training are implemented. Before adding B, review the real-data audit, rating policy, time windows and classical-model sharing assumptions. Then add B embeddings/scoring, preserve real domain identities, and implement controlled mixture branches. Repeated seeds, B adaptation, the A-exposure-matched control, classical baseline and mitigations remain further work. The defaults are small feasibility settings, not a final experiment budget.

## Layout

| Path | Purpose |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Managed environment and console command |
| `src/cp4285/data.py` | Official download, ID audit and pilot partitions |
| `src/cp4285/model.py`, `pilot.py` | Neural baseline, training, checkpoints and metrics |
| `src/cp4285/cli.py` | `cp4285` command |
| `tests/test_pipeline.py` | Meaningful data/evaluator/lifecycle checks |
| `configs/pilot.toml` | Reviewable experiment defaults |
| `slides/index.html` | Six-slide narrative with data and implementation status |
| `data/`, `runs/`, `reports/` | Ignored generated inputs and outputs |

The project now lives in its own repository, separate from the course archive. See [MIGRATION.md](MIGRATION.md) for the layout change. No deployment or course submission is configured.
