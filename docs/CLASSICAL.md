# Classical comparator

Status: implemented, with **contributor-reported real Amazon runs from the originating workspace** (28 September 2026). Those runs were not rerun during the SASRec integration; raw logs are not included here. The results are **single-seed, validation-split exploratory results**, not tuned final numbers. The protocol differs from the SASRec pilot (see [Protocol](#protocol)), so the two sets of numbers are not directly comparable yet.

## Rigorous study (in progress, 28 September 2026)

`uv run cp4285 classical study` supersedes the single-seed sweeps below for reporting. SLIST is out of the study scope at the team's request (its code stays in `models.py`).

**Team decisions, 28 September ([issue #8](https://github.com/Hong-yiii/cp4285_reccomender_systems/issues/8)):** two tasks, each with one reported metric. The same metric chooses the tuned model and gets the confidence intervals; NDCG@10 only breaks ties. B levels stop at 100%; the 140% level is dropped.

| Task | Command | Hidden per user | Metric |
| --- | --- | --- | --- |
| Next item | `study` (`--protocol last`) | The last review | **Hit@10**: 1 if it is in the top 10 |
| Next items | `study --protocol time` | Up to 10 new products reviewed in the test window | **Recall@10**: hits ÷ min(10, number hidden) |

- **Model selection on validation only:** QR rank k ∈ {32, 64, 128, 256} × popularity weight β ∈ {0, 0.25}, clean data. The earlier NDCG@10 tuning chose k = 256, β = 0 (validation NDCG@10 0.0119, 0.0130, 0.0141, 0.0150 for the four ranks), at the edge of the grid. Tuning is being rerun on Hit@10.
- **Test split:** models are refitted on train + validation events and scored over all 368,228 A items.
- **Paired design:** the same 50K sampled test users at every seed and level.
- **Three seeds:** each changes the sampled B reviewers and the QR random start.
- **Uncertainty:** paired bootstrap 95% CI (1,000 user resamples) and a one-sided p-value for each change against clean (`evaluate.paired_change`), on the task's metric.
- **Mechanism control:** a QR model with rank k × (1 + level). If fixed capacity drives the drop, it should degrade much less.
- **Artifacts:** create-only JSON and NPZ; per-(seed, level) checkpoints under `reports/classical/study_ckpt_*` let a stopped run resume. Checkpoints hold ranks, so any metric can be recomputed from them.

Next item, observed in this workspace, seed 0 only (k = 256, β = 0 from NDCG tuning; test split):

| Hit@10 | Clean | B = 10% | B = 50% | B = 100% |
| --- | --- | --- | --- | --- |
| Markov chain | 0.0314 | 0.0314 | 0.0314 | 0.0314 |
| QR, k = 256 | 0.0215 | 0.0213 | 0.0207 | 0.0196 |
| QR, k grows with data | 0.0215 | 0.0216 | 0.0210 | 0.0209 |
| Popularity | 0.0133 | 0.0133 | 0.0133 | 0.0133 |

The fixed-rank model loses 8.9% of Hit@10 at 100% B; the growing-rank control loses 3.1%, consistent with the capacity explanation. Seeds 1–2 and Hit@10 intervals are still to be reported. For scale, published full-ranking SASRec scores on 25K-item Amazon'23 categories are 0.020–0.043 NDCG@10 ([ETEGRec, SIGIR 2025](https://arxiv.org/abs/2409.05546)); A here has 368K items.

## Next-items task (`--protocol time`)

`time_split` in `data.py` uses one global cutoff T and window W instead of each user's own last events:

- **Validation:** train on events before T − W; targets are each user's first 10 new products reviewed in [T − W, T).
- **Test:** train on events before T; targets are the first 10 new products in [T, T + W). Events from T + W on are never used, and B is also cut at the end of training.
- **Users:** everyone with at least one earlier event is scored, however few events they have, so the test is not limited to heavy reviewers. "New" means not reviewed before the window; in these files no user reviews a product twice, so no target is lost.

Defaults are T = 2022-01-01 and W = 365 days (validation targets from 2021, test targets from 2022). The validation training cutoff, 2021-01-01, is the neural pilot's `initial_cutoff`. Candidates measured on the Electronics 5-core file (1.64M users; data end 12 September 2023):

| T | W | Users scored | Mean targets | Users with ≥ 2 targets |
| --- | --- | --- | --- | --- |
| 2021-01-01 | 180 days | 425K | 1.76 | 38.7% |
| 2022-01-01 | 180 days | 338K | 1.69 | 35.5% |
| **2022-01-01** | **365 days** | **557K** | **2.13** | **48.2%** |
| 2022-09-01 | 365 days | 391K | 2.08 | 43.2% |
| 2023-01-01 | 180 days | 222K | 1.90 | 36.7% |

The chosen window scores the most users, and half of them have two or more targets, so Recall@10 carries information Hit@10 does not. Leave-last-out `--targets m` is not an alternative beyond m = 2: only 54.7% of users have the 7 events m = 3 needs, and 5.9% have the 21 events m = 10 needs.

Caveats:

- Order inside the window does not count. Markov predicts one step ahead, so it should look weaker relative to QR here than on the next-item task; that comes from the task.
- The official 5-core filter counts each user's and product's reviews over the whole history, including after T. That leaks a little future information into which users and products exist, for both workstreams.
- Classical models are refitted at each level, so no hidden event is trained on. A continual-learning (neural) version needs its own held-out user group whose post-T reviews never enter the training stream.

## Question

When interactions from B (Movies & TV) are added to training at increasing volume, how much does next-event ranking on A (Electronics) degrade for classical recommenders? Which model property decides that?

## Models (`src/cp4285/classical/models.py`)

| Model | What it does | Path by which B can affect A |
| --- | --- | --- |
| `MostPop` | Global popularity | Only when B items are also ranked (`--full-catalog`) |
| `Markov` | Raw last-event → next-event transition counts | Only through users with events in both domains |
| `MarkovQRSVD` | The transition matrix compressed to rank k with a randomized QR range finder (Halko et al. 2011). This is the factorised-transition idea of FPMC (Rendle et al. 2010) | **Shared rank budget.** The top-k components are chosen over A and B together |
| `SLIST` | Closed-form linear item-item model ([Choi et al., WWW 2021](https://arxiv.org/abs/2103.16104)), ported from the [authors' code](https://github.com/jin530/SLIST): co-occurrence (SLIS) and transition (SLIT) objectives, one Cholesky solve | **Shared item budget.** The dense N×N matrix covers only the N most popular items in the mixed data |
| `QRSVD`, `IncrementalQRSVD` | User×item PureSVD (Cremonesi et al. 2010), batch or streamed with QR residual updates in the style of Brand (2006) | Shared rank budget |

The `EXPERIMENT.md` note that isolated MF factors are an isolation control applies here directly. With disjoint B users and items, the training matrices are block-diagonal. `Markov` is then provably unaffected, and SLIST's item-item solve separates too (tested in `tests/test_classical.py`). Degradation needs a **shared, fixed capacity**: rank k for QR-SVD, or the item budget N for SLIST.

## Protocol

- Input: official Amazon'23 5-core ID files, the same files `cp4285 download` fetches. `--b data/raw/0core/Movies_and_TV.csv.gz` uses the unfiltered B file (17.3M events), because the 5-core B file only supplies 61% of A's training volume.
- Every rating counts as an event. `--protocol last` (default): leave-last-out per user (the official Amazon'23 rule): last event is test, second-to-last validation, the rest training. `--protocol time`: the next-items split above. Neither is the neural pilot's cutoff/cohort protocol yet.
- `--targets m` (default 1) holds out each user's last m events as test targets and the m before them as validation targets; a repeated item counts once. m = 1 reproduces the rule above exactly. Users with fewer than 2m + 1 events are kept but lose training events first, then validation targets, so a large m leaves many users ranked with little or no history; a time-cutoff split suits the next-items task better (issue #8).
- Test scoring (`study`, and `sweep`/`stream --split test`) refits on train + validation events, so the first test target is one step ahead of the history, as in the original SASRec evaluation. Before 28 September 2026 `sweep --split test` skipped the validation event and so predicted two steps ahead; reported sweeps used the validation split and are unaffected.
- Designs (`contaminate.py`):
  - `disjoint`: whole B timelines become new users.
  - `shared`: the 168,901 A users (10.3%) who also appear in B get their B events from before their validation event merged into their timeline. This is capped at about 15% of A volume.
- Evaluation: full ranking over all 368,228 A items (or A + B with `--full-catalog`), excluding the user's training items. Metrics are NDCG@10/20, Hit@10/20, Recall@10/20 and MRR on 20K sampled validation users. Hit@K is 1 if any of a user's targets reaches the top K; Recall@K is the number that do divided by min(m, K); NDCG@K divides by the ideal DCG of min(m, K) hits; MRR uses the best-ranked target. With m = 1, Hit@K equals Recall@K (older reports label it Recall). `base_share` reports how much of each model's capacity sits on A.
- The models are refitted at each level. `classical stream` streams users instead, but so far only for PureSVD.

## Results: disjoint B (0-core Movies & TV), k=64, SLIST N=20K

Contributor-reported NDCG@10 on A validation users, one originating-workspace run (seed 0, before the port's default became 4285); raw log remains in that workspace, not independently verified by this integration. The level is B events as a share of A training events.

| B injected | MostPop | Markov | Markov QR-SVD (A share of rank) | SLIST (A share of items) |
| --- | --- | --- | --- | --- |
| 0% | 0.0092 | 0.0230 | 0.0138 (1.00) | 0.0212 (1.00) |
| 10% | 0.0092 | 0.0230 | 0.0138 (0.93) | 0.0211 (0.96) |
| 50% | 0.0092 | 0.0230 | 0.0124 (0.44) | 0.0202 (0.70) |
| 100% | 0.0092 | 0.0230 | 0.0122 (0.25) | 0.0194 (0.49) |
| 140% | 0.0092 | 0.0230 | 0.0112 (0.16) | 0.0184 (0.39) |

Findings to verify with more seeds:

- The unconstrained local model (Markov) does not change. Both fixed-capacity models degrade steadily: QR-SVD by −19% and SLIST by −13% at 140%.
- SLIST at N=30K reaches 0.0220 on clean A, and at 50K it would likely pass Markov, but that needs about 20 GB (SoC cluster). Paper settings for Diginetica (α=0.8, 256-day age decay) score 0.0159 on Amazon. α=0.5 without age decay scores 0.0212.
- User×item PureSVD is below popularity on this data (0.0053 at k=64), so the next-event structure matters more than long-term taste.
- `shared` design at its 15% cap: Markov 0.0230 → 0.0227, Markov QR-SVD 0.0138 → 0.0136.
- `--full-catalog` at 61%: MostPop falls from 0.0092 to 0.0078 because popular B items enter the top-10.

## Running the study

Proposed in [issue #8](https://github.com/Hong-yiii/cp4285_reccomender_systems/issues/8); not yet run with Hit@10 tuning or the time protocol. Downloading the data is an explicit decision (about 720 MB); none of these commands runs as part of setup or tests.

```sh
uv sync --locked
uv run cp4285 download --domain both          # 5-core Electronics (A) and Movies & TV
mkdir -p data/raw/0core                       # 0-core Movies & TV (B, 17.3M events)
curl -L -o data/raw/0core/Movies_and_TV.csv.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Movies_and_TV.csv.gz
B=data/raw/0core/Movies_and_TV.csv.gz

# 1. Quick check (about 1 minute each; smoke numbers, not results)
uv run cp4285 classical sweep --b $B --split test --users 100000 --eval-users 5000 --levels 0 0.5
uv run cp4285 classical sweep --b $B --split test --users 100000 --eval-users 5000 --levels 0 0.5 --protocol time

# 2. Next item: tune on validation Hit@10, then 3 seeds x B = 0, 10, 50, 100%
uv run cp4285 classical study --b $B

# 3. Next items: targets in 2022, validation in 2021, Recall@10
uv run cp4285 classical study --b $B --protocol time
#    other windows: --cutoff 2022-01-01 --window-days 365 --max-targets 10
```

- **Time:** tuning fits 8 QR models; each (seed, level) then takes about 12–15 minutes on a 16-core laptop with 27 GB of RAM, so one study is roughly 3 hours. 140% ran out of memory on that machine, which is one reason it is dropped.
- **Stopping and resuming:** each finished (seed, level) is saved under `reports/classical/study_ckpt_k{k}_b{beta}_n{users}[_time...]/`. Rerun the same command, or add `--k K --beta BETA` to skip tuning, and finished levels load from disk. The existing next-item checkpoints (k = 256, β = 0) are reused if Hit@10 tuning picks the same values.
- **Output:** `reports/classical/study_<timestamp>.json` holds the tuning grid, per-level Hit@10, Recall@10, NDCG@10 and base share, and each change against clean with its 95% CI and p-value on the task's metric (`primary_metric`). `study_<timestamp>_per_user.npz` holds the per-user scores behind the intervals. Nothing is overwritten.
- **Reading it:** a model is affected by B if its change's CI excludes 0. The mechanism check is whether "QR, rank grows" degrades clearly less than "QR, rank 256".

## Commands

```sh
uv run cp4285 download --domain both                  # if not already present
uv run cp4285 classical stats                         # A/B user overlap
uv run cp4285 classical sweep --slist --b data/raw/0core/Movies_and_TV.csv.gz --levels 0 0.1 0.5 1.0 1.4
uv run cp4285 classical sweep --design shared --levels 0 0.05 0.1 0.15
uv run cp4285 classical sweep --full-catalog
uv run cp4285 classical stream --forget 0.9
# quick check: --users 50000 --eval-users 3000 --slist-items 5000
```

The first load caches integer arrays under `data/processed/` (about 40 seconds). A full sweep level takes 2–4 minutes on a 16-core laptop with 27 GB of RAM. Results go to timestamped JSON under `reports/classical/`, and existing files are never overwritten. The 0-core B file is not fetched by `cp4285 download`. Its URL is `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Movies_and_TV.csv.gz`.

## Open work

- Run both studies ([Running the study](#running-the-study)) once the team agrees the method in issue #8.
- Align with the pilot protocol (cutoff cohort, familiar-item vocabulary, retention anchor) so classical and neural curves share one evaluation.
- Continued-training variant for the transition models (streamed `MarkovQRSVD`), rather than refitting.
- SLIST on Diginetica, the setting it was designed for, which needs a manual dataset download.
