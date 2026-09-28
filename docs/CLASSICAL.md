# Classical comparator

Status: implemented, with **contributor-reported real Amazon runs from the originating workspace** (28 September 2026). Those runs were not rerun during the SASRec integration; raw logs are not included here. The results are **single-seed, validation-split exploratory results**, not tuned final numbers. The protocol differs from the SASRec pilot (see [Protocol](#protocol)), so the two sets of numbers are not directly comparable yet.

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
- Every rating counts as an event. Leave-last-out per user (the official Amazon'23 rule): last event is test, second-to-last validation, the rest training. This is **not** the pilot's cutoff/cohort protocol.
- Designs (`contaminate.py`):
  - `disjoint`: whole B timelines become new users.
  - `shared`: the 168,901 A users (10.3%) who also appear in B get their B events from before their validation event merged into their timeline. This is capped at about 15% of A volume.
- Evaluation: full ranking over all 368,228 A items (or A + B with `--full-catalog`), excluding the user's training items. Metrics are NDCG@10/20, Recall@10/20 and MRR on 20K sampled validation users. `base_share` reports how much of each model's capacity sits on A.
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

- Repeat with 3 seeds and the test split. Tune k, β, N and α on validation only.
- Align with the pilot protocol (cutoff cohort, familiar-item vocabulary, retention anchor) so classical and neural curves share one evaluation.
- Continued-training variant for the transition models (streamed `MarkovQRSVD`), rather than refitting.
- SLIST on Diginetica, the setting it was designed for, which needs a manual dataset download.
