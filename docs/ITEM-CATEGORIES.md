# Item subcategories from Amazon 2023 metadata

22 October 2026. Agent-built tooling, proposed to the team. Not yet a team decision.

## Purpose

The group wants a denser interaction subset than a whole Amazon category. Item metadata has a hierarchical `categories` path, such as `Video Games > PC > Accessories`. Grouping items by a path prefix gives candidate subcategories. Joining those groups to the 5-core rating files shows which ones are dense enough to use.

Many items have no path. A text classifier assigns them a predicted subcategory, and every row records whether its label was observed or predicted.

## Coverage of `categories`

The full audit (`meta-audit`, run on Runpod on 21 October) covers 35,393,189 items in the 34 metadata files (33 categories plus `Unknown`). No `parent_asin` repeats within a file, and no item lacks the `categories` key. 3,892,249 items (11.0%) have an empty path and 7,778 have a path shorter than two levels.

- `All_Beauty`, `Amazon_Fashion`, `Digital_Music`, `Health_and_Personal_Care`, `Subscription_Boxes` and `Unknown` are 100% empty. `Software` is 79% empty and Movies_and_TV 43%.
- Most other files are 0–12% empty; Cell Phones is 19% and Magazine Subscriptions 25%.
- Roots usually match the file. Some files mix in other roots; Movies_and_TV, for example, has genre roots such as `Comedy`.
- Level-2 paths contain promotional nodes, for example `Video Games > Save up to 50% on Astro Gaming`. They form small groups that the summary exposes.

## Method

A subcategory is the first $L$ levels of the path, with $L = 2$ by default (root and one subcategory).

1. `meta-download` fetches `meta_<Category>.jsonl.gz` from the official UCSD mirror, verifies the gzip and keeps the grouping and text columns as parquet.
2. `meta-audit` counts missing, empty and shallow paths per file and writes the full category tree.
3. `meta-classify` represents item text (title, brand, features, description) as hashed TF-IDF word and word-pair counts. It fits a logistic-loss linear model (scikit-learn `SGDClassifier`) on items whose path is observed. Only items without a path deep enough are predicted.
   - Classes are level-$L$ paths with at least 200 items.
   - Training uses a uniform sample of up to 200,000 labelled items. A fifth of the sample is held out for metrics.
   - An item with a shallow path may only receive labels below that path.
   - A prediction is kept only if its confidence reaches a threshold. The threshold is the lowest one at which held-out precision reaches 0.8. Other items stay `unknown`.
   - Files that are empty learn from a donor file (`DONORS` in `src/cp4285/common/metadata.py`). For example, `All_Beauty` uses `Beauty_and_Personal_Care`. `Unknown` and `Subscription_Boxes` use every file.
4. `meta-group` writes one row per item and file, with `parent_asin, domain, subcategory, full_path, label_source, confidence`. It also writes `subcategory_summary.parquet` with item counts per label source. Where a 5-core rating file is present, the summary adds users, ratings and density $= \text{ratings} / (\text{users} \times \text{rated items})$.
5. `meta-publish` uploads only these derived tables to a private Hugging Face dataset. Raw metadata stays at the source.

A full-scale zero-shot NLI model would need one forward pass per item and label, over 35M items and hundreds of labels. An earlier version used `BAAI/bge-small-en-v1.5` sentence embeddings. On Video_Games the TF-IDF model was more accurate (0.859 against 0.770 held-out accuracy, macro-F1 0.620 against 0.334) and ran in 34 seconds on a laptop instead of 12 minutes, so the pipeline no longer needs a GPU.

## Commands

```sh
uv sync --locked --extra meta
uv run cp4285 meta-download --category all      # about 26 GB compressed; resumable
uv run cp4285 meta-audit
uv run cp4285 meta-classify --category all
uv run cp4285 download --domain all             # 5-core rating files; six small categories have none
uv run cp4285 meta-group
HF_TOKEN=... uv run cp4285 meta-publish --repo <user>/amazon23-item-categories
```

On a large machine, set `CP4285_DUCKDB_MEMORY` (default `2GB`) and `CP4285_DUCKDB_THREADS` (default `4`). Outputs go to `data/meta/out`. Each stage refuses to overwrite its outputs. With `--category all`, finished categories are skipped, so a rerun resumes.

## Full run (22 October)

A Runpod CPU pod (16 vCPU) ran every stage. The output is the private Hugging Face dataset [`randomwish/amazon23-item-categories`](https://huggingface.co/datasets/randomwish/amazon23-item-categories), which holds `item_categories/` (one parquet folder per file), `subcategory_summary.parquet`, `category_tree.parquet`, `audit.json` and per-file `metrics/`.

- 35,393,189 rows, one per item and file. 31,493,162 (89.0%) are observed, 2,935,207 (8.3%) predicted and 964,820 (2.7%) unknown.
- Of the 3,900,027 items without a deep enough path, 75% received a prediction above the threshold.
- Product files score 0.90–0.98 held-out accuracy. Files whose second level is a genre score lower: Books 0.588, CDs_and_Vinyl 0.422, Movies_and_TV 0.372. No Movies_and_TV threshold reached 0.8 precision, so its 322,638 unlabelled items stay `unknown`. Books labelled 104,316 of 528,673.
- Donor-trained files: Amazon_Fashion, All_Beauty and Health_and_Personal_Care labelled almost every item. Digital_Music (CDs_and_Vinyl donor) labelled 7,095 of 70,530, and Unknown (all files) 267,032 of 390,005.
- Among groups with at least 2,000 rated items and 20,000 users, the densest are `Video Games > Nintendo Switch` (density $1.1 \times 10^{-3}$), `Industrial & Scientific > Adhesives, Sealants & Lubricants` and `Musical Instruments > Microphones & Accessories`. Without a size filter, tiny promotional groups top the density ranking.

The density ranking is a starting point for choosing a domain. It has not been checked against the experiment's cutoff and sequence-length requirements.

## Local dry run (CPU, 22 October)

This was a software check on two small files. It is not the reported result.

| file | items | empty paths | classes | model | held-out accuracy | macro-F1 | predicted / pending |
|---|---:|---:|---:|---|---:|---:|---:|
| Magazine_Subscriptions | 3,391 | 834 | 4 | embedding | 0.760 | 0.725 | 714 / 834 |
| Magazine_Subscriptions | | | 4 | TF-IDF | 0.744 | 0.683 | 712 / 834 |
| Video_Games | 137,269 | 12,637 | 12 | embedding | 0.770 | 0.334 | 12,038 / 12,637 |
| Video_Games | | | 12 | TF-IDF | 0.859 | 0.620 | 12,637 / 12,637 |

Row counts reconciled: all 140,660 items appear once. Low macro-F1 comes from rare platform classes such as Virtual Reality.

A first full run with the embedding model on a Runpod RTX 4090 finished the download, the audit and 30 of 34 files before the pod stopped over a billing issue. Its host GPU was then taken, so the pod could not restart and its disk was unreachable.

## Limits

- Predicted labels are model output. Keep `label_source` in any experiment, and report results with and without predicted items.
- The 0.8 precision target is measured on observed items. Items with empty paths may differ from them, for example by having less text.
- Donor labels for empty files are an assumption. `All_Beauty` items get `Beauty & Personal Care` subcategories, which Amazon may not have intended.
- Amazon Reviews 2023 is for research use. The published table holds IDs and labels only.
