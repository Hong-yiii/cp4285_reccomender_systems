# Item subcategories from Amazon 2023 metadata

22 October 2026. Agent-built tooling, proposed to the team. Not yet a team decision.

## Purpose

The group wants a denser interaction subset than a whole Amazon category. Item metadata has a hierarchical `categories` path, such as `Video Games > PC > Accessories`. Grouping items by a path prefix gives candidate subcategories. Joining those groups to the 5-core rating files shows which ones are dense enough to use.

Many items have no path. A text classifier assigns them a predicted subcategory, and every row records whether its label was observed or predicted.

## Coverage of `categories`

A read of the first 3,000 items of each of the 34 metadata files (33 categories plus `Unknown`) gives this picture. The full audit replaces it after the cloud run.

- Most files have 0–9% empty paths. Exceptions are Cell Phones (17%) and Magazine Subscriptions (23%).
- `All_Beauty`, `Amazon_Fashion`, `Digital_Music`, `Health_and_Personal_Care`, `Subscription_Boxes` and `Unknown` are 100% empty. `Software` is 85% empty.
- Roots usually match the file. Some files mix in other roots. Movies_and_TV, for example, has `Comedy` as a root on 10% of the sample.
- Level-2 paths contain promotional nodes, for example `Video Games > Save up to 50% on Astro Gaming`. They form small groups that the summary exposes.

## Method

A subcategory is the first $L$ levels of the path, with $L = 2$ by default (root and one subcategory).

1. `meta-download` fetches `meta_<Category>.jsonl.gz` from the official UCSD mirror, verifies the gzip and keeps the grouping and text columns as parquet.
2. `meta-audit` counts missing, empty and shallow paths per file and writes the full category tree.
3. `meta-classify` embeds item text (title, brand, features, description) with `BAAI/bge-small-en-v1.5`. It then fits a softmax classifier on items whose path is observed.
   - Classes are level-$L$ paths with at least 200 items.
   - Training uses a uniform sample of up to 200,000 labelled items. A fifth of the sample is held out for metrics.
   - An item with a shallow path may only receive labels below that path.
   - A prediction is kept only if its confidence reaches a threshold. The threshold is the lowest one at which held-out precision reaches 0.8. Other items stay `unknown`.
   - Files that are empty learn from a donor file (`DONORS` in `src/cp4285/common/metadata.py`). For example, `All_Beauty` uses `Beauty_and_Personal_Care`. `Unknown` and `Subscription_Boxes` use every file.
4. `meta-group` writes one row per item and file, with `parent_asin, domain, subcategory, full_path, label_source, confidence`. It also writes `subcategory_summary.parquet` with item counts per label source. Where a 5-core rating file is present, the summary adds users, ratings and density $= \text{ratings} / (\text{users} \times \text{rated items})$.
5. `meta-publish` uploads only these derived tables to a private Hugging Face dataset. Raw metadata stays at the source.

A full-scale zero-shot NLI model would need one forward pass per item and label, over about 48M items and hundreds of labels. The embedding classifier needs one pass per item.

## Commands

```sh
uv sync --locked --extra meta
uv run cp4285 meta-download --category all      # about 26 GB compressed; resumable
uv run cp4285 meta-audit
uv run cp4285 meta-classify --category all --device cuda
uv run cp4285 download --domain all             # 5-core rating files for the density summary
uv run cp4285 meta-group
HF_TOKEN=... uv run cp4285 meta-publish --repo <user>/amazon23-item-categories
```

Outputs go to `data/meta/out`. Each stage refuses to overwrite its outputs. With `--category all`, finished categories are skipped, so a rerun resumes.

## Local dry run (CPU, 22 October)

This was a software check on two small files. It is not the reported result.

| file | items | empty paths | classes | held-out accuracy | macro-F1 | predicted / pending |
|---|---:|---:|---:|---:|---:|---:|
| Magazine_Subscriptions | 3,391 | 834 | 4 | 0.760 | 0.725 | 714 / 834 |
| Video_Games | 137,269 | 12,637 | 12 | 0.770 | 0.334 | 12,038 / 12,637 |

Row counts reconciled: all 140,660 items appear once.

A follow-up comparison on 12,000 Video_Games items changed the default L2 penalty from $10^{-4}$ to $10^{-5}$. This raised accuracy from 0.779 to 0.808 and macro-F1 from 0.325 to 0.467. The table above predates that change. Low macro-F1 comes from rare platform classes such as Virtual Reality. Video_Games classification took 12 minutes on a 16-core CPU.

## Limits

- Predicted labels are model output. Keep `label_source` in any experiment, and report results with and without predicted items.
- The 0.8 precision target is measured on observed items. Items with empty paths may differ from them, for example by having less text.
- Donor labels for empty files are an assumption. `All_Beauty` items get `Beauty & Personal Care` subcategories, which Amazon may not have intended.
- Amazon Reviews 2023 is for research use. The published table holds IDs and labels only.
