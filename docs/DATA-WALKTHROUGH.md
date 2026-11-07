# Data walkthrough — looking through and analysing this checkout

A guided tour for someone opening the repo with the intent to **read the data, not run the models**.
Every command below was run in this checkout on 6 November 2026 and the quoted output is real.
Nothing here downloads a dataset, trains anything, or writes into `runs/`, `data/` or `reports/`.

Assumed setup: `uv sync --locked` from the repo root, then `uv run <command>`. The locked environment has
**duckdb, numpy, scipy, torch** plus the optional `meta`/`llm` extras — and **no pandas, matplotlib or
pyarrow**. Read parquet with duckdb and JSON with the stdlib; that is what the recipes below do.

## Three rules before you open a file

1. **Committed evidence vs local residue.** Only ~250 files are in git, 86 of them under
   `replication/peso/results/`. `data/`, `runs/`, `reports/` and `local/` are ignored (`.gitignore:23-25`),
   so they exist only on this machine and will not survive a fresh clone.
   Check any path: `git check-ignore -v <path>`.
2. **Protocols are not comparable.** A number from the classical sweep, the ID-based SASRec arm, the
   category-label SASRec arm and the category language model answer different questions on different target
   spaces. Section 8 has the wall table. `recall@k` is an alias of `hit@k` in every `cp4285` artifact.
3. **Synthetic is not Amazon.** The offline `demo`/`metrics-check` artifacts carry generated users
   (`synthetic-user-0`) and are software checks only. Verify provenance per artifact, as in §3.

## The five layers, at a glance

| Layer | Path | Size here | In git? | What it is |
| --- | --- | --- | --- | --- |
| Published results | `replication/peso/results/` | ~3 MB | yes | metrics-only JSON from the PESO replication and the Oct attack grid |
| Local run artifacts | `runs/` | ~85 MB | no | `metrics.json` per run + per-user rank dumps |
| Item catalogue | `data/meta/`, `data/meta-dryrun/` | 637 MB + 191 MB | no | Amazon'23 metadata → level-2/3/4 subcategory tables |
| Prepared event streams | `data/processed/` | 1.3 GB | no | two cached numpy encodings of Electronics + Movies & TV |
| Classical reports | `reports/classical/`, `runs/qrrec-2026-09-28/` | ~10 MB | no | September QR-SVD/Markov sweeps on real Amazon data |

## 1. Ten minutes of orientation

```sh
uv run cp4285 doctor            # {"python": "3.12.3", "torch": "2.14.0+cu130", "duckdb": "1.5.5", "cuda": false}
du -sh data/* runs/* reports replication/peso/results
git ls-files | wc -l            # 254 tracked files
git ls-files 'replication/peso/results/*' | wc -l    # 86 committed result files
```

Then read, in this order: [README.md](../README.md) → [CONTEXT.md](../CONTEXT.md) →
[PESO-REPLICATION.md](PESO-REPLICATION.md) (protocol + every result) → the dated
[progress updates](updates/README.md). `reference.html` is the reviewer walkthrough with dated checkpoints.

## 2. Start where the evidence ships: `replication/peso/results/`

[The results README](../replication/peso/results/README.md) is the index: it names the script that wrote
each file, and gives SHA-256 for the seed-42/43/44 metric files and the analysis reports. Those are
**metrics only** — no users, items or weights.

Two files answer "what did we measure" and "what survives" fastest:

```sh
# Headline: is the PESO − single-LoRA gap larger than seed spread? (own-block NDCG@10, blocks 1-3)
uv run python -c "
import json
s = json.load(open('replication/peso/results/summary-3seeds.json'))
for m in ('pretrain','lora','peso'):
    a = s[m]['averages']['blocks 1-3 (paper)']['ndcg@10']
    print(f\"{m:9} mean={a['mean']:.4f} sd={a['sd']:.4f}\")
d = [p-q for p,q in zip(s['peso']['averages']['blocks 1-3 (paper)']['ndcg@10']['seeds'],
                        s['lora']['averages']['blocks 1-3 (paper)']['ndcg@10']['seeds'])]
print('peso - lora per seed:', [round(x,5) for x in d])"
# pretrain  mean=0.0137 sd=0.0010
# lora      mean=0.0147 sd=0.0008
# peso      mean=0.0146 sd=0.0007
# peso - lora per seed: [8e-05, -0.00037, -7e-05]
```

The paper claims +0.0012; the paired difference here is −0.0001 with per-seed scatter of the same size.

```sh
# The attack grid's pre-computed relations (204 runs, dose → spike → residue → cost → stealth)
uv run python -c "
import json
for line in json.load(open('replication/peso/results/grid/lattice.json'))['relations'][:6]: print(line)"
# | relation | n | Spearman rho | signs | note |
# | spike_push_rank -> residue_push_rank | 159 | +0.45 | 98/159 agree in sign | one-off attacks |
# | spike_push_rank -> residue_push_rank | 90 | +0.69 | 69/90 agree in sign | one-off, isolated layout only |
# | spike_push_rank -> residue_push_rank | 69 | +0.27 | 29/69 agree in sign | one-off, shared layout only |
```

Each `runs[]` row in that file is one attacked branch with `spike_push_rank` (effect in the attacked block),
`residue_push_rank` (effect after clean updates), `decay_slope`, `separability_auc` (detectability),
`cost_ndcg_mean_late`, `retention_ndcg` and the two rollback deltas. `interventions[]` holds the
adapter-deletion arms. `notebooks/02_peso_replication.ipynb` redraws the same committed files as tables.

## 3. Read one local run end to end: `runs/lm-runs/grid/`

114 branch folders (`lm-<domain>-s<seed>-<arm>`) from the 24–26 October grid, plus the seed-42 pilot and
`grid-aggregate.json`. One run is `metrics.json` + optional `dumps/`:

```sh
uv run python -c "
import json
m = json.load(open('runs/lm-runs/grid/lm-ins-s42-iso-bw3/metrics.json'))
print(sorted(m))                                    # stage, config, cells, losses, attack, rollback, timing, learner
print(m['config']['data'], '|', m['config']['revision'])   # provenance: source blocks + code revision
print(m['push_name'])                               # the category that was pushed
c = m['cells']['state1_block1']
print(round(c['hit@10'],4), c['users'], {k: round(v,3) for k,v in c['push'].items() if k!='users'})"
```

Anatomy worth memorising:

- `cells["state<i>_block<j>"]` — the continual-learning matrix as a dict. `state i` = trained through block
  `i`; the own-state cells (`state1_block1`, …, `state4_block4`) plus the retention cells
  (`state4_block0`…`state4_block3`). Each cell has `users`, `hit@1/5/10`, `ndcg@`, `mrr@` and, when a
  category was pushed, a `push` sub-dict (`mean_rank`, `hit@1/5/10`) — **lower `mean_rank` is better**.
- `attack` — the injected stream as measured, not as requested: `alpha` vs `alpha_requested`, `shill_users`,
  `shill_examples`, `real_examples`, `target_share`, `target_items`, `js_training_categories`, and
  `separability_auc` (1.00 = the fakes are trivially separable, i.e. loud).
- `rollback` / `timing` — adapter-deletion scores and `train_seconds`/`score_seconds`.
- `learner` — base model and revision, LoRA rank/alpha/target modules.

Re-derive the summary table instead of hand-computing it (create-only `--json` if you want to save it):

```sh
uv run cp4285 adapter-summary runs/lm-runs/grid/lm-ins-s42-init/metrics.json \
  runs/lm-runs/grid/lm-ins-s42-iso-{clean,bw1,bw3,bw5,clone3,clone3-drop}/metrics.json
```

| Branch | Condition | Attack | Own-block Hit@1 | RetentionLoss_A | Push rank change, attacked block | Shill AUC |
| --- | --- | --- | --- | --- | --- | --- |
| iso-clean | isolated | none | 0.5885 | −0.0010 | | |
| iso-bw1 | isolated | bandwagon 0.01 | 0.5881 | +0.0000 | −0.2437 | 1.000 |
| iso-bw3 | isolated | bandwagon 0.03 | 0.5880 | −0.0010 | −1.7805 | 1.000 |
| iso-bw5 | isolated | bandwagon 0.05 | 0.5861 | +0.0010 | −2.8569 | 1.000 |
| iso-clone3 | isolated | clone 0.03 | 0.5869 | −0.0040 | −1.6310 | 0.779 |

**Dump-format trap.** `runs/lm-runs/grid/*/dumps/state*_block*.jsonl` from `adapter-blocks` are one line per
user with `{user, target, rank}` — the pushed rank only. `cp4285 flips` needs the *other* layout,
`{user, target, ranked[]}` written by `replication/peso/peso_eval.py`, and raises `KeyError: 'ranked'` on
these. The `ranked` dumps are not in this checkout (§6).

## 4. The catalogue layer: `data/meta/`

Built by `cp4285 meta-*` from Amazon'23 metadata. The full 22 October run labelled 35.4M items, and its
tables are published as the private HF dataset `randomwish/amazon23-item-categories`
([ITEM-CATEGORIES.md](ITEM-CATEGORIES.md)). Local contents are asymmetric, which trips people up:

- `out/` — the whole run: `item_categories/domain=*/**/*.parquet` (34 Hive directories, 30 distinct domains),
  `subcategory_summary.parquet`, `category_tree.parquet`, `metrics/<domain>.json`.
- `items/` and `raw/` — **only** `Musical_Instruments`. `meta-audit`, `meta-classify` and `meta-group` need
  these inputs, so they cannot be re-run for other domains here.
- `out-level3/`, `out-level4/` — Instruments only, at 75 and 289 subcategories.
- `data/meta-dryrun/` — the 22 October CPU dry run (Magazine Subscriptions, Video Games). Ignore it.

```sh
uv run python - <<'PY'
import duckdb
con = duckdb.connect()
print(con.execute("""SELECT label_source, count(*) n, round(100.0*count(*)/sum(count(*)) OVER (),1) pct
  FROM read_parquet('data/meta/out/item_categories/*/*.parquet') GROUP BY 1 ORDER BY 2 DESC""").fetchall())
# [('observed', 31493162, 89.0), ('predicted', 2935207, 8.3), ('unknown', 964820, 2.7)]

print(con.execute("""SELECT domain, count(*) subcats, sum(items) items, sum(rated_items) rated
  FROM 'data/meta/out/subcategory_summary.parquet'
  WHERE domain IN ('Musical_Instruments','Video_Games','CDs_and_Vinyl') GROUP BY 1 ORDER BY 1""").fetchall())
# [('CDs_and_Vinyl', 75, 701959, 89370), ('Musical_Instruments', 41, 213593, 24587), ('Video_Games', 53, 137269, 25612)]
PY
```

`subcategory_summary` columns: `domain, subcategory, items, observed, predicted, unlabelled, rated_items,
users, ratings, density`. `item_categories`: `parent_asin, subcategory, full_path, label_source, confidence`.
A `predicted` label is a TF-IDF guess, so catalogue-level counts (41 Instruments subcategories) are **not**
the experiment's categories: the study's labels come from the blocks actually used, recorded in
`runs/categories/<dataset>-cat2/category_stats.json` — Instruments 30 categories (50,023 observed /
3,486 predicted, classifier holdout accuracy 0.9198), Video Games 43 (24,883 / 1,052, 0.8059),
CDs & Vinyl 59 (39,191 / 0). The sibling `.index.cat2.json` maps item index → `<cat_N>` token and
`.label_source.json` records, per item, whether that mapping is observed or predicted.

## 5. Prepared streams: `data/processed/` and `runs/*/prepared.json`

Two cached encodings of Amazon'23 Electronics (A) + Movies & TV (B), written by
`experiment/classical_data.py`. Keys are `au/ai/at` (+`bu/bi/bt`) and the scalars `an`, `bn`, `n_users` —
**`an`/`bn` are item-vocabulary sizes, not row counts**; rows are `len(au)`. Item ids index each domain's own
vocabulary, while users share one vocabulary across A and B, so user overlap is real.

Verified on 6 November:

| Cache | A events / items | B events / items | shared users |
| --- | --- | --- | --- |
| `…-0f7515e0279b.npz` | 15,473,536 / 368,228 | 7,441,129 / 197,943 | 2,129,328 |
| `…-712dd8616b3b.npz` | 15,473,536 / 368,228 | 17,158,519 / 747,764 | 7,488,824 |

Same A events (identical item and timestamp multisets) in two different row orders, and two different
Movies & TV vintages. **Never mix arrays from the two files** — the shared user numbering differs, and no
original `parent_asin`/`user_id` survives the encoding.

Split here is official Amazon leave-last-out (last = test, second-to-last = validation), which is *not* the
cutoff protocol of the neural pilot. The neural pilot's prepared form is JSON instead:
`runs/metrics-check-1792660917/prepared.json` (`metadata`, `item_ids`, `train`, `validation`, `retention`,
`continuation`) beside its `audit.json` — a good read for what the audit records (sha256, duplicate and tied
row counts, sequence-length percentiles) — but that particular run is **synthetic** (24 users,
`synthetic-user-0`).

## 6. Classical real-data reports, and what is *not* here

`reports/classical/**/study_*.json` (scope `amazon`) and `runs/qrrec-2026-09-28/*.json` hold 14 sweeps —
one row per model × contamination level (`model, seed, level, ndcg@10, recall@10, base_share, injected,
frac_actual`), with sibling `_per_user.npz` (`ranks_<model>` per-user target ranks) and the `qrrec-README.md`
that explains the commands. Interpreted numbers: [CLASSICAL.md](CLASSICAL.md).

Deliberately absent from this checkout:

| Missing | Where it is | Consequence |
| --- | --- | --- |
| `data/raw/Electronics.csv.gz`, `Movies_and_TV.csv.gz` | broken symlinks into a deleted `dev/cp4285-recommender-systems/` | `audit`/`prepare`/`classical` cannot run; re-download needs an explicit decision |
| PESO blocks (`…/data/Instruments_cl_ori`) | NUS SoC `/mnt/scratch/w/wyenzhe/peso/`, pods; upstream has no licence | every `metrics.json` here cites a path that is not local |
| Checkpoints, weights, J-lens tensors | pods / HF release | only metric aggregates are committed |
| Nov dose, strata, activity, dormancy runs | SoC cluster, Slurm jobs 1035987–1045609 | `cp4285 dose-summary` on the Oct grid fails: *"Dumps lack push_rank"* — read the written tables in [CONTINUED-TRAINING.md](CONTINUED-TRAINING.md) |
| `ranked` top-k dumps | not retained | `cp4285 flips` has no local input |

## 7. Five questions you can answer in one sitting

Every command below has been run in this checkout; the `#` lines under it are its real output.

**1. Does a pushed category survive later clean updates, and in which layout?** Average the committed
lattice over domain × adapter topology:

```sh
uv run python -c "
import json, statistics as st
from collections import defaultdict
agg = defaultdict(list)
for r in json.load(open('replication/peso/results/grid/lattice.json'))['runs']:
    if r['kind'] in ('bandwagon','clone'): agg[(r['domain'], r['condition'])].append(r)
for k in sorted(agg):
    v = agg[k]
    print(k, 'n=' + str(len(v)), 'spike', round(st.fmean(x['spike_push_rank'] for x in v), 2),
          'residue', round(st.fmean(x['residue_push_rank'] for x in v), 2))"
# ('cd','isolated') n=15 spike -4.04 residue -1.29    ('cd','shared') n=9  spike -3.19 residue 0.06
# ('ins','isolated') n=27 spike -1.52 residue -0.19   ('ins','shared') n=21 spike -1.63 residue -0.02
# ('vg','isolated') n=15 spike -1.30 residue -0.08    ('vg','shared') n=9  spike -1.78 residue 0.15
```

Lasting residue is a CDs + isolated-adapter reading; every shared-adapter layout ends at or above clean.

**2. How does the push scale with dose?** The same cell along the Instruments dose arms:

```sh
uv run python -c "
import json
for arm in ('clean','bw1','bw3','bw5'):
    m = json.load(open(f'runs/lm-runs/grid/lm-ins-s42-iso-{arm}/metrics.json'))
    c = m['cells']['state1_block1']; a = m.get('attack') or {}
    print(arm, 'alpha', round(a.get('alpha',0),3), 'fakes', a.get('shill_users'),
          'pushed events', a.get('shill_examples'), 'auc', a.get('separability_auc'),
          '| pushed rank', round(c['push']['mean_rank'],2), 'pushed top5', round(c['push']['hit@5'],3),
          '| own hit@10', round(c['hit@10'],4))"
# clean alpha 0    fakes None pushed events None auc None    | pushed rank 9.87 top5 0.053 | hit@10 0.9876
# bw1   alpha 0.01 fakes 15   pushed events 87   auc 1.0     | pushed rank 9.63 top5 0.048 | hit@10 0.9887
# bw3   alpha 0.03 fakes 62   pushed events 263  auc 1.0     | pushed rank 8.09 top5 0.073 | hit@10 0.9881
# bw5   alpha 0.05 fakes 123  pushed events 448  auc 0.99993 | pushed rank 7.01 top5 0.157 | hit@10 0.9881
```

Rank gain grows with dose while the original task does not move (own-block Hit@10 varies by 0.001), and the
1% top-5 share sits *below* clean: the low-dose end is noisy and non-monotone, which is why the written
threshold reading is cautious. `cp4285 dose-summary` is the paired-bootstrap version but needs
post-October dumps (§6).

**3. Is the attack detectable?** `attack.separability_auc` by kind: bandwagon ≈1.000 at every dose, clone
0.733–0.792. Loud fakes and quiet clones push about the same (−1.78 against −1.63 ranks at 3%).

**4. How much of a category label is a model guess?** Label quality differs by study domain, which bears on
question 1:

```sh
uv run python -c "
import json, glob
for p in sorted(glob.glob('runs/categories/*/category_stats.json')):
    s = json.load(open(p)); lab = s['label_source']; n = sum(lab.values())
    print(p.split('/')[2], 'level', s['level'], 'cats', len(s['categories']), lab,
          'predicted %', round(100*lab.get('predicted',0)/n, 1))"
# CDs_and_Vinyl_cl_0.6_0_5_3_0-cat2  level 2 cats 59 {'observed': 39191}                    predicted % 0.0
# Instruments_cl_ori-cat2            level 2 cats 30 {'observed': 50023, 'predicted': 3486} predicted % 6.5
# Video_Games_cl_0.6_0_5_3_0-cat2    level 2 cats 43 {'observed': 24883, 'predicted': 1052} predicted % 4.1
```

**5. Does the category channel help ID-based SASRec?** Re-derive the committed summary from its own six runs:

```sh
uv run cp4285 blocks-summary runs/cluster-blocks/blocks-sasrec-seed42-clr1e-4/metrics.json \
  runs/cluster-blocks/blocks-sasrec-seed4{3,4}/metrics.json \
  runs/cluster-blocks/blocks-category-seed4{2,3,4}/metrics.json
```

Blocks 1–3, own state, three seeds — sasrec Hit@10 0.0212 ± 0.0016, category 0.0202 ± 0.0012, popularity
0.0191, paired difference −0.0010 ± 0.0019 — which reproduces the committed
`runs/cluster-blocks/summary-3seeds.json` exactly. No top-10 change; what moves is rare-target ranking, in
the `unseen` / `1-4` / `5+` slices of `averages` (targets seen 5+ times: 0.0452 against 0.0430).

## 8. Before you quote a number

| Family | Data | Target space | Typical scale | Documented in |
| --- | --- | --- | --- | --- |
| Classical sweep | Amazon Electronics + Movies & TV | items, leave-last-out | NDCG@10 ~0.03 | [CLASSICAL.md](CLASSICAL.md) |
| SASRec on PESO blocks (`cp4285 blocks`) | Instruments blocks | items (~53K) | Hit@10 ~0.02 | [CONTEXT.md](../CONTEXT.md), `runs/cluster-blocks/summary-3seeds.json` |
| Category LM (`cp4285 adapter-blocks`) | same blocks, PR 14 level-2 labels | 30/43/59 categories | Hit@1 ~0.59, Hit@10 ~0.99 | [PESO-REPLICATION.md](PESO-REPLICATION.md), [CONTINUED-TRAINING.md](CONTINUED-TRAINING.md) |
| PESO item replication | PESO released blocks | semantic-ID products | NDCG@10 ~0.014 | [PESO-REPLICATION.md](PESO-REPLICATION.md) |

Also: category LM Hit@10 near 1.0 means the target space is tiny, not that the model is good — always compare
to the run's own clean twin or the popularity/history baseline in the same table. Rank deltas are
baseline-confounded across domains, so prefer normalised gain and top-K visibility
([CONTINUED-TRAINING.md, dataset covariates](CONTINUED-TRAINING.md#dataset-covariates-2-november-2026)).
Three seeds is the norm; a single-seed reading has already failed to replicate twice. Artifacts are
create-only on purpose — write new summaries to a new path, never over a run folder.

## 9. To produce data rather than read it

`uv run cp4285 demo --output runs/<new-name>` is the only no-approval, offline, CPU path and yields synthetic
numbers. `download`, `audit`, `prepare`, `pilot` and `classical` need the real 5-core files and an explicit
decision to download them. `blocks`, `block-categories` and `adapter-blocks` need PESO-format block folders
plus the `llm` extra, and GPU time that bills someone's account; `robustness-plan` prints the matrix but
requires `--data-root`, `--categories-root` and `--model`. Full detail: [IMPLEMENTATION.md](IMPLEMENTATION.md).
