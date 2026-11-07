# Category-granularity poisoning (7 November 2026)

`granularity-2026-11-07.json` and `.md` are the paired analysis of the
category-granularity poisoning grid. They contain aggregate endpoints and per-seed
pairs only: no user IDs, item IDs or model weights.

The question comes from the fourth STePS image, "Forgetting only shows up at the
product level", which compares forgetting at 30, 75 and 299 Instruments labels. This
grid asks the companion poisoning question: does a 3% update move the pushed category
more or less as labels get finer, and do adapter isolation and rollback still protect
the ranking?

## Runs

| Level | Labels | Effective labels | Runs | Source |
| --- | ---: | ---: | --- | --- |
| 2 | 30 | 5.6 | reused October grid | `lm-ins-s{42,43,44}-{arm}` |
| 3 | 75 | 14.6 | 18 new | `lm-ins-cat3-s{42,43,44}-{arm}` |
| 4 | 299 | 62 | 18 new | `lm-ins-cat4-s{42,43,44}-{arm}` |

Effective labels are the ladder's block 1–4 means in
`ladder/ladder-2026-10-24.json`. Each level uses the same Instruments users, blocks
and events. The pushed category is the one whose share of blocks 1–4 test targets is
nearest 1.3%: `Keyboards & MIDI` at level 2, `Wind & Woodwind Accessories` at level 3
and `Electric Guitar Amplifiers & Preamps` at level 4. Arms are `init`,
`shared-clean`, `iso-clean`, `shared-bw3`, `iso-bw3`, `iso-bw3-drop`, seeds 42–44.
`bw3` is 3% bandwagon shills in block 1; `drop` deletes the attacked adapter
immediately; `iso-bw3` also scores its final state with the attacked adapter
switched off (the late rollback).

Level 3 and level 4 ran on the NUS SoC Slurm cluster. Each level/seed group stayed on
one GPU type: level 3 seeds 42 and 43 on H100-47, seed 44 on A100-40; level 4 seed 42
on H200-141, seed 43 on H100-96, seed 44 on H100-47 (its init ran on A100-80). The
November grid was queued as jobs 1089204–1089235, with retries 1089582–1089585,
1089830–1089834, 1090823–1090827 and 1090905. Several retries were needed for a
missing `SRC` environment and for co-tenant OOMs; a failed run never left a
`metrics.json`.

## Analysis

The runner is unchanged from the October catalogue: `adapter_blocks.py`,
`corruption.py`, `shilling.py` and `robustness.py` hashed identically in the cluster
checkout and this repository. The new launcher and analysis are commit `6f284bf`.

```sh
python replication/peso/granularity_attack.py granularity-2026-11-07.json \
    --root <cluster>/lm-runs \
    --level 2 lm-ins --level 3 lm-ins-cat3 --level 4 lm-ins-cat4 \
    --seeds 42,43,44
```

Level-2 dumps predate the per-user `push_rank` field, so `granularity_attack.py`
falls back to the run's aggregate `push` cell there. Levels 3 and 4 pair users
through the dumps and add top-10 churn and bootstrap-free sd across seeds. A negative
rank change means the pushed category moved up; a positive removal value means it
ended below clean. `drop` removes the whole block adapter, which also removes that
block's clean learning; no clean-drop control was run.

Raw dumps, `theta0.pt` files and run folders stay on the cluster and are not
committed. The committed markdown is the script's printed table; regenerate it from
the JSON with `--help`-visible settings in `granularity_attack.py`.
