# Category-granularity poisoning, 7 November 2026

The fourth STePS image compares forgetting at 30, 75 and 299 Instruments labels. This
share asks the poisoning counterpart: does a 3% update move the pushed category more or
less as labels get finer, and do adapter isolation and rollback still protect the
ranking? Results and limits:
[CONTINUED-TRAINING.md](../CONTINUED-TRAINING.md#category-granularity-poisoning-at-30-75-and-299-labels-7-november-2026).

## What ran

- 36 new Instruments runs on the NUS SoC cluster: PR 14 levels 3 and 4 (75 and 299
  labels), seeds 42–44, `init`, shared/isolated clean, 3% bandwagon in block 1,
  immediate rollback, and late rollback from the isolated attacked run.
- Level 2 (30 labels) reuses the October grid. Effective labels are 5.6, 14.6 and 62.
- Targets are the category nearest 1.3% of blocks 1–4 test targets: `Keyboards & MIDI`,
  `Wind & Woodwind Accessories`, `Electric Guitar Amplifiers & Preamps`.
- Clean twin for every attacked run; attacked users never enter an evaluation set.

## Results

- Isolated target moves **−1.6 ± 0.6**, **−16.9 ± 3.2** and **−8.6 ± 9.6** ranks at
  30/75/299 labels.
- Top-5 lift is **+4.0**, **+0.5** and **+9.3** percentage points; top-10 lift is
  **+18.5**, **+2.7** and **+19.5** pp.
- The 299 cell is seed-dominated: one null seed and two large pushes. Do not read its
  mean as a stable effect.
- Isolated residue after three clean updates: **−2.7 ± 1.2** ranks at 75 (all seeds),
  against −0.3 at 30 and +0.1 at 299. Shared adapters wash out at every level.
- Late rollback leaves the target **+1.9**, **+5.5** and **+9.6** ranks below clean at
  30/75/299 — the only monotone protection result.
- Immediate deletion at 75 still leaves the target **7.5 ± 1.9** ranks above clean;
  at 30/299 the direction is inconsistent.
- On average, 7.9% of the ten top-10 slots change at 75 and 11.7% at 299.

## Interpretation

- Raw rank change is baseline-confounded: the target starts at #9, #39 and #26. The
  visible effect follows target depth, not attack size alone.
- Finer labels do **not** simply make the attack stronger. They move the mean more at
  75, but top-5 stays protected because the target starts deep. At 299 the seed spread
  dominates.
- Removing the poisoned adapter late increasingly buries the target as labels get
  finer. Deleting the whole block adapter also removes that block's clean learning, so
  part of the drop is not poison removal.
- The target changes with the level by the 1.3% rule, so this is a granularity-level
  comparison, not a causal effect of label count. One domain, one model, one attack.

## Next steps

- [The group] Task: decide whether the 299-label cell needs more seeds or a second
  domain before it is summarised anywhere.
- [The group] Task: run a clean-drop control (drop a clean block adapter) to separate
  over-correction from the loss of clean block learning.
- [The group] Task: keep these as agent-proposal results; they are PESO Instruments
  blocks, not our Amazon Electronics/Movies & TV results.
