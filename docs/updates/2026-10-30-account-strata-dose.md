# Account strata and critical-mass dose, 30 October 2026

First pass of the account-quality and dose experiments on Instruments, seeds 42–44. No cross-domain or item-level claim.

## Account quality

- Tooling: `experiment/user_strata.py`; `cp4285 robustness-plan --stage strata`; results in [CONTINUED-TRAINING.md](../CONTINUED-TRAINING.md#first-results-30-october-2026).
- 30 runs completed (jobs 1035987–1035989).
- Reach does not order accounts: top and bottom matched-account deltas are −2.92 and −2.90 ranks (SD 0.44 and 0.43).
- Concentrating the same event budget into fewer top-reach accounts weakens the push: k=4 uses 45 accounts at ~5.8 insertions each and moves the category −1.57 ranks, against −2.92 for 177 accounts at ~1.5 insertions each.
- The bottom-side value arms keep the full account count by construction; they are a determinism check, not an independent k contrast.

## Critical mass

- Tooling: `cp4285 robustness-plan --stage dose`; `cp4285 dose-summary`; results in [CONTINUED-TRAINING.md](../CONTINUED-TRAINING.md#first-results-30-october-2026).
- 24 corrected runs completed (jobs 1038273–1038275), after the first 21 dose folders were discarded for storing the target rank instead of the pushed rank.
- Pre-registered threshold: **1%** (mean −2.13 ranks, 95% CI −3.22 to −1.55; all three seed-level intervals below zero).
- Top-5 reaches 50% at 3% and 81% at 5%; top-10 saturates at 1%; the pushed category never reaches rank 1.
- Low doses are non-monotone: 0.1% +0.47, 0.25% −0.80, 0.5% −0.22. The 0.1% arm edited 9 users; that point is noise, not a backfire.

## Limits

One domain, three seeds, one attacked block, category targets, isolated adapters. Reach is confounded with history length and item popularity. The value invariant is an assumption, not an estimated exchange rate. The 1% threshold is the first point meeting the rule, not a smooth law.

## Next steps

- [The group] Review the strata and dose tables and confirm the 1% threshold reading.
- [The group] Decide whether the activity stratum is worth a second pass; reach showed no ordering.
- [The group] Choose one second domain for the dose curve if the 1% threshold is to be tested for transfer.
- [The group] Record the run IDs and the `bad-dumps-2026-10-29` archive in the next meeting notes.
