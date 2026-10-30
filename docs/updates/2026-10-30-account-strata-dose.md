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

## Cross-domain replication

- Video Games and CDs & Vinyl, seeds 42–44, six doses from 0.25% to 5%, plus top/bottom reach at 3%: 54 new runs, jobs 1041298–1041303.
- The mean-rank threshold is around 1% in all three domains. CDs' formal 0.25% point is fragile (one positive seed, pooled interval crosses zero); all three CD seeds are negative from 0.5%.
- Normalised gain at 3–5% is similar across domains (0.42–0.53). Raw rank change is not comparable: CDs' clean category starts near rank 14, Instruments near 9.
- Top-5 visibility does not transfer: Instruments 55% at 3% and 81% at 5%; Video Games 25% and 46%; CDs 13% and 24%.
- Reach top > bottom in all three Video Games seeds (−3.91 against −3.23). Instruments is null; CDs is inconclusive because the bottom side ranges from 0.00 to −5.74.
- The six `cross-init` runs were unused (branches pointed at the existing `lm-{domain}-s{seed}-init`); they can be deleted.

### Next steps

- [The group] Decide whether the top-5 visibility gap is a dataset property or a target-choice artefact.
- [The group] Add seeds for the CDs reach contrast before treating the top-vs-bottom difference as real.
- [The group] Use normalised gain and top-K, not raw rank, in the poster.
