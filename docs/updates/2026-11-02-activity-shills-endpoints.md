# Activity, classic shills and endpoint limits, 2 November 2026

75 runs across jobs 1045579–1045587 and 1045607–1045609. All completed without failed lines. Results and explanations: [CONTINUED-TRAINING.md](../CONTINUED-TRAINING.md#activity-classic-shills-and-endpoint-probes-2-november-2026).

## Account activity

- High activity beat low activity in every domain at 3% and matched accounts: Instruments −3.65 against −1.58; Video Games −5.14 against −2.52; CDs −7.09 against −2.59.
- Low activity is not dormancy. The result weakens the “dormant accounts are easy” version of the threat; true dormant accounts remain untested.
- Plausible explanation: high-activity users contribute more training targets and more diverse item contexts. Activity is confounded with history length and item popularity.
- Per-user check (Instruments seed 42, random 3%): edited users −5.17 against untouched −5.12; edited × activity +0.06 (95% bootstrap −0.19 to +0.47). The aggregate effect is model-level, not that edited high-activity users’ own ranks move more.

## Classic shills

- Random and Average fills are as strong as bandwagon/clone at 3%; the attack is not an artefact of the more elaborate fills.
- Profile AUC: random 0.82–0.87, average 0.94–0.95, bandwagon 1.00, clone 0.63–0.76.
- Clone is the stealthy fake-account attack; average and bandwagon are easy to separate.

## CDs with six seeds

- Mean-rank threshold 0.5%: −1.73 (95% CI −4.66 to −0.60).
- Top-5 stays at 3–4% even at 1%, so CD does not reproduce the Instruments visibility effect.
- Reach: top −4.85 (SD 1.21), bottom −3.62 (SD 2.14); top stronger in four of six, still noisy.

## Breadth and endpoint

- k=4 weakens the attack against full accounts: Video Games −2.11 against −3.91; CDs −3.84 against −4.77.
- 10%/20% on Instruments: mean rank −6.13/−6.45, top-5 97–99%/99.7–99.9%, top-1 0–0.5%.
- Plausible explanation: fewer accounts touch fewer user contexts, and repeated insertions into one history are partly redundant. Top-5 saturates before top-1 because top-1 is a single slot with a dominant alternative per user.

## Next steps

- [The group] Re-run the activity contrast with longer histories held fixed, or define true dormancy by a time gap.
- [The group] Use the Random/Average result as the Lam–Riedl replication and cite the paper in the related-work section.
- [The group] Decide whether the CD reach contrast is worth more seeds or should be reported as inconclusive.
- [The group] Keep top-1 out of the critical-mass claim; only top-5 and mean rank are supported.

## Dataset covariates

- Top-5 visibility tracks category concentration: Instruments (30 categories, HHI 0.298, 55% at 3%), Video Games (43, 0.204, 25%), CDs (59, 0.113, 13%).
- Normalised gain at 3–5% stays flat (0.42–0.53), so the mean-rank effect is similar; the visibility gap follows the length of the category tail.
- Target base rate, density, cross-block overlap and JS drift do not order the gap. Ratings are not in the block files.
- Three domains only: the covariates are correlated, so this is descriptive, not causal.

Table: `CONTINUED-TRAINING.md#dataset-covariates-2-november-2026`.
