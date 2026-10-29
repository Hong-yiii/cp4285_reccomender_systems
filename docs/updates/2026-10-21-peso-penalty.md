# Progress update — 21 Oct 2026

Oct 21, 2026

Made strong enough to act, PESO's penalty cut forgetting a little (one seed, not yet significant) but did not give the paper's gain. Looking inside the model shows that the forgetting sits in the broad first code of each item, not the exact item.

## Catastrophic forgetting evaluation in RecSys

- **The published penalty cannot act.** The paper's Eq. 12 at its largest λ pulls about 10⁻⁴ as hard as the task gradient on the adapters, and the released code about 10⁻⁷. Released PESO and single LoRA log the same training loss to three or four decimals ([calibration](../PESO-REPLICATION.md#penalty-calibration--21-october-2026-seed-42)).
- **A penalty that acts (O8, λ = 400, seed 42)** ([results](../PESO-REPLICATION.md#o8-a-penalty-that-acts--21-october-2026-seed-42)):
  - **Forgetting:** backward transfer NDCG@10 is −0.0029 against single LoRA's −0.0038, and old blocks end about 8% higher.
  - **Significance:** user by user, the gain is not significant (90 vs 75 users, p = 0.28).
  - **Table 2's score:** the own-state score is unchanged (0.0169 vs 0.0167), so the paper's reported gain is not reproduced even with an acting penalty.
- **Table 2 can't see retention.** It scores each block right after training, so the transfer matrix's old-block columns are where a retention method has to show up.

## Model architecture and interpretability

- **Item-level J-lens** (all four codes, 15 layers; [results](../PESO-REPLICATION.md#item-level-j-lens--21-october-2026-seed-42)):
  - **Timing:** the broad item group is readable early, and the exact item (codes 3–4) only in the last few layers.
  - **Forgetting:** it sits in the broad code. Code 1 is the only code whose rank worsens on average, while the exact-item codes improve over updates. This corrects our earlier inference from the first code alone.
  - **No hidden retention:** users whose output got worse while the middle layers held are rarer than chance pairing predicts (9–18% against 21–25%).
- **Drift is not forgetting.**
  - **Codes 3–4:** the largest drift is inner-layer and goes with learning.
  - **Code 1:** the forgetting code has the largest item-head part, which PESO leaves unpenalized.
  - **The acting penalty:** it shrank the inner drift by 26–50% but left that item-head part unchanged, and the newest block's exact-item codes were learned less well.

## Unified package and deliverables

- **Notebook 02:** new sections for penalty size, the J-lens, the O8 comparison and the transfer matrix. All numbers come from committed aggregate reports in `replication/peso/results/seed42/reports/`, with no user IDs.
- **Notebook 03 (Colab):** a fresh-environment test on Modal ran end to end. The PESO − single LoRA gap matched exactly, while Pretrain came out about 13% lower from environment numerics. The released weights are verified lossless. The notebook also has a two-figure J-lens side note.
- **A 12-minute general-audience talk** explains the approach and these results without maths ([talk](../TALK-GENERAL-AUDIENCE.md)).

## Infrastructure lessons

- **Check free space before writing.** The 150 GB Runpod volume filled up, so publishing the O8 model failed silently and the A100 sat idle for about 3.8 h. The fix was deleting the 19 Oct debug runs' weights (82.6 GB; logs and results kept) and publishing by hand with SHA-256 checks. A 15-minute health check now includes a volume write test.
- **Runpod's EUR-IS-1 data center had a DNS and network incident.** Two new pods could not pull their image and were terminated within minutes. Pod creation also needed `allowedCudaVersions: ["13.2"]`, because `minCudaVersion` found no hosts.
- **Cost:** Runpod billing reads $38.49 for 19–21 Oct ($13.97, $20.88 and $3.64 so far for 21 Oct, which may still be updating). Modal is about $91 of its $100 budget.

## Next steps

- **[The group] Seeds:** repeat single LoRA and PESO λ = 400 on seeds 43–44 with the same evaluation pairs, to test whether the retention gain is real (about $12 on an A100; needs a Runpod top-up).
- **[The group] O2:** penalize the item embedding and output layers, at a matched strength. The J-lens ties forgetting to them.
- **[The group] λ sweep:** 100, 400 and 1,600, to trace the retention-versus-learning trade-off.
- **[The group] Team release:** publish the team-repo weights release and point notebook 03 at `main` once the branch is merged.
- **[The group] Talk:** move the general-audience talk into the deck and assign slide owners.
