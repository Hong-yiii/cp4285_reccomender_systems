# Progress update — 20 Oct 2026

Oct 20, 2026

We replicated PESO, an LLM-based continual recommender (ICLR 2026), on its released Instruments blocks with three seeds. Its baselines reproduce, but its claimed gain does not. In the released code the PESO penalty is about a millionth of the loss, so PESO trains almost exactly like plain continual LoRA. The transfer matrix, per-user flips and a Jacobian-lens probe all agree.

## Catastrophic forgetting evaluation in RecSys

- **Protocol (PESO's).** Interactions are split by time: block 0 (the first 60%) is for initial training, followed by four incremental blocks. Each block is tested leave-one-out per user. The model generates 10 items by constrained beam search, scored with Hit@5/10 and NDCG@5/10, averaged over D2–D4 ([protocol](../PESO-REPLICATION.md#setup)).
- **NDCG@10, three seeds (paper in brackets):** Pretrain 0.0137 ± 0.0010 (0.0131); single evolving LoRA 0.0147 ± 0.0008 (0.0150); PESO 0.0146 ± 0.0007 (0.0162) ([results](../PESO-REPLICATION.md#table-2-replication--20-october-2026-seeds-42-43-44)).
- **PESO's gain over single LoRA does not reproduce.** Paired by seed, PESO − single LoRA is −0.0001 ± 0.0002 NDCG@10 against the paper's +0.0012, and +0.0002 ± 0.0001 Hit@10 against +0.0015.
- **Seed noise matches the claimed effect.** Single LoRA's gain over Pretrain ranges from −0.0008 to +0.0025 NDCG@10 across seeds, so one seed cannot show a 0.0012 effect.
- **Transfer matrix (seed 42).** Every saved state was tested on every block (20 beams, ≤2,500 users per block). Backward transfer on NDCG@10 is −0.0038 for single LoRA and −0.0061 for PESO: both forget, and PESO does not forget less ([results](../PESO-REPLICATION.md#transfer-matrix-and-per-user-flips--20-october-2026-seed-42)).
- **The counter: hits churn under flat averages.** At each update about 45% of earlier top-10 hits are lost and nearly as many new ones gained. Dormant and continuing users lose slightly more (46–53%).
- **Per user, PESO and single LoRA rarely disagree.** Only 11–27 of about 2,000 users per block differ between them, and no state–block pair is significant (smallest McNemar p = 0.06).
- **Metric cutoff is open.** The meeting notes mention Recall@20, and our meeting synthesis records @10 vs @20 as undecided. The transfer matrix now reports both (20 beams).

## Model architecture and scaling strategy

- **Model.** Llama-3.2-1B with LoRA rank 8. Items are tokenized into 4-token semantic IDs (RQ-VAE over item text): tokenization, not plug-and-play LLM use.
- **Finding: most weights sit outside the penalty.** PESO's penalty covers only the LoRA adapters (5.6M parameters). The item-token embeddings and output layer (about 265M each) train without it ([finding 3](../PESO-REPLICATION.md#code-and-data-findings-checked-not-hypotheses)).
- **Finding: the penalty barely acts.** The KL is taken over a softmax of every entry of each LoRA matrix (4,096–65,536 entries), which divides it by the matrix size. Measured from checkpoints it is 10⁻⁶ against a loss of about 2, with a gradient about 10⁻⁷ of the training gradient ([finding 10](../PESO-REPLICATION.md#code-and-data-findings-checked-not-hypotheses)).
- **Finding: the item tokenizer saw future items.** It covers all 53,509 items, including 13,038 that first appear after block 0 (finding 6).
- **Scaling.** Only 1B so far. The natural next size is Llama-3.2-3B: same family and tokenizer, with backbone size as the single variable.

## E-commerce simulation and seasonal shifts

- PESO's stream is a time shift within one category. Our mixture or seasonal-percentage question is not yet covered.
- A mixture stream needs our own block builder, plus an Amazon metadata download for item text. PESO releases tokenizer code but not its block builder.
- The meeting's "alter training methodology, not model structure" fits PESO-style regularizers. Candidate next tests: extending the penalty to the item-token layers, then replay.

## Unified package and deliverables

- **Repository.** `replication/peso/` holds the Modal and Runpod runners, our evaluator, the user-group labeller and the summariser. The committed metric files (`replication/peso/results/`) let anyone reproduce the tables from a clean clone.
- **Notebook 02.** `notebooks/02_peso_replication.ipynb` shows the comparison tables, paired differences, the transfer matrix, the flip counter, the penalty size and the J-lens probe, all from committed files. Figures wait on a plotting dependency decision.
- **Notebook 03 (Colab).** `notebooks/03_peso_colab.ipynb` clones the repo, downloads our trained seed-43 weights from a GitHub release (9.7 GB, Llama 3.2 licence included) and re-evaluates them with PESO's own evaluator. It points at a testing fork until it is verified.
- **Interpretability result.** The Jacobian lens (Anthropic, July 2026) reads the coming recommendation from layer 4, about 4× better than the logit lens. Internal and output forgetting mostly move together. PESO's internal drift equals single LoRA's ([results](../PESO-REPLICATION.md#jacobian-lens-probe--20-october-2026-seed-42-direction-o7)).

## Infrastructure lessons

- **Cost so far:** Modal about $87 of the $100 workspace budget, including about $16 lost to a client disconnect plus preemption restarts. Runpod $24.08 (Runpod billing, 19–20 October).
- **Use on-demand pods for long runs.** Modal GPU containers are preemptible, and PESO cannot resume mid-block.
- **Train on one GPU.** Two-GPU training hit a copy race in upstream PESO and a checkpoint race in transformers 4.52.4. One GPU with gradient accumulation 4 keeps the same effective batch.
- **Host maintenance can break CUDA.** A Runpod maintenance restart left CUDA unusable (error 802) for about 1 h 45 min.

## Next steps

- **[The group] Review result:** decide whether PESO stays as the LLM baseline given no measured gain, and fix the metric cutoff (@10, @20 or both).
- **[The group] Agree protocol:** datasets, blocks, metrics and seeds for the common benchmark.
- **[The group] Working penalty (O8):** rerun PESO's incremental blocks with the penalty scaled by matrix size, the single variable, then a small λ sweep. This tests whether a proximal penalty that acts helps at all.
- **[The group] Verify the Colab notebook:** run notebook 03 from the fork, then point it at the team repo and publish the team release.
- **[The group] Same-protocol baselines:** run SASRec and a classical model on PESO's Instruments blocks, so all models share blocks and metrics.
- **[The group] Scaling:** first point, Llama-3.2-3B, one seed (about $45; needs a Runpod top-up).
- **[The group] Interpretability follow-up:** extend the J-lens readout from the first code token to whole items, and repeat on a second seed.
- **[The group] Poster:** pick figures from notebook 02 and agree the plotting dependency.
