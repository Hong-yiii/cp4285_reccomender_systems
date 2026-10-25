# PESO replication and extension directions

Started 19 October 2026; replication completed 20 October 2026 with three seeds. **Status: Pretrain and single evolving LoRA reproduce; PESO's reported gain over single evolving LoRA does not.** This is a user-requested exploratory workstream, not a team-confirmed change to the project protocol. Numbers here are measured on PESO's released, preprocessed Amazon Musical Instruments blocks. They are **not** results for our Electronics / Movies & TV setup and are not comparable with the SASRec or classical numbers elsewhere in this repository.

## What is being replicated

Yoo et al., "Continual Low-Rank Adapters for LLM-based Generative Recommender Systems", ICLR 2026 ([arXiv 2510.25093v3](https://arxiv.org/abs/2510.25093), [code](https://github.com/hsyoo32/peso)). A Llama-3.2-1B model generates the next item as semantic-ID tokens (an RQ-VAE code per item, built from item text). Interactions are split chronologically: the first 60% form block D1, which the paper calls the pretraining set, and the rest form four equal incremental blocks. In this project's terms, D1 training is *initial offline training* of the recommender; Llama-3.2-1B's language-model pretraining by Meta is separate. A single LoRA adapter keeps training block after block. PESO adds a penalty, weighted by λ, on the KL divergence between the softmax-normalized current adapter and the adapter saved after the previous block.

Target: paper Table 2, Instruments, averaged over D2–D4. The values below were checked against the arXiv v3 PDF, p. 8; the PDF itself is not stored in this repository.

| Method | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| Pretrain | 0.0166 | 0.0216 | 0.0115 | 0.0131 |
| Single evolving LoRA | 0.0181 | 0.0253 | 0.0127 | 0.0150 |
| PESO | 0.0193 | 0.0268 | 0.0138 | 0.0162 |

Success criterion, set before running: the ordering Pretrain < Single < PESO is reproduced, and NDCG@10 lands within about 10% relative of the table. Otherwise the gap is documented with the seed spread.

## Setup

- **Upstream code:** commit `edbec5f68ddc83e39f22a041f1f2eaf1c4ad172e` (pushed 10 May 2026). The repository has **no licence file**, so it is cloned at that commit inside the run image and in ignored `local/peso/`. It is never copied into this repository. [`replication/peso/modal_app.py`](../replication/peso/modal_app.py), [`peso_eval.py`](../replication/peso/peso_eval.py) and [`user_groups.py`](../replication/peso/user_groups.py) are our own code.
- **Data:** the released `Instruments_cl_ori` block files, semantic-ID index and item metadata from that commit. Neither the RQ-VAE tokenizer nor the block builder is rerun.
- **Hardware:** Modal, one A100-80GB per stage, using upstream's `torchrun` + DeepSpeed ZeRO-2 bf16 path with a single process. Plain LoRA and PESO train in parallel containers after the shared pretraining stage. Training writes checkpoints to container-local disk; finished runs are copied to the `cp4285-peso` volume.
- **Software:** upstream's pins (torch 2.5.1+cu121, transformers 4.52.4, peft 0.15.2, accelerate 1.7.0, deepspeed 0.16.9, flash-attn 2.7.4.post1, numpy 1.26.4) on a CUDA 12.1.1 devel image. `pip freeze`, `nvcc` and `nvidia-smi` output are saved with each training stage.
- **Base model:** `meta-llama/Llama-3.2-1B`, gated, pinned to revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`. One copy lives on the volume and is downloaded with the workspace's Hugging Face secret.
- **Upstream settings kept:** LoRA r=8, α=16, dropout 0.05 on q/k/v/o/gate/up/down; `embed_tokens` and `lm_head` fully trained; learning rate 2e-4 for block 0 and ×0.1 later; cosine schedule, warm-up 0.01, weight decay 0.01, AdamW; 4 epochs per block; micro-batch 8 and effective batch 32; validation-selected epoch per block; history window 20; temperature 0.8; λ=2.0; constrained beam search with 10 beams, as in the README (the code default is 20); seed 42.

**Deviations from upstream, both forced by the environment:**

1. **Python 3.10 instead of 3.9.** Modal images need Python ≥ 3.10. Every package version is unchanged.
2. **One GPU with gradient accumulation 4, instead of two GPUs with accumulation 2.** The per-step micro-batch (8), effective batch (32), optimizer, schedule and data stay the same; only the sampling order differs, as it would with any other seed. Two-process runs on Modal failed twice, in two different places:
   - **Block-0 copy race in upstream code.** `continual_train.py` lets rank 1 continue once the copied block-0 *directory* exists. Rank 1 then read a half-copied `model.safetensors` (`MetadataIncompleteBuffer`).
   - **Best-checkpoint race in transformers 4.52.4.** `Trainer._save_checkpoint` checks on every rank whether the newest best checkpoint folder exists (`trainer.py:3204–3206`), but only rank 0 writes the model and deletes old checkpoints. A rank that checked before rank 0's write kept the previous best, which rank 0 then deleted, giving `Can't find a valid checkpoint … checkpoint-12`. This happened on a network volume and again on local disk.
   
   Neither failure involves the PESO method itself. A single process removes both.

Commands, from the repository root:

```sh
modal run replication/peso/modal_app.py --stage smoke --run smoke-<name>   # pipeline check
modal deploy replication/peso/modal_app.py
python replication/peso/calls.py spawn replicate 42 43 44 > runs/peso/calls.json
python replication/peso/calls.py watch runs/peso/calls.json
```

Long stages are spawned on the deployed app. A first attempt used `modal run --detach`. On 19 October a brief local network disconnect cut off the clients, and Modal cancelled all three seeds about two hours into block-0 training. A detached local entrypoint keeps only its last triggered call alive. That attempt cost about $16 and produced no results.

`replicate` trains a shared block-0 pretraining stage, then single evolving LoRA (`--shift_flag lora`, λ=0) and PESO (`lora_kldiv_latest`, λ=2.0), both reusing that block 0 via `--skip_block0`. It then runs upstream `continual_test.py` for all three. The completion record for each stage is create-only on the `cp4285-peso` Modal volume, so a finished or partial stage cannot be overwritten.

### Move to a Runpod on-demand pod — 19 October 2026

Modal GPU containers are preemptible, and Modal's preemption guide says `nonpreemptible` "is not supported for GPU Functions". Two seeds were preempted and restarted from scratch, because upstream cannot resume mid-block. Seed 42 was cancelled on Modal at the user's request, marked `CANCELLED.txt` in its run folder, and restarted on Runpod. Seeds 43 and 44 continue on Modal under the identical configuration.

- **Runpod hardware:** 1× A100-SXM4-80GB, Secure Cloud, on-demand (REST v2 creates no spot pods), $1.79/h when created.
- **Storage:** a 150 GB standard network volume ($0.07/GB/month) mounted at `/workspace`, in the same data center. Account-specific pod and volume IDs are kept in ignored `local/WORKSPACE.md`.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Our venv uses Python 3.10 with upstream's pins, as on Modal.
- **Base model:** the pinned Llama-3.2-1B revision, pushed from the Modal volume, so the pod needs no Hugging Face token. Its `model.safetensors` SHA-256, `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`, matches Hugging Face's published LFS checksum for that revision; the Modal copy matches as well. Incoming single connections to the pod ran below 1 MB/s, so the file was sent over 8 parallel SSH streams (about 10 MB/s) and verified after joining.
- **Runner:** [`replication/peso/pod_run.py`](../replication/peso/pod_run.py) runs on the pod under `setsid`, independent of this client. Training stages run one after another: in the pod smoke run, concurrent plain-LoRA and PESO training ran out of memory, because one training process holds about 48 GB of the 80 GB. The three test runs share the GPU as separate processes.
- **Difference from Modal:** this image has no `nvcc`, where the Modal image had CUDA 12.1 `nvcc`. A debug smoke run on the pod checks whether upstream's DeepSpeed path needs it.

## Code and data findings (checked, not hypotheses)

1. **Released data ≠ paper datasets.** The paper reports Instruments, Movies & TV and Books. The repository releases Instruments, CDs & Vinyl, Video Games and Yelp, and no block-construction script. Only Instruments can be compared directly with Table 2.
2. **One seed.** Every script defaults to seed 42, and Table 2 reports no spread.
3. **The PESO penalty covers only the LoRA matrices.** `compute_kl_divergence_loss` (`Fine-tuning/continual_lora.py:160–216`) uses each adapter's A and B. The trainer adds it only for modules that implement `compute_continual_loss` (`continual_train.py:34–53`). `embed_tokens` and `lm_head` are trained as full copies with no penalty. From the Llama-3.2-1B configuration this is roughly 5.6M LoRA parameters against roughly 263M per embedding/output matrix; the run logs print the exact counts.
4. **Block sizes (Instruments).**

   | Block | Users | Training targets |
   | --- | --- | --- |
   | Block 0 | 17,046 | 107,696 |
   | Block 1 | 1,772 | 9,653 |
   | Block 2 | 1,821 | 9,692 |
   | Block 3 | 2,289 | 14,233 |
   | Block 4 | 2,238 | 13,097 |

   Each block's test target is its user's last item, with the second-to-last held out for validation.
5. **No target leakage across blocks.** A block's test item is never a training target for the same user in a later block. It does reappear in later histories for returning users (1,221 of block 0's 17,046 test users). Training scores only the response tokens, so this is exposure as input, not supervision. Re-evaluating an old block's test users after later blocks is therefore a valid retention measurement under this caveat.
6. **The semantic-ID index covers future items.** All 53,509 items have IDs, including 13,038 that first appear after block 0. The tokenizer therefore had access to the text of items that arrive later.
7. **"Final model" evaluation restores each block's state.** For test block j, upstream reloads block j's saved adapter and `embed_tokens`/`lm_head`. Table 2 therefore averages each block's own test after training on that block. It contains no old-block retention.
8. **Averaging is ambiguous.** The paper reports metrics "averaged over D2, …, D4" (code blocks 1–3) but fine-tunes at stages t = 2, …, 5 (PDF p. 8). We report blocks 1–3 (the paper's statement) and 1–4.
9. **Downloads were authorized.** The user approved downloading PESO's released data in the 19 October plan, and the Llama 3.2 licence was accepted on the user's Hugging Face account. Copies live only in ignored `local/` and `data/` folders and on the run volumes. No Amazon metadata was downloaded for the replication itself. On 22 October the category extension downloaded the Amazon 2023 Musical_Instruments metadata (155 MB) with PR 14's `cp4285 meta-download`, for the title join below. It lives in ignored `data/meta/`.
10. **The released penalty is numerically negligible.** The softmax runs over every entry of a flattened LoRA matrix (4,096–65,536 entries), which makes both distributions nearly uniform. For small changes, KL ≈ ‖Δ − mean(Δ)‖²/(2n), so the matrix size divides the penalty. [`penalty_size.py`](../replication/peso/penalty_size.py) recomputes it from the seed-42 checkpoints at the end of each block, where it peaks:
    - **Value:** 0.7–1.3 × 10⁻⁶ in float32, against a training loss of about 2–3. The approximation matches within 1–3%.
    - **Gradient:** norm 1.4–1.9 × 10⁻⁶, against logged gradient norms of about 6–11.
    - **In bfloat16,** the training dtype, the computed value (about 5 × 10⁻⁴) is mostly rounding error.
    - **So:** with the same seed, PESO trains almost exactly like single LoRA, which explains the null result below. This is about the released code at `edbec5f`; the paper's own implementation may differ.
11. **Upstream evaluation samples inside beam search.** `continual_test.py` sets only `pad_token_id` and `eos_token_id` on the generation config (lines 116–117). `generate` therefore inherits the base model's `generation_config.json`: `do_sample: true`, temperature 0.6, top-p 0.9. Every Table 2 number in this replication comes from seeded beam sampling, as the released code does, rather than deterministic beam search. `peso_eval.py` reuses these settings, so it still matches upstream exactly. With item IDs, four code positions leave enough valid beams. With one-token category targets, top-p can keep fewer categories than there are beams. The extra beams then take masked tokens, and upstream's position table raises `KeyError: 2` on the first test user. A CPU reproduction with transformers 4.52.4 and a small random Llama shows that failure for one 30-token position under sampling, and none for four 256-token positions or under deterministic beam search. The category runs therefore decode deterministically (below). The [decoding check](#decoding-check--23-october-2026-seed-43-finding-11) measures its effect on seed 43: re-drawing the sampling moves single Table 2 cells by up to 12 users, and deterministic decoding scores every method higher without changing PESO ≈ single LoRA.

## Our additions

- [`peso_eval.py`](../replication/peso/peso_eval.py) evaluates any saved block state *i* on any block *j*'s test users. It reuses upstream's loader, data, collator, generation settings and metrics. It dumps each user's best-first top-10 and asserts that its ordering equals upstream `get_topk_results`. The smoke run checks that state *j* on block *j* reproduces upstream `continual_test.py`.
- [`cp4285 flips`](../src/cp4285/experiment/flips.py) reads those dumps and reports:
  - the matrix R[i][j] for NDCG@10 and Hit@10, plus backward transfer;
  - between consecutive states, how many users kept, lost, gained or never had the target in their top 10;
  - movement within the top 10, the lost-hit rate and an exact McNemar test on lost vs gained;
  - per-user forgetting events.
  
  `--groups` slices every count by user group.
- [`transfer_report.py`](../replication/peso/transfer_report.py) runs `cp4285 flips` on both methods' dumps and adds the head-to-head PESO-vs-single-LoRA flips on the same users.
- [`jlens.py`](../replication/peso/jlens.py) and [`jlens_report.py`](../replication/peso/jlens_report.py) are our Jacobian-lens probe and its summary (direction O7).
- [`penalty_size.py`](../replication/peso/penalty_size.py) measures the PESO penalty and its gradient from saved checkpoints (finding 10).
- [`jlens_items.py`](../replication/peso/jlens_items.py) and [`jlens_items_report.py`](../replication/peso/jlens_items_report.py) read all four item codes at every layer, with a permutation baseline for internal vs output change.
- [`penalty_calibrate.py`](../replication/peso/penalty_calibrate.py) compares each penalty variant's pull with the task gradient on the adapters. [`penalty_variants.py`](../replication/peso/penalty_variants.py) runs upstream training with the size-normalized penalty, and the `pod_run.py` stages `calibrate` and `variant` drive both.
- [`item_drift.py`](../replication/peso/item_drift.py) measures how far the trained item layers move per block and the O2 item-penalty weight whose pull matches the task gradient; `penalty_variants.py --item_penalty_weight` applies that penalty (`pod_run.py` method `peso-size-w<λ>-i<μ>`).
- [`variant_report.py`](../replication/peso/variant_report.py) and [`variant_jlens_compare.py`](../replication/peso/variant_jlens_compare.py) compare a variant with single LoRA and released PESO on the same users, inside and out.
- [`user_groups.py`](../replication/peso/user_groups.py) labels each block's test users as new, dormant or continuing, following the paper's Table 3 groups (PDF p. 9). For Instruments block 3 (the paper's D4) this gives 1,684 new, 329 dormant and 276 continuing users.

## Results

### Smoke check — 19 October 2026 (`smoke5-seed42`)

Upstream `--debug` samples 100 training and 100 validation examples per block; test sets stay full. Pretraining, plain LoRA and PESO (all five blocks) completed on one A100-80GB each, and module subsets were extracted. On block 1 (1,772 users), upstream `continual_test.py` and our evaluator returned identical metrics. Both were 0.0 on every metric, because a model trained on 100 examples per block recommends nothing correctly, so this shows only that the pipelines run and agree on an empty result. The non-zero check uses the full run's diagonal.

Measured on full block-0 training: 1.48 optimizer steps per second (about 47 examples per second), so the 13,464 steps take about 2.5 hours. Evaluation took about 0.18 seconds per test user (beam 10, batch 1).

### Table 2 replication — 20 October 2026, seeds 42, 43, 44

Seeds 43 and 44 ran on Modal and seed 42 on the Runpod pod, all with the identical configuration above. Each value is the mean ± sd across the three seeds, with the paper's Table 2 Instruments value in brackets. The upstream metric files and `summary-3seeds.json` are committed under [`replication/peso/results/`](../replication/peso/results/README.md), with checksums. [`notebooks/02_peso_replication.ipynb`](../notebooks/02_peso_replication.ipynb) reproduces these tables from them.

Blocks 1–3 (D2–D4, the paper's stated average):

| Method | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| Pretrain | 0.0174 ± 0.0004 (0.0166) | 0.0239 ± 0.0015 (0.0216) | 0.0116 ± 0.0006 (0.0115) | 0.0137 ± 0.0010 (0.0131) |
| Single evolving LoRA | 0.0180 ± 0.0005 (0.0181) | 0.0255 ± 0.0011 (0.0253) | 0.0123 ± 0.0006 (0.0127) | 0.0147 ± 0.0008 (0.0150) |
| PESO | 0.0177 ± 0.0011 (0.0193) | 0.0257 ± 0.0012 (0.0268) | 0.0120 ± 0.0007 (0.0138) | 0.0146 ± 0.0007 (0.0162) |

Blocks 1–4 give the same picture: NDCG@10 is 0.0132, 0.0146 and 0.0144 respectively.

Paired by seed, so that both methods share each seed's block-0 checkpoint and data, PESO minus single evolving LoRA on blocks 1–3 is:

| | Seed 42 | Seed 43 | Seed 44 | Mean (sd) | Paper |
| --- | --- | --- | --- | --- | --- |
| NDCG@10 | +0.0001 | −0.0004 | −0.0001 | −0.0001 (0.0002) | +0.0012 |
| Hit@10 | +0.0001 | +0.0004 | +0.0002 | +0.0002 (0.0001) | +0.0015 |

Against the success criterion set before running:

- **Pretrain and single evolving LoRA reproduce.** Every metric lands within about 5% of Table 2, except Pretrain's Hit@10 (+11%). PESO's Hit@10 is within 4%.
- **PESO's NDCG@10 is 10% below the paper's.** It ties single evolving LoRA instead of beating it, so the ordering Single < PESO does not reproduce. The paired difference is about a tenth of the reported gain and, for NDCG@10, indistinguishable from zero.
- **Seed-to-seed variation is as large as the claimed effect.** Single evolving LoRA's own improvement over Pretrain ranges from −0.0008 to +0.0025 NDCG@10 across seeds, so the paper's single-seed Table 2 could not separate a 0.0012 gain from seed noise.

What these numbers can and cannot support:

- **The environment is not the obvious cause.** Pretrain and single evolving LoRA match, which argues against a broken setup.
- **The documented deviations remain candidate causes.** Python 3.10 and one GPU with gradient accumulation 4 change the sampling order, though not the effective batch.
- **The paper's own data may differ.** Its exact data processing for Table 2 is not released (finding 1).
- **Per-user tests agree (seed 42).** On the transfer matrix below, PESO and single LoRA differ on only a few dozen users per block, and no state–block pair is significant under McNemar's test.
- **A mechanism is measured.** In the released code the penalty is about 10⁻⁶ (finding 10), so PESO ≈ single LoRA is expected. The paper's reported gain would need another explanation, such as seed variation or an implementation that differs from the release.

This is a statement about this replication, not a claim that PESO cannot help elsewhere.

### Evaluator validation and hardware sensitivity — 20 October 2026

Run on an RTX 4090 pod (seed 42, block 1, 1,772 users, state 1), before the transfer matrix:

| Check | Single LoRA NDCG@10 / Hit@10 | PESO NDCG@10 / Hit@10 |
| --- | --- | --- |
| Upstream `continual_test.py`, original A100 run | 0.018677 / 0.031038 | 0.019332 / 0.032731 |
| Upstream `continual_test.py`, RTX 4090 | 0.018493 / 0.030474 | 0.018876 / 0.030474 |
| Our `peso_eval.py`, RTX 4090, batch 1 | 0.018493 / 0.030474 | 0.018876 / 0.030474 |
| Our `peso_eval.py`, RTX 4090, batch 16 | 0.015317 / 0.027652 | 0.015578 / 0.028217 |

- **Our evaluator reproduces upstream exactly on the same GPU.** Its per-user dumps are a faithful view of upstream's metric.
- **Changing GPU type moves a few users.** A100 to RTX 4090 changed block-1 hits by 1 (single LoRA) and 4 (PESO) of 1,772 users, about ±0.0002–0.0005 NDCG@10. That is the same size as the paired PESO − single LoRA differences in the table above. Effects this small are at the level of hardware numerics, so cross-GPU comparisons need that caveat.
- **Batched generation is not valid here.** With left padding and batch 16, 0 of 1,772 (single LoRA) and 3 of 1,772 (PESO) top-10 lists matched batch 1, likely because upstream's constrained-decoding helper assumes unpadded input. All our evaluations therefore use batch 1, as upstream does.

### Transfer matrix and per-user flips — 20 October 2026 (seed 42)

Every saved state of single LoRA (states 0–4) and PESO (states 1–4; state 0 is the shared block-0 checkpoint) was evaluated on every block's test users with our evaluator.
- **Settings:** RTX 4090, batch 1, 20 beams so that @20 metrics exist, and at most 2,500 seeded test users per block. Only block 0 (17,046 users) is subsampled; blocks 1–4 have 1,772–2,289 users each.
- **Not Table 2's numbers:** with 20 beams and the subsample, absolute values differ from the 10-beam, all-user results above.
- **Where:** `transfer_report.py` wrote [`replication/peso/results/seed42/reports/transfer.json`](../replication/peso/results/seed42/reports/transfer.json) (counts only; the per-user dumps stay on the run volume), and notebook 02 prints it.

| | Backward transfer, NDCG@10 | Backward transfer, NDCG@20 | Last state on block 0, NDCG@10 |
| --- | --- | --- | --- |
| Single evolving LoRA | −0.0038 | −0.0053 | 0.0116 |
| PESO | −0.0061 | −0.0065 | 0.0092 |

Backward transfer is the mean over blocks 0–3 of (score after the last block − score right after training that block).

- **Neither method avoids forgetting, and PESO does not forget less.** Both lose quality on earlier blocks as training continues. On this seed PESO's backward transfer is more negative; with one seed that is not evidence that PESO forgets more.
- **The per-user counter shows heavy churn under flat averages.** At each update about 45% (single LoRA) and 47% (PESO) of earlier top-10 hits are lost, and nearly as many new hits are gained. Summed over 20 block-steps, single LoRA keeps 606, loses 496 and gains 467. Dormant and continuing users lose slightly larger shares (46–53%) than new users (45–47%), but those counts are small.
- **Head to head, PESO and single LoRA rarely disagree.** On the same state, block and users, only 11–27 of 1,772–2,500 users per pair are hits for one method but not the other. None of the 20 pairs differs significantly: the smallest exact McNemar p is 0.06, before any correction for multiple comparisons.

### Jacobian-lens probe — 20 October 2026 (seed 42, direction O7)

Our implementation of the J-lens (Gurnee et al., 2026), [`jlens.py`](../replication/peso/jlens.py):
- **The lens:** rows of W_U·J_ℓ, with J_ℓ = ∂h_L/∂h_ℓ taken at the decision point (the last prompt token) and averaged over 200 fixed test prompts. The logit lens (J = I) is the baseline.
- **The readout:** the rank of the target item's first semantic-ID code among the 256 first codes, at layers 4, 8 and 12, for 500 test users per block, on every saved state.
- **The drift split:** each state's lens rows minus block 0's are split exactly into an item-head part, (U_i − U_0)·J_i, and an inner-layer part, U_0·(J_i − J_0).
- **Where:** [`jlens_report.py`](../replication/peso/jlens_report.py) wrote [`results/seed42/reports/jlens.json`](../replication/peso/results/seed42/reports/jlens.json).

MRR after the last block, with the share of targets ranked in the top 10 of 256 in brackets (single LoRA; PESO differs by at most 0.001):

| Readout | Layer 4 | Layer 8 | Layer 12 |
| --- | --- | --- | --- |
| J-lens | 0.097 (22%) | 0.095 (23%) | 0.107 (28%) |
| Logit lens | 0.023 (4%) | 0.035 (10%) | 0.088 (20%) |
| Final output | 0.213 (43%) | | |

- **The J-lens reads the coming recommendation early.** At layer 4 it ranks the target's first code about 4× better than the logit lens, and at layer 8 nearly 3× better. That is roughly half as well as the final output.
- **Internal and output changes mostly move together.** From each block's own state to the last state (blocks 1–3), changes in mid-layer and output ranks correlate at Spearman 0.59–0.65. For about 11% of users the output rank worsened while the J-lens rank did not. That is limited evidence of internal retention beneath output forgetting.
- **The lens drifts steadily, through both parts.** By state 4 the layer-4 lens has moved 36% of its block-0 norm: an item-head part of 21% and an inner-layer part of 27% (vector parts, so the sizes do not add). At layer 12 the item head dominates (19% vs 14%).
- **PESO's drift equals single LoRA's to three decimals at every state and layer,** as finding 10 predicts.
- **Limits:** one seed and the first code token only (256 groups), which the item-level probe below extends. Workspace properties were shown on much larger models, so this is exploratory.

### Item-level J-lens — 21 October 2026 (seed 42)

[`jlens_items.py`](../replication/peso/jlens_items.py) extends the probe in two ways.
- **All four codes.** The target's codes are fed teacher-forced after the prompt, so each code is read where the model writes it. Upstream's constrained decoding allows exactly that level's 256 codes there.
- **All 15 layers before the output.** The same 200 lens prompts and 500 readout users per block are used.

Run on single LoRA states 0–4 and PESO state 4, on an RTX 4090 in about 2.8 h. [`jlens_items_report.py`](../replication/peso/jlens_items_report.py) wrote [`results/seed42/reports/jlens_items.json`](../replication/peso/results/seed42/reports/jlens_items.json).

**Checks:**
- **Level 0 reproduces `jlens.py` up to bfloat16 numerics.** J-lens ranks match exactly for 79–83% of users, and final-output ranks for 77%.
- **The state-1 drift split for code 1 reproduces the earlier run to three decimals,** although it is computed differently: as rows_i(U_i) − rows_i(U_0), not with a ΔU cotangent.

| Code (state 4, single LoRA) | Final MRR | J-lens MRR, layer 4 / 10 / 14 | Users worse, own state → state 4 | Mean rank change | Layer-4 drift: item head / inner |
| --- | --- | --- | --- | --- | --- |
| 1 (broad group) | 0.212 | 0.097 / 0.106 / 0.205 | 41% | +0.40 | 0.21 / 0.27 |
| 2 | 0.250 | 0.192 / 0.200 / 0.240 | 43% | −0.79 | 0.15 / 0.32 |
| 3 | 0.607 | 0.151 / 0.268 / 0.585 | 24% | −2.6 | 0.11 / 0.46 |
| 4 (exact item) | 0.727 | 0.108 / 0.198 / 0.681 | 17% | −4.5 | 0.10 / 0.50 |

Chance MRR is 0.024 at every level. "Worse" compares each block's own state with state 4, for blocks 1–3 (1,500 users).

- **Codes 3–4 are decided late.** Their readouts stay low until about layer 9 and then rise steeply, while codes 1–2 are flat through the middle layers.
- **Forgetting sits in the broad codes, not the exact item.** This corrects an inference we drew from the first-code probe alone.
  - Code 1 is the only code that worsens on average.
  - Given the correct earlier codes, codes 3–4 *improve* for most users.
  - Beam search keeps only the top candidates at each code, and the target's first code sits near rank 15. So a small slip at code 1 knocks items out of the top 10, which fits the 45% churn.
- **No hidden retention.** Output and J-lens item-level changes correlate (Spearman 0.26 at layer 0, rising to 0.80 at layer 14). The share of users whose output got worse while the J-lens did not is 9–18%, *below* a within-block permutation baseline of 21–25% (sd ≈ 0.6%).
- **Drift is not forgetting.** The largest drift (codes 3–4) is inner-layer and goes with improvement. The forgetting code (code 1) has the largest item-head part, at layer 14 even 0.18 against 0.10.
  - **Prediction for O8:** a working LoRA-only penalty would mainly restrain learning. Penalizing the item layers (O2) is the likelier lever on forgetting.
  - **Status:** these are correlations from one seed.
- **PESO state 4 equals single LoRA state 4** to about three decimals in every readout and drift value.

### Penalty calibration — 21 October 2026 (seed 42)

[`penalty_calibrate.py`](../replication/peso/penalty_calibrate.py) measures the next-item loss gradient on the LoRA adapters alone, on 64 block-*b* test prompts in steps of 32 at the state where block *b* starts. It compares that with each penalty variant's gradient at PESO's actual end-of-block drift. The report is [`results/seed42/reports/penalty_calibration.json`](../replication/peso/results/seed42/reports/penalty_calibration.json).

| λ = 2 | Penalty gradient ÷ task gradient (blocks 1–4) |
| --- | --- |
| Released (mean over 112 modules) | 2–3 × 10⁻⁷ |
| Paper's Eq. 12 as written (sum over matrices, ×112) | 2–3.5 × 10⁻⁵ |
| Size-normalized (KL × entries, float32) | 0.004–0.006 |

- **The adapters' task gradient is 4.6–8.2, about as large as the item layers' (4.7–7.9).** The earlier comparison against the whole-model gradient norm was therefore not misleading.
- **Even the paper's summed formula is far too weak at its largest λ (8).**
- **The size-normalized form matches the task's pull at λ ≈ 390.** The O8 run uses λ = 400 (`peso-size-w400`).
- **Direct evidence from the training logs (seed 42, block 1):** released PESO and single LoRA log the same training loss to three or four decimals (2.9075 vs 2.9075, 2.7933 vs 2.7936, 2.6767 vs 2.6769). With the size-normalized penalty at λ = 400, the logged loss, which includes the penalty, runs 0.3–0.5 higher from the first logging step, and the gradient norms rise. That penalty acts.

### O8: a penalty that acts — 21 October 2026 (seed 42)

**Setup.** PESO was retrained on blocks 1–4 from the shared block-0 checkpoint, with one change: the size-normalized penalty at λ = 400 (`pod_run.py variant --method peso-size-w400`, wrapper [`penalty_variants.py`](../replication/peso/penalty_variants.py)).
- **Training:** about 2.2 h on an A100. The publish step was recovered by hand after the network volume filled up; the completion record says so.
- **Evaluation:** our evaluator on the 8 pairs that backward transfer and new-block accuracy need, with the same users and settings as the transfer matrix above (20 beams, at most 2,500 users).
- **Reports:** [`variant_report.py`](../replication/peso/variant_report.py) and [`variant_jlens_compare.py`](../replication/peso/variant_jlens_compare.py) wrote [`o8_peso-size-w400.json`](../replication/peso/results/seed42/reports/o8_peso-size-w400.json) and [`o8_jlens_peso-size-w400.json`](../replication/peso/results/seed42/reports/o8_jlens_peso-size-w400.json).

| NDCG@10 | Single LoRA | Released PESO | PESO, λ = 400 (size-normalized) |
| --- | --- | --- | --- |
| Backward transfer (blocks 0–3) | −0.0038 | −0.0061 | **−0.0029** |
| Old blocks 0–3 after block 4 | 0.0130 | 0.0110 | **0.0140** |
| Own state, blocks 1–3 (Table 2's view) | 0.0167 | 0.0172 | 0.0169 |
| Own state, blocks 1–4 | 0.0165 | 0.0166 | 0.0165 |
| Newest block 4 | 0.0157 | 0.0146 | 0.0154 |

NDCG@20 and Hit@10 point the same way. For example, NDCG@20 backward transfer goes from −0.0053 to −0.0038, but the NDCG@20 own-state average over blocks 1–4 drops from 0.0198 to 0.0192.

- **An acting penalty reduces list-level forgetting a little.** Backward transfer improves by about a quarter, and old blocks end about 8% higher.
- **That gain is not significant on this seed.** On the old blocks after block 4, the variant gets 90 users right that single LoRA misses, against 75 the other way (exact test p = 0.28; at top-20, 114 vs 93, p = 0.16). No single state–block pair is significant.
- **It does not raise Table 2's own-state score** (0.0169 vs 0.0167). Table 2 only scores each block right after training, so it cannot see a retention effect. The paper's reported +0.0012 own-state gain is not reproduced even with an acting penalty.
- **The mechanism acts where expected.** At state 4 the inner-layer part of the J-lens drift shrinks by 26–50% across the four codes (layer 4). The item-head part is unchanged for code 1 (0.21 → 0.22) and slightly larger for codes 3–4 (0.10 → 0.13).
- **It has a learning cost on fine detail.** On the newest block, the exact-item codes rank lower: code 3 MRR 0.589 → 0.535, code 4 MRR 0.711 → 0.655. At top-20 the newest block has 28 variant-only against 44 LoRA-only hits (p = 0.08). On old blocks, codes 1–2 readouts are unchanged and codes 3–4 dip slightly.
- **Reading:** penalizing only the adapters restrains the inner drift, much of which is learning. It leaves the item-head drift tied to code 1 untouched. This is one seed and one λ.

**Next:**
- repeat single LoRA and λ = 400 on seeds 43–44, with the same matrix pairs;
- O2 (penalize the item layers) at a matched strength;
- a short λ sweep (100, 400, 1,600).

### Decoding check — 23 October 2026 (seed 43, finding 11)

The released seed-43 weights (GitHub release `llama-peso-weights-seed43-v1`, rebuilt by [`release_ckpt.py`](../replication/peso/release_ckpt.py) with every SHA-256 checked) were re-evaluated on the NUS SoC cluster's A100-80GB with our evaluator: blocks 1–3, 10 beams, batch 1, all users (`pod_run.py table2`). Each method ran twice, with the released beam sampling and with deterministic beam search. Blocks 1–3 means:

| Decoding | Pretrain Hit@10 | Pretrain NDCG@10 | LoRA Hit@10 | LoRA NDCG@10 | PESO Hit@10 | PESO NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Released sampling, committed Modal run | 0.0223 | 0.0128 | 0.0260 | 0.0152 | 0.0264 | 0.0149 |
| Released sampling, cluster re-draw | 0.0202 | 0.0114 | 0.0270 | 0.0148 | 0.0259 | 0.0144 |
| Deterministic beam search | 0.0243 | 0.0134 | 0.0285 | 0.0159 | 0.0286 | 0.0160 |

- **Two sampling draws of the same weights disagree.** Upstream seeds once and tests blocks 0–4 in turn, so each block's random stream depends on the blocks before it; our evaluator starts a fresh stream per cell. Single cells moved by up to 12 of about 1,800 users (Pretrain on block 2: 32 against 44 hits). That is larger than the 1–4 users the earlier hardware check found, and larger than most paired PESO − LoRA differences.
- **Deterministic decoding scores higher.** Against the two sampling draws it adds 0.0015–0.0041 Hit@10 and 0.0006–0.0020 NDCG@10 to every method, because sampling sometimes keeps lower-probability beams.
- **The replication's conclusion holds.** PESO − single LoRA NDCG@10 is −0.0004 under both sampling draws and +0.0001 deterministically, against the paper's +0.0012. LoRA − Pretrain is +0.0025 to +0.0034.

So the released sampling adds evaluation noise of about the size of PESO's claimed gain, and the paper's single-seed Table 2 could not separate the two. Absolute Table 2 numbers depend on the decoding, so comparisons should keep it fixed.

### Detectable effects — 23 October 2026 (seed 43, five more draws)

Five more sampling draws of the same seed-43 weights (evaluation seeds 1–5, settings as above) measure how far a Table 2 comparison moves when only the evaluation changes. [`noise_components.py`](../replication/peso/noise_components.py) splits each comparison's variance into evaluation, test-user and training-seed terms from the per-user dumps. It turns them into the smallest effect a two-sided 5% test detects with 80% power, 2.80 standard errors. The report, [`results/noise/noise-components-2026-10-23.json`](../replication/peso/results/noise/noise-components-2026-10-23.json), holds aggregates only. Blocks 1–3, all 5,882 test users, NDCG@10:

| Comparison | Evaluation, one draw (SE) | Test users (SE) | Training seeds (SE) | Smallest detectable, 1 seed sampled | 1 seed deterministic | 3 seeds sampled |
| --- | --- | --- | --- | --- | --- | --- |
| PESO − single LoRA | 0.00050 | 0.00026 | below the evaluation noise | 0.0016 | 0.0007 | 0.0009–0.0011 |
| Single LoRA − Pretrain | 0.00092 | 0.00105 | 0.0014 | 0.0047–0.0056 | 0.0039–0.0049 | 0.0027–0.0040 |

The seed terms come from the three replication seeds, each evaluated once with sampling. For PESO − single LoRA they spread by 0.00023, less than one draw's evaluation noise, so the seed part cannot be separated (95% upper bound 0.0014). One seed with per-user dumps cannot tell whether a user's difference persists across seeds, so the ranges run from none of it persisting to all of it.

- PESO − single LoRA ranges from −0.0010 to +0.0007 over the six sampling draws (mean −0.0001) and is +0.0001 deterministically.
- Swapping single LoRA for PESO changes 11 users' Hit@10 outcome under deterministic decoding (5 lost, 6 gained). Two sampling draws of the same single-LoRA weights change 111 (67 lost, 44 gained).
- Three sampled seeds could detect 0.0009–0.0011 unless seed variance is far larger than they show. The replication measured −0.0001 against the paper's +0.0012.
- Single LoRA − Pretrain carries more noise of every kind (table above).

A second deterministic product seed (the prefix run's item-only arm, seed 42, 20 beams) measures how much of a user's difference persists across seeds. The final report, [`results/noise/noise-components-2026-10-24.json`](../replication/peso/results/noise/noise-components-2026-10-24.json), adds it and category seeds 42–44:

| Comparison | Two or three seeds | Test users (SE) | Persists across seeds | Smallest detectable, 1 deterministic seed | Any number of seeds |
| --- | --- | --- | --- | --- | --- |
| Products, PESO − single LoRA, NDCG@10 | −0.0001, +0.0001 | 0.0002 | 0% | 0.0006 | 0 |
| Products, single LoRA − Pretrain, NDCG@10 | +0.0017, +0.0021 | 0.0011 | 41% | 0.0031 | 0.0020 |
| Products, forgetting (state 4 − own, blocks 0–3), NDCG@10 | −0.0044, −0.0049 | 0.0009 | 48% | 0.0026 | 0.0018 |
| Categories, PESO − single LoRA, Hit@1 | +0.0010, +0.0004, +0.0003 | 0.0007 | 0% | 0.0020 | 0 |
| Categories, single LoRA − Pretrain, Hit@1 | −0.0043, −0.0014, −0.0009 | 0.0022 | 47% | 0.0067 | 0.0042 |
| Categories, forgetting, Hit@1 | −0.0008, +0.0011, −0.0018 | 0.0021 | 58% | 0.0061 | 0.0045 |

- One deterministic seed would catch the paper's +0.0012: no user effect persists for PESO − single LoRA, and the two seeds agree within 0.0002.
- For single LoRA − Pretrain and for forgetting, about half of each user's difference persists, which sets the last column's floor on this test set. More seeds cannot lower it; more users can.
- The seed term of single LoRA − Pretrain is unsettled: 0.0003 from the two deterministic seeds, 0.0014 from the three sampled replication seeds after removing evaluation noise.
- Product forgetting is about five standard errors in each seed; category forgetting stays below its 0.005 Hit@1 detection limit.

### New products — 23 October 2026 (seed 43 and SASRec, experiment 3)

Can a model recommend products it never trained on? [`new_items.py`](../replication/peso/new_items.py) groups each product-level test target by what the scored state saw of it in training. A cold target was never a training target or in a training history; rare means one to four training targets, common five or more. It scores two deterministic PESO transfer matrices (20 beams): the released seed 43, whose own-state cells match the Table 2 run's lists exactly, and the prefix run's item-only arm, seed 42. SASRec runs with and without the category channel, three seeds each. The SASRec runs were repeated with `cp4285 blocks --dumps` to get per-user lists, and the repeats reproduce the 22 October metrics exactly. The report, [`results/new_items/new-items-2026-10-24.json`](../replication/peso/results/new_items/new-items-2026-10-24.json), holds aggregates only (the 23 October report has seed 43 alone). Top-20 hits, blocks 1–3, own states:

| Group | Targets | Single LoRA, 2 seeds | PESO, 2 seeds | SASRec, 3 seeds | SASRec + category channel, 3 seeds | Popularity |
| --- | --- | --- | --- | --- | --- | --- |
| Cold | 1,246 | 0 | 0 | 0 | 0 | 0 |
| Input only | 10 | 0 | 0 | 0 | 0 | 0 |
| Rare | 1,718 | 10–11 | 10–12 | 0 | 0 | 0 |
| Common | 2,908 | 259–261 | 256–261 | 188–190 | 168–171 | 168 |

- No model recommends a cold product: zero hits in 1,246 bound the rate below 0.3% at 95% confidence. Pretrain, with 2,300 cold targets by its own exposure, also scores none.
- PESO aims near them: its top guess shares a cold target's first code 6.8–7.2% of the time, against 0.6–3.0% for SASRec. TIGER's cold-start step, adding unseen items that share a retrieved ID's first three codes ([Rajput et al.](https://arxiv.org/abs/2305.05065), section 4.3), could reach at most 6–7 of the 1,246, because 41,263 of the 45,914 three-code prefixes hold a single product.
- Of blocks 1–3's own-state hits, state 4 keeps 54–58% of common ones for single LoRA and PESO, against 76–88% for SASRec. Rare hits fade faster (11 of 43 kept), but the counts are small.
- A category-channel seed rerun on a different CPU type (xcnf instead of xcnc) drifted to own-state Hit@10 0.0181 from 0.0190, about the size of the channel's own effect. The scored run is the one pinned to the original node type.

### Category-prefixed IDs — 24 October 2026 (the prefix run, seed 42)

Does a category token in front of the four codes help item-level prediction? [`category_prefix.py`](../replication/peso/category_prefix.py) builds `Instruments_cl_catitem`, the same blocks with each ID written as `<cat_k><a_..><b_..><c_..><d_..>`. Both arms trained with seed 42, upstream micro-batch 8 and deterministic decoding, blocks 1–4 on a full H100 NVL and block 0 on a half share, with otherwise identical settings. [`prefix_report.py`](../replication/peso/prefix_report.py) strips the category token, so a hit needs the exact item in both, and scores the arms on the same 5,882 users ([`results/prefix/prefix-seed42.json`](../replication/peso/results/prefix/prefix-seed42.json)). Blocks 1–3, own states, 20 beams:

| Method | NDCG@10, plain IDs | NDCG@10, prefixed | Prefixed − plain per user (± SE) | Hit@10 lost / gained | Top guess in the target's category, plain / prefixed |
| --- | --- | --- | --- | --- | --- |
| Pretrain | 0.0152 | 0.0139 | −0.0013 ± 0.0010 | 50 / 44 | 48% / 41% |
| Single LoRA | 0.0174 | 0.0171 | −0.0003 ± 0.0011 | 63 / 68 | 53% / 48% |
| PESO | 0.0176 | 0.0168 | −0.0008 ± 0.0011 | 67 / 66 | 53% / 47% |

No difference is significant (McNemar p = 0.61–1.0), and forgetting is unchanged (single LoRA's backward transfer −0.0048 against −0.0049 NDCG@10). The prefixed arm names the right category less often than the plain arm's top product implies, and below the history rule (57%) and the category-trained model (60%).

## Category targets from PR 14 — 22 October 2026

Item-level data is sparse: each block's density is $2\times10^{-4}$ to $9\times10^{-4}$. This extension asks how pretraining, single evolving LoRA and PESO behave when every event is named by its item's subcategory instead of the item itself. The subcategories are PR 14's standard dataset, the first two levels of the Amazon Reviews 2023 `categories` path, used unchanged at the user's request. Users, blocks, event order, upstream code and every setting above stay the same. Only the index file changes.

### From PESO items to PR 14 subcategories

PESO's `item.json` carries each item's title and category path but no `parent_asin`. [`category_index.py`](../replication/peso/category_index.py) joins it to the Amazon 2023 Musical_Instruments metadata (`cp4285 meta-download --category Musical_Instruments`, 213,593 items) on exact title and full path, then to PR 14's labels:

- All 53,509 items match. For 1,281 items several ASINs share the title and path.
- 50,023 labels are observed; each equals the first two levels of PESO's own path. PR 14's text classifier predicted the other 3,486. One item had two conflicting predictions and takes the more confident one. No item is unknown.
- Each item becomes one token `<cat_k>`, one per subcategory, 30 in all. Upstream adds the 30 tokens, decodes over them with constrained beam search and scores exact matches, so a hit means the right subcategory.

```sh
python replication/peso/category_index.py local/peso/data/Instruments_cl_ori \
    data/meta/items/Musical_Instruments.parquet "data/meta/out/item_categories/domain=Musical_Instruments" \
    local/peso/data/Instruments_cl_cat replication/peso/results/category/category_stats.json
```

The statistics are in [`results/category/category_stats.json`](../replication/peso/results/category/category_stats.json):

| Block | Users | Events | Items | Categories | Item density | Category density | Top category's share of test targets |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 17,046 | 141,788 | 40,471 | 30 | $2.06\times10^{-4}$ | 0.28 | 53.1% |
| 1 | 1,772 | 13,197 | 8,346 | 17 | $8.92\times10^{-4}$ | 0.44 | 54.5% |
| 2 | 1,821 | 13,334 | 8,325 | 17 | $8.80\times10^{-4}$ | 0.43 | 55.0% |
| 3 | 2,289 | 18,811 | 9,617 | 20 | $8.55\times10^{-4}$ | 0.41 | 49.3% |
| 4 | 2,238 | 17,573 | 9,131 | 18 | $8.60\times10^{-4}$ | 0.44 | 47.2% |

Density is events over users times distinct targets, as in PR 14. Categories are 480–1,350 times denser than items, but heavily skewed. Instrument Accessories alone is about half of every block's test targets. The effective number of test categories, $\exp(H)$ of their distribution, is 5.6, and the 10 largest cover 97.2% of test targets. Fifteen categories have fewer than 50 items (70 items in all). Most are promotion or storefront nodes such as "Prime Card Bonus". They hold 0.085% of training targets and 14 of 25,166 test targets, so they cannot move any metric by more than about 0.06 percentage points.

### Metrics and baselines

[`category_report.py`](../replication/peso/category_report.py) scores our evaluator's dumps through `cp4285.experiment.metrics`, the shared rank-based definitions: Hit@1/5/10/20, NDCG@5/10/20 and MRR@10. A target outside the 20 beams counts as a miss, which leaves every metric at $K \le 20$ exact. With 30 candidates, Hit@10 saturates near 1 and Hit@20 is trivial. Hit@1, Hit@5, NDCG@10 and MRR@10 carry the comparison.

Two baselines rank all categories for upstream's own test users. Popular ranks by the block's training-target counts. History ranks the user's 20 most recent categories by frequency, most recent first on ties, then falls back to popular. The report scores both on exactly the users in each model dump. Every model number below appears beside them, and results are also given for targets with observed labels only.

| Baseline | Block | Users | Hit@1 | Hit@5 | Hit@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popular | 0 | 17,046 | 0.5313 | 0.8557 | 0.9699 | 0.7317 | 0.6571 |
| popular | 1 | 1,772 | 0.5446 | 0.8612 | 0.9802 | 0.7444 | 0.6706 |
| popular | 2 | 1,821 | 0.5497 | 0.8561 | 0.9736 | 0.7421 | 0.6698 |
| popular | 3 | 2,289 | 0.4928 | 0.8563 | 0.9716 | 0.7165 | 0.6362 |
| popular | 4 | 2,238 | 0.4718 | 0.8633 | 0.9808 | 0.7148 | 0.6306 |
| history | 0 | 17,046 | 0.5588 | 0.8992 | 0.9773 | 0.7669 | 0.6993 |
| history | 1 | 1,772 | 0.5615 | 0.9114 | 0.9870 | 0.7765 | 0.7085 |
| history | 2 | 1,821 | 0.5870 | 0.9132 | 0.9786 | 0.7858 | 0.7234 |
| history | 3 | 2,289 | 0.5679 | 0.9144 | 0.9808 | 0.7759 | 0.7097 |
| history | 4 | 2,238 | 0.5626 | 0.9196 | 0.9875 | 0.7754 | 0.7070 |

Category scores are not comparable with the item-level numbers above: the task has 30 candidates instead of 53,509.

### Category-target PESO — 24 October 2026 (seeds 42–44)

[`category_summary.py`](../replication/peso/category_summary.py) pairs the three seeds' reports ([`results/category/summary-3seeds.json`](../replication/peso/results/category/summary-3seeds.json)). Own states, blocks 1–3, mean ± standard deviation over seeds:

| Model | Hit@1 | Hit@5 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| Pretrain | 0.5973 ± 0.0011 | 0.9270 ± 0.0016 | 0.7951 ± 0.0003 | 0.7332 ± 0.0006 |
| Single LoRA | 0.5951 ± 0.0007 | 0.9273 ± 0.0011 | 0.7946 ± 0.0006 | 0.7321 ± 0.0007 |
| PESO | 0.5957 ± 0.0004 | 0.9271 ± 0.0011 | 0.7946 ± 0.0005 | 0.7322 ± 0.0005 |
| History baseline | 0.5722 | 0.9130 | 0.7794 | 0.7139 |
| Popularity baseline | 0.5290 | 0.8579 | 0.7343 | 0.6589 |

- Every model beats both baselines, by 0.023–0.025 Hit@1 over history and 0.066–0.068 over popularity.
- Updates add nothing: single LoRA and PESO sit 0.0016–0.0022 Hit@1 below Pretrain, within the 0.0042 floor that persistent user effects set (detectable effects, above).
- PESO − single LoRA is +0.0006 Hit@1 (paired). Backward transfer is −0.0005 (single LoRA) and −0.0008 (PESO) Hit@1, −0.0007 NDCG@10 for both.
- 78–87% of the lead over the history rule comes from targets whose category is already in the user's last 20 events; with those removed, the model still ranks unseen categories above a first-order Markov rule (Hit@1 0.36–0.37 against 0.33; [`category_explore.py`](../replication/peso/category_explore.py), [`results/category/explore-3seeds.json`](../replication/peso/results/category/explore-3seeds.json)).

### Granularity ladder — 24 October 2026 (experiment 1, seeds 42–44)

The same events labelled at PR 14 levels 2, 3 and 4 (`cp4285 meta-group --level N`; 30, 75 and 299 labels; products without a label that deep take their deepest ancestor, 68 at level 3 and 2,481 at level 4), plus the products themselves (the release seed 43 and the prefix run's item-only seed 42). [`ladder.py`](../replication/peso/ladder.py) reads every rung's transfer matrices ([`results/ladder/ladder-2026-10-24.json`](../replication/peso/results/ladder/ladder-2026-10-24.json); per-level reports in `results/category/level3/` and `level4/`). NDCG@10, blocks 1–3, own states; ranges are over seeds:

| Rung | Effective labels | Shift between blocks (JS) | Single LoRA − Pretrain | Drift after one update | Drift after four updates |
| --- | --- | --- | --- | --- | --- |
| Level 2 | 5.6 | 0.005 | −0.1% | 0.0% to −0.1% | −0.1% to −0.4% |
| Level 3 | 14.6 | 0.011 | +0.2% | 0.0% to −0.2% | −0.4% to −0.7% |
| Level 4 | 62 | 0.023 | +1.0% | −0.1% to +0.2% | −1.6% to +0.9% |
| Products | 1,574 | 0.63 | +15% | −13% to −14% | −32% to −37% |

- The update gain grows smoothly: on the category levels alone it rises about 1.1 percentage points per tenfold increase in effective labels (p < 0.001, nine points).
- Forgetting has no trend across category levels (p = 0.48–0.88) and appears only at products, where the block-to-block shift also jumps 28-fold. These runs cannot separate identity from shift.
- PESO equals single LoRA at every rung (within ±0.0006 NDCG@10). Scoring observed labels only leaves every gain unchanged.
- Experiment 2 at finer levels: explorations are 17%, 30% and 54% of targets. The repeats' part of the lead over the history rule falls from 82–87% to 75–76% to 21–43%, and the model's novel-rank Hit@1 over the Markov rule grows from 1.1 to 1.6 times (`results/category/level3/explore-3seeds.json`, `level4/explore-3seeds.json`).

### Items against categories — 23 October 2026 (seed 43)

Does a model trained on sparse items predict the next category as well as one trained on categories? [`category_projection.py`](../replication/peso/category_projection.py) rewrites the item-level dumps of the released seed-43 weights (deterministic beam search, 20 beams, all users of blocks 1–3) as category lists. Each target becomes its item's category, and each ranked list becomes its items' categories in order, first occurrence kept. 0.2% of generated codes name no item and are dropped. `category_report.py` then scores them on the same users as the category-trained seed 43 ([`results/category/item-projection-seed43.json`](../replication/peso/results/category/item-projection-seed43.json)). Blocks 1–3 means:

| Model | Hit@1 | Hit@5 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| Item-trained Pretrain, projected | 0.5141 | 0.8444 | 0.6999 | 0.6481 |
| Item-trained single LoRA, projected | 0.5360 | 0.8251 | 0.6976 | 0.6534 |
| Item-trained PESO, projected | 0.5368 | 0.8247 | 0.6978 | 0.6538 |
| Category-trained single LoRA | 0.5953 | 0.9265 | 0.7950 | 0.7325 |
| History baseline | 0.5722 | 0.9130 | 0.7794 | 0.7139 |
| Popularity baseline | 0.5290 | 0.8579 | 0.7343 | 0.6589 |

An item model's top 20 items span only 3.2 categories on average (between 1 and 8), mostly the popular accessory groups. Read as a category recommender, it falls below the popularity baseline on NDCG@10 and below the history baseline on every metric. Training on the dense category target beats it by 0.06 Hit@1 and 0.10 NDCG@10. The short lists cap the projected Hit@5 and above, so Hit@1 and MRR@10 are the fairest comparison; the gap holds there too.

### Running on the NUS SoC cluster

The category runs use the SoC compute cluster's Slurm queue instead of Modal or Runpod. [`replication/peso/slurm/`](../replication/peso/slurm/) wraps `pod_run.py`, which now reads its paths, dataset and index file from `PESO_*` environment variables. Its defaults remain the Runpod layout and the released data.

- **Environment:** the cluster's NFS creates only about 7 to 20 files per second, so a venv written there takes hours. [`build_env.sbatch`](../replication/peso/slurm/build_env.sbatch) runs `pod_run.py setup` on a compute node's local disk, about 17,000 files per second, and packs the result into one tar file. Each GPU job unpacks it to local `/tmp`. Versions are upstream's pins, as above (torch 2.5.1+cu121, transformers 4.52.4, peft 0.15.2, flash-attn 2.7.4.post1, Python 3.10).
- **Base model:** the pinned revision, copied from a machine with Hugging Face access, so the cluster holds no token. Its `model.safetensors` checksum matches the one above.
- **Decoding:** deterministic beam search (finding 11). The jobs load the base model from a copy whose `generation_config.json` sets `do_sample: false` and drops temperature and top-p; the weights are hard links to the pinned file. Upstream code is unchanged, and training does not read the generation config.
- **GPUs:** flash-attn 2 needs compute capability 8.0 or higher, which rules out the cluster's T4, Titan RTX and Titan V nodes. The A100-80GB queue was long, so the smoke run used an H100 NVL (96 GB). Each training job records its GPU, and a seed's methods all run on the same card.

Commands, from the cluster workspace on a login node:

```sh
PESO_WORKSPACE=$PWD bash cp4285/replication/peso/slurm/setup.sh   # bootstrap venv, beam model, env build
sbatch --gres=gpu:h100-47:1 --export=ALL,STAGE=smoke,RUN=cat-smoke2-seed42 \
    cp4285/replication/peso/slurm/peso.sbatch
for s in 42 43 44; do bash cp4285/replication/peso/slurm/submit_seed.sh $s; done
```

Each seed is queued as four short jobs, because every stage fits the `gpu` partition's 3-hour limit while the 3-day `gpu-long` queue was full. They run `replicate` twice, then `matrix` twice. Stages resume and refuse to overwrite, so the second job of each pair only finishes what a time limit cut off.

### Smoke check — 22 October 2026 (`cat-smoke2-seed42`)

Upstream `--debug` trains on 100 examples per block. On one H100 NVL, pretraining took 186 s, single LoRA 403 s and PESO 423 s. On block 1 (1,772 users, 10 beams, batch 1), upstream `continual_test.py` and our evaluator returned identical metrics: Hit@1 0.5085, Hit@5 0.8550, Hit@10 0.9577, NDCG@5 0.6929, NDCG@10 0.7265. `category_report.py` reproduces the same five values from the dump. Even 100 examples per block bring the model close to the popularity baseline on those users (Hit@1 0.5446, NDCG@10 0.7444). Evaluation took 0.087 s per test user. The first smoke run, before the decoding change, trained normally and then failed on the first test user (finding 11).

## Reproduce in Colab (trained weights)

[`notebooks/03_peso_colab.ipynb`](../notebooks/03_peso_colab.ipynb) re-evaluates our trained **seed-43** weights with upstream's unchanged `continual_test.py` and prints recorded, reproduced and paper numbers side by side. It does not retrain.

**Requirements:**
- an A100 or L4 runtime;
- a Hugging Face token stored as the Colab secret `HF_TOKEN`, from an account that has accepted Meta's Llama 3.2 licence. It is needed only for the gated base model.

**What it does:** clones this repository first (settings `REPO` and `BRANCH`; the recorded numbers it compares against are read from `replication/peso/results/`), then clones PESO at `edbec5f`, installs the pinned torch 2.5.1 and transformers 4.52.4, downloads the pinned base model, and downloads our weights from the GitHub release of `REPO`. Every release file is checked against its SHA-256. By default it evaluates blocks 1–3 (the paper's average, about 30–60 minutes); `FULL = True` evaluates all five blocks. A closing side note plots the seed-42 J-lens probe from the committed report, with no GPU needed, and lists its implications.

**The weights:**
- **Where:** GitHub release `llama-peso-weights-seed43-v1`, named *Llama-3.2-1B-PESO-replication*: 13 assets, 9.71 GB, login-free once published.
- **Testing first:** until the notebook is verified, `REPO`, `BRANCH`, the Colab badge and the release point at a personal fork ([`randomwish/cp4285_reccomender_systems`](https://github.com/randomwish/cp4285_reccomender_systems/releases/tag/llama-peso-weights-seed43-v1), branch `feat/peso-replication`). The team repository holds the same release as a draft. Before merging, set `REPO = "Hong-yiii/cp4285_reccomender_systems"`, `BRANCH = "main"` and the badge, and publish the team release.
- **What's in it:** each method's LoRA adapters, with every block's snapshot, and its trained embedding/output-layer copies. The block-0 state is stored once and rebuilt for the other methods.
- **How it's made:** [`export_bundle.py`](../replication/peso/export_bundle.py) strips the frozen base weights; [`release_bundle.py`](../replication/peso/release_bundle.py) flattens and deduplicates; `modal_bundle.py --target github` builds it on Modal.
- **Licence:** as Llama 3.2 derivatives, the release ships Meta's `LICENSE.txt`, a `NOTICE.txt` with the required attribution, "Built with Llama", and a name beginning with "Llama".
- **Earlier copy:** a private Hugging Face repo holds the same weights in the unflattened layout (`modal_bundle.py --repo`; its location is in the ignored `local/WORKSPACE.md`).

**Testing:** [`modal_colab_test.py`](../replication/peso/modal_colab_test.py) runs the notebook file top to bottom in a clean container on a Colab-like A100-40GB, with only a Hugging Face token. That is the "another user" test.

**Fresh-user test, 20 October 2026.** The notebook ran top to bottom with no cell errors. It used a clean Python 3.12 container on an A100-40GB, cloned the fork, and downloaded the base model and the public release. Blocks 1–3, reproduced against recorded:

| Method | NDCG@10 reproduced | Recorded | Hit@10 reproduced | Recorded |
| --- | --- | --- | --- | --- |
| Pretrain | 0.0111 | 0.0128 | 0.0200 | 0.0223 |
| Single evolving LoRA | 0.0148 | 0.0152 | 0.0270 | 0.0260 |
| PESO | 0.0144 | 0.0149 | 0.0259 | 0.0264 |

- **The comparison holds.** The paired PESO − single LoRA NDCG@10 is −0.0004 both times, and single LoRA and PESO land within about 3%.
- **Pretrain does not match.** It is 13% lower on average and 24% lower on block 2, where about 8 of 1,821 users change.
- **The weights are not the cause.** On the seed-42 checkpoints, every one of the 145 frozen tensors the release leaves out equals Meta's base model. The frozen embedding and output originals also match on all 128,256 original token rows; their 1,024 added item-code rows are never used at inference, which runs on the trained copies the release ships.
- **What differs is the environment.** The test used Python 3.12, numpy 2 and the A100-40GB; the recorded run used Python 3.10, numpy 1.26.4 and the A100-80GB. Upstream's evaluation path has no version-sensitive randomness, so we attribute the gap to numerics in beam search but have not isolated it.
- **Read reproduced numbers with this spread in mind:** larger than the 1–4 users seen between GPU types on block 1.

## Directions after replication

Each experiment changes one variable against the replicated configuration. "Authors" means stated in the paper's conclusion (PDF p. 10: "drift-aware and more personalized regularization" and "integrating our approach with other parameter-efficient tuning methods"); "ours" means it follows from the findings above.

| # | Direction | Source | Single variable |
| --- | --- | --- | --- |
| A1 | Drift-aware regularization | Authors | Fixed λ vs λ scaled by measured drift between consecutive blocks |
| A2 | Personalized regularization | Authors | Uniform vs group-weighted penalty (dormant/new users) |
| A3 | Other parameter-efficient methods | Authors (prompt tuning vs LoRA discussed in App. E, PDF p. 27) | Adapter type or LoRA rank, λ fixed |
| O1 | Seed variance | Ours, finding 2 | Seeds 42/43/44 with the same configuration |
| O2 | Penalty scope | Ours, finding 3 | Penalty on LoRA only vs LoRA plus item-token embeddings/output layer; rollback of each component localizes the change |
| O3 | Retention matrix | Ours, finding 7 | Measurement only: R[i][j], backward transfer and flips |
| O4 | Scale | Ours | Backbone size (Llama-3.2-1B → 3B) or base precision (bf16 vs 4-bit), λ fixed |
| O5 | Cross-domain shift | Ours; the team question | Our Electronics → Movies & TV stream; needs our own block builder and tokenizer |
| O6 | Future-item tokenizer | Ours, finding 6 | Tokenizer built from block-0 items only vs from all items |
| O7 | Internal vs output forgetting | Ours; Jacobian lens, Gurnee et al. 2026 | Per state, the mid-layer J-lens readout of the target item's first code token vs its output rank. Drift split into unembedding change (ΔW_U·J, unpenalized) and residual-pathway change (W_U·ΔJ). Exploratory; a null result is valid |
| O8 | A penalty that acts | Ours, finding 10 | Released penalty vs the same KL scaled by each matrix's entry count n (in effect a proximal L2 term whose size no longer depends on matrix size), then a small λ sweep |

**Status, 20 October 2026:**
- **Done:** O1 (three seeds), and O3 and O7 for seed 42 (results above).
- **Priorities changed:** finding 10 shows the released penalty is too small to act. A1, A2 and O2 reweight or extend that same penalty, so they would inherit the problem.

Recommended next step: **O8**, using the existing block-0 checkpoints, so only the incremental blocks are retrained.
- **Single variable:** the penalty's normalization (released vs scaled by n), with the seed and everything else fixed.
- **Then:** O2, penalty scope, at the working strength. Roughly 98% of trainable parameters sit outside the penalty (finding 3).
- **Not yet run:** O2, O4–O6 and A1–A3. O8 has one setting (λ = 400, seed 42; results above).
