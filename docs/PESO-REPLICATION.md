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
9. **Downloads were authorized.** The user approved downloading PESO's released data in the 19 October plan, and the Llama 3.2 licence was accepted on the user's Hugging Face account. Copies live only in ignored `local/` and `data/` folders and on the run volumes. No Amazon metadata was downloaded.

## Our additions

- [`peso_eval.py`](../replication/peso/peso_eval.py) evaluates any saved block state *i* on any block *j*'s test users. It reuses upstream's loader, data, collator, generation settings and metrics. It dumps each user's best-first top-10 and asserts that its ordering equals upstream `get_topk_results`. The smoke run checks that state *j* on block *j* reproduces upstream `continual_test.py`.
- [`cp4285 flips`](../src/cp4285/experiment/flips.py) reads those dumps and reports:
  - the matrix R[i][j] for NDCG@10 and Hit@10, plus backward transfer;
  - between consecutive states, how many users kept, lost, gained or never had the target in their top 10;
  - movement within the top 10, the lost-hit rate and an exact McNemar test on lost vs gained;
  - per-user forgetting events.
  
  `--groups` slices every count by user group.
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
- **No per-user test has been run yet.** A McNemar test on the kept/lost/gained counts needs the retention-matrix dumps (O3), which have not been run.

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

## Reproduce in Colab (trained weights)

[`notebooks/03_peso_colab.ipynb`](../notebooks/03_peso_colab.ipynb) re-evaluates our trained **seed-43** weights with upstream's unchanged `continual_test.py` and prints recorded, reproduced and paper numbers side by side. It does not retrain.

**Requirements:**
- an A100 or L4 runtime;
- a Hugging Face token stored as the Colab secret `HF_TOKEN`, from an account that has accepted Meta's Llama 3.2 licence. It is needed only for the gated base model.

**What it does:** clones this repository first (settings `REPO` and `BRANCH`; the recorded numbers it compares against are read from `replication/peso/results/`), then clones PESO at `edbec5f`, installs the pinned torch 2.5.1 and transformers 4.52.4, downloads the pinned base model, and downloads our weights from the GitHub release of `REPO`. Every release file is checked against its SHA-256. By default it evaluates blocks 1–3 (the paper's average, about 30–60 minutes); `FULL = True` evaluates all five blocks.

**The weights:**
- **Where:** GitHub release `llama-peso-weights-seed43-v1`, named *Llama-3.2-1B-PESO-replication*: 13 assets, 9.71 GB, login-free once published.
- **Testing first:** until the notebook is verified, `REPO`, `BRANCH`, the Colab badge and the release point at a personal fork ([`randomwish/cp4285_reccomender_systems`](https://github.com/randomwish/cp4285_reccomender_systems/releases/tag/llama-peso-weights-seed43-v1), branch `feat/peso-replication`). The team repository holds the same release as a draft. Before merging, set `REPO = "Hong-yiii/cp4285_reccomender_systems"`, `BRANCH = "main"` and the badge, and publish the team release.
- **What's in it:** each method's LoRA adapters, with every block's snapshot, and its trained embedding/output-layer copies. The block-0 state is stored once and rebuilt for the other methods.
- **How it's made:** [`export_bundle.py`](../replication/peso/export_bundle.py) strips the frozen base weights; [`release_bundle.py`](../replication/peso/release_bundle.py) flattens and deduplicates; `modal_bundle.py --target github` builds it on Modal.
- **Licence:** as Llama 3.2 derivatives, the release ships Meta's `LICENSE.txt`, a `NOTICE.txt` with the required attribution, "Built with Llama", and a name beginning with "Llama".
- **Earlier copy:** a private Hugging Face repo holds the same weights in the unflattened layout (`modal_bundle.py --repo`; its location is in the ignored `local/WORKSPACE.md`).

**Testing:** [`modal_colab_test.py`](../replication/peso/modal_colab_test.py) runs the notebook file top to bottom in a clean container on a Colab-like A100-40GB, with only a Hugging Face token. That is the "another user" test.

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

Recommended next steps, given the result: first O3, the retention matrix and per-user flips for the existing seed-42 checkpoints. It shows *where* PESO and single LoRA differ, if anywhere, at no extra training cost. Then O2, penalty scope: with roughly 98% of trainable parameters outside the penalty, a near-zero effect is what finding 3 would predict. None of these directions has been run.
