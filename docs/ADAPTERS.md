# Adapters and parameter-efficient updates for sparse sequential recommendation

Short survey for the CP4285 group project, 22 October 2026. Apart from LoRA in the exploratory PESO replication, none of these methods runs in our code, and none is a team decision. Paper figures use each paper's own protocol and are not comparable with each other or with our runs.

## Our setting

The implemented neural model ports the original SASRec (Kang & McAuley, 2018) behind the shared `Recommender` ABC (`models/sasrec.py`, `models/sasrec_learner.py`). It has hidden size 50, two blocks, a 50-item window and tied item embeddings. A reviewed item counts as implicit feedback. Adam covers every parameter. `begin_continuation()` rebuilds Adam once, and continuation supervises only the fresh final target. The code has no adapter, LoRA, replay or category input.

We counted parameters with the repo's `SASRec` class. At PESO's 53,509 Instruments items the model has 2,704,000 parameters, 98.9% of them in the item table. The two blocks hold 25,900. With the 30 subcategories as vocabulary the total is 30,050. A rank-8 LoRA on all ten block `Linear` layers would add 8,000 parameters. Two bottleneck adapters per block with $m = 8$ would add 3,432.

The PESO replication is a user-requested exploratory workstream on branch `feat/peso-replication`; paths under `replication/peso/` below refer to that branch. Its LoRA holds about 5.6M parameters, while `embed_tokens` and `lm_head` (about 263M each) train in full without penalty. The released penalty is about $10^{-6}$. A size-normalised penalty at $\lambda = 400$ moved seed-42 NDCG@10 backward transfer from −0.0038 to −0.0029 (per-user exact test $p = 0.28$), with no own-block gain. Forgetting sat in the first, broad semantic-ID code, where item-head drift was largest.

In `replication/peso/results/category/category_stats.json`, item density is $2.1\times10^{-4}$ in block 0 and about $8.7\times10^{-4}$ in blocks 1–4. Level-2 subcategory density is 0.28–0.44, about 480–1,350 times higher. Instrument Accessories holds 47–55% of each block's test targets. Ten labels cover 97% of test targets, so Hit@10 saturates near 1 (`docs/EXPERIMENT.md`). Fifteen of the 30 labels cover 70 items between them, mostly store or promotion labels such as "Prime Card Bonus". A text classifier predicted the label for 3,486 items.

Adapters, LoRA included, choose which weights may move. They add no information about rare items, and in SASRec the rare items live in the item table.

## LoRA and continual LoRA

LoRA (Hu et al., 2022) freezes $W_0 \in \mathbb{R}^{d\times k}$ and learns $W = W_0 + BA$, with $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$ and $r \ll \min(d,k)$. Each adapted matrix gains $r(d+k)$ parameters. Biderman et al. (2024) find that LoRA learns less than full fine-tuning on code and maths, and forgets less of the base model.

PESO (Yoo et al., 2026) keeps one evolving adapter with a proximal KL term towards the previous block's adapter. In its Table 8, single evolving LoRA scored 0.0150 NDCG@10, against 0.0127 and 0.0149 for full fine-tuning at two learning rates. The pretraining learning rate did worse on the small blocks than not updating at all (App. C.2). Cumulative designs freeze past adapters and add new ones. O-LoRA (Wang et al., 2023) keeps task subspaces orthogonal. InfLoRA (Liang & Li, 2024) restricts updates to an interference-free subspace. SD-LoRA (Wu et al., 2025) learns magnitudes for past adapters. All three were designed for task- or class-incremental learning in NLP and vision. LSAT (Shi et al., 2024) trains separate long- and short-term LoRAs on TALLRec. In that study, neither full retraining nor fine-tuning improved AUC significantly. On PESO's chronological split, summing all past adapters lost 26.77% NDCG@5 against single LoRA (Table 1). O-LoRA and LSAT landed within 0.0006 NDCG@10 of single LoRA (Table 9).

For SASRec, LoRA would wrap the ten block `Linear` layers with $B = 0$, so scores start unchanged at the phase transition. `begin_continuation()` would freeze the base weights and build Adam over the factors. `describe()` and the `IMPLEMENTATION` tag would record the policy. At $d = 50$, rank 8 adds 8,000 parameters to a 25,900-parameter trunk. If the item table is also frozen, rare items cannot improve. If it trains, LoRA controls about 1% of the model. CPU cost is negligible. We found no primary source that applies LoRA to an ID-based SASRec for continual updates.

PESO already uses LoRA, and the replication points away from adapter-only penalties. About 98% of trainable parameters sit outside the penalty. The acting penalty mostly restrained inner-layer drift, much of which went with learning. Uncommitted seed-42 reports in `replication/peso/results/seed42/reports/` cover an item-layer penalty (O2) and a $\lambda$ sweep. For O2, backward transfer is −0.0030 against −0.0038, and own-block NDCG@10 on blocks 1–3 is 0.0159 against 0.0167 for single LoRA. These runs are not yet written up. PEFT's TrainableTokens method trains only chosen embedding rows, which could cover the 1,024 item-code tokens. Its documentation warns that DeepSpeed support may be incomplete.

## Bottleneck adapters

Houlsby et al. (2019) insert two bottleneck modules per Transformer layer, after attention and after the feed-forward sublayer. Each computes $h + W_{\uparrow}\,\sigma(W_{\downarrow}h)$ with $m \ll d$, adds $2md + d + m$ parameters and starts near the identity. Only adapters, layer norms and the task head train. On GLUE this came within 0.4% of full fine-tuning with 3.6% extra parameters per task. Pfeiffer et al. (2021) found one adapter after the feed-forward layer on par with two.

In `SASRecBlock.forward`, the insertion points are `attended` before the `+ queries` residual and the `forward_out` output before the `+ normalized` residual. Zero-initialised up-projections keep the transition exact. The learner change mirrors LoRA's. Per-domain adapters could isolate domain B in the two-domain proposal of `CONTINUED-TRAINING.md`. They leave the item table untouched. Evidence is strong in NLP, and recommender evidence comes from the transfer studies.

## Adapter transfer in sequential recommenders

PeterRec (Yuan et al., 2020) freezes a NextItNet CNN pre-trained on source-domain sequences. Each target task trains bottleneck "model patches" and a new output layer. On Tencent cold-user tasks, scored against 99 sampled negatives, it matched full fine-tuning (HR@5 0.351 against 0.352 on ColdRec-1) with 9.4% as many tuned parameters (Table 3). Its cold users have long source-domain histories, which our stream lacks.

Fu et al. (2024) benchmark adapters for TransRec-style models, which encode items from text or images (J. Wang et al., 2022). With SASRec over BERT item encoders, Houlsby adapters reached HR@10 32.52 against 32.83 for full fine-tuning, training 2.23% of the parameters (Table 2). LoRA reached 30.89 and prompt tuning 20.42 (Table 3). Adapters in the user encoder alone reached 3.17 (Table 4), so the useful adaptation was on the item side. UniSRec (Hou et al., 2022) freezes a text encoder and its sequence encoder and tunes a mixture-of-experts adaptor. On its Amazon Instruments split with full ranking, the text-plus-ID variant raised Recall@10 from SASRec's 0.1118 to 0.1267. FDSA kept the best NDCG@10 (0.0796 against 0.0748, Table 3). These methods help cold start because items carry content. Our ID-only SASRec would first need item text from the metadata pipeline.

## Prompt and prefix tuning

Prompt tuning (Lester et al., 2021) trains soft input tokens on a frozen model and matches full tuning only beyond billions of parameters. Prefix tuning (Li & Liang, 2021) trains a continuous prefix of about 0.1% of the parameters and beat fine-tuning with little data. L2P (Z. Wang et al., 2022) retrieves prompts from a learned pool for rehearsal-free continual learning. PPR (Wu et al., IEEE TKDE) builds each user's prefix from profile attributes. Its light variant freezes SASRec and tunes under 1% of parameters for cold-start users.

Our Amazon files hold only `user_id`, `parent_asin`, `rating` and `timestamp`, so PPR has no input here. Prompts would also take positions from SASRec's fixed 50-slot window. PESO's App. E argues that prompt pools add retrieval at every beam-search step. Fu et al. found prompt tuning far below adapters. We expect little gain for sparsity.

## LLM recommenders with LoRA and semantic IDs

TALLRec (Bao et al., 2023) tunes LLaMA-7B with LoRA on yes/no preference prompts. With 64 training samples on movies it reached AUC 0.6748 against SASRec's 0.5048 (Table 3). That task scores one given item, so it says little about top-$K$ ranking. TIGER (Rajput et al., 2023) trains a 13M-parameter encoder-decoder to generate RQ-VAE semantic IDs built from item text. Similar items share code prefixes, so rare items borrow statistics. Semantic IDs beat random IDs (Beauty Recall@10 0.0648 against 0.0434, Table 2), and the model could retrieve items held out of training (Fig. 5). PESO combines both lines. For SASRec, a category channel shares item structure more cheaply.

## Category-aware sequential models

FDSA (Zhang et al., 2019) runs separate self-attention over item and feature sequences. S3-Rec (Zhou et al., 2020) pre-trains on attribute–item mutual information and gains most with limited data. DIF-SR (Xie et al., 2022) moves side information from the input into attention. CaFe (Li et al., 2022) targets sparsity directly. Two SASRec-style encoders read the category ("intent") sequence and the item sequence. Their outputs are summed. Training minimises $L = L_c + L_v$, and ranking uses $P(c)\,P(v)$. On Amazon data with density $2\times10^{-5}$, CaFe reports NDCG@5 0.3733 against 0.1418 for SASRec and 0.0670 for FDSA (Table 2). Its gain over BERT4Rec exceeded 100% without an item-frequency filter and fell below 10% when items needed 20 interactions (Fig. 4). These figures use 100 popularity-sampled negatives, and sampled metrics can reorder models (Krichene & Rendle, 2020).

CaFe keeps item-level targets and borrows density from categories. Our category-only run turns the task into a 30-way prediction where one label takes about half the targets. `category_report.py` already scores it against popularity and history baselines. In the learner, an item→category map can live inside the model, so `train()` and `score()` keep their signatures. The cost is a 31-row table and a second small encoder, roughly twice SASRec's compute. A PESO analogue, a category token before the four codes, is untested.

## Replay and regularisation baselines

ADER (Mi et al., 2020) fine-tunes SASRec each cycle with replayed exemplars and distillation towards the previous model. Each item gets exemplars in proportion to its frequency, chosen by herding. On DIGINETICA over 16 weekly cycles, Recall@20 was 47.28% for plain fine-tuning, 47.66% with EWC (Kirkpatrick et al., 2017), 50.03% for joint retraining and 50.21% for ADER (Table 2). The authors attribute the small gaps to old items recurring. In our protocol, replay is a separately labelled condition supplied by the experiment layer. `fork()` can provide the frozen teacher, and `score()` the full-catalogue logits. Frequency-proportional exemplars would amplify the category skew. PESO's Table 8 adds a cumulative-retraining baseline at 0.0141 NDCG@10 against 0.0150 for single LoRA.

## Comparison

| Method | Frozen | Trained | Fits SASRec ABC? | Fits PESO? | Cost | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Single evolving LoRA | Base weights | $A$, $B$; PESO also item layers | Yes; about 1% of the model if items train | Native | CPU-trivial; about 2.2 A100-hours per PESO variant | Strong in NLP; reproduced on PESO data |
| PESO proximal term | Previous adapter as anchor | As LoRA | Yes, as an extra loss term | Native; released term about $10^{-6}$ | As LoRA | Paper gain not reproduced; acting penalty small, n.s. |
| Cumulative or multi-LoRA (O-LoRA, InfLoRA, SD-LoRA, LSAT) | Past adapters | New adapter per block | Yes; state grows | Yes | Storage grows per block | Built for task-incremental NLP and vision; PESO Tables 1 and 9 show losses or ±0.0006 |
| Houlsby or Pfeiffer adapters | Trunk; item table optional | Bottlenecks, layer norms | Yes; 3,432 parameters at $m = 8$ | Needs its own penalty code | CPU-trivial | Strong in NLP; best PEFT for text TransRec |
| PeterRec patches | Pre-trained CNN | Patches, output layer | Cross-domain variant only | No | Low | One study; sampled metrics |
| UniSRec adaptor | Text and sequence encoders | MoE adaptor (+ ID embeddings) | Needs item text | Text already enters via RQ-VAE IDs | Medium | Full ranking; Instruments Recall@10 +13% |
| Prompt or prefix tuning (PPR, L2P) | Backbone | Prompts or a generator | Awkward; PPR needs profiles | Library support; pool overhead | Low | Weak: HR@10 20.42 against 32.52 |
| TALLRec-style LLM with LoRA | LLM | LoRA | No | PESO's own family | High | Few-shot AUC only; LSAT saw no significant update gain |
| TIGER semantic IDs | Nothing | Seq2seq model, RQ-VAE | No | Already used | Medium to high | Semantic beat random IDs; cold-start retrieval |
| CaFe-style category channel | Nothing | Item and category encoders | Yes, with an item→category map | Category token; untested | About 2× SASRec on CPU | Large sparse-data gains under sampled metrics |
| ADER replay and distillation | Nothing | Full model | Yes, as a labelled condition; teacher via `fork()` | Needs rebuilt block files | Buffer plus teacher pass | Plain fine-tuning close behind; EWC marginal |

## Recommended next experiment

Agent proposal (not a team decision).

Implemented on 22 October as `cp4285 blocks` (see the [runbook](IMPLEMENTATION.md#peso-block-sasrec-study)) and run on the NUS SoC cluster's CPU nodes, three seeds per arm. The continuation learning rate $10^{-4}$ won on block-1 validation Hit@10 (0.0192, against 0.0181 at $3\times10^{-4}$ and 0.0164 at $10^{-3}$). Own-block scores, blocks 1–3, mean ± sd across seeds:

| Arm | Hit@1 | Hit@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| SASRec | 0.0037 ± 0.0010 | 0.0212 ± 0.0016 | 0.0108 ± 0.0013 | 0.0077 ± 0.0013 |
| SASRec + category channel | 0.0035 ± 0.0009 | 0.0202 ± 0.0012 | 0.0106 ± 0.0010 | 0.0077 ± 0.0010 |
| Popularity | 0.0033 | 0.0191 | 0.0097 | 0.0069 |
| Channel − SASRec, paired by seed | −0.0001 | −0.0010 ± 0.0019 | −0.0002 ± 0.0009 | 0.0000 |

The channel does not improve top-10 ranking, and both arms sit just above popularity. Backward transfer is about zero for both ($|\Delta| \le 0.0006$ on every metric). The channel does lift rare targets, as intended. Uncut MRR rises from $3.1\times10^{-5}$ to $9.4\times10^{-5}$ for targets never trained on and from $8.9\times10^{-5}$ to $3.4\times10^{-4}$ for targets trained one to four times, in every seed. Those ranks remain in the thousands, so Hit@10 for both groups stays at zero. A level-2 category cannot separate its items: Instrument Accessories alone holds 25,949 of the 53,509. The channel also costs frequent targets a little (Hit@10 0.0452 → 0.0430). A finer category level is the obvious next variable. This work used PR 14's level 2 as its standard, at the user's choice.

Test whether a dense category channel improves item-level ranking and retention for a continually updated SASRec. Use PESO's Instruments blocks and the subcategory labels already joined by `category_index.py`. Both are local, so nothing new is downloaded. The upstream repository has no licence, so the files stay in ignored folders, as in the replication. Keep PESO's users, block order and leave-one-out targets.

Compare the implemented SASRec with the same model plus a CaFe-style category channel. The channel is the only variable. Train on block 0 and select on validation Hit@10 with NDCG@10 as tie-break, the pilot's rule pending team confirmation. Then continue on blocks 1–4 with fresh targets and one Adam reset. Choose one continuation learning rate on block-1 validation for both arms. The pilot default of $0.1\times$ the initial rate is in line with PESO's App. C.2. Run three seeds.

Report full-catalogue Hit@10, the team's reported metric, with NDCG@10 and MRR@10 from `experiment/metrics.py`. Add the transfer matrix $R[i][j]$, backward transfer and a popularity baseline. Split targets by block-0 frequency (unseen, 1–4, 5 or more), since CaFe's claim concerns infrequent items. Report cold-start targets separately. The familiar-item policy drops them, and 13,038 PESO items first appear after block 0. Repeat the scores on observed-label targets, and settle the 15 tiny labels before running.

Each arm has about 2.7M parameters and should fit on CPU; run time is unmeasured. The new code is a model variant with its own `IMPLEMENTATION` tag, an item→category map and a reader for PESO's block files. Results go to new run directories. A null result is informative. This experiment does not test adapters. A frozen-trunk adapter arm can follow, but the trunk holds only about 1% of SASRec's parameters.

## Sources

Each source was checked online in this session. Quoted figures come from the named table, figure or appendix.

- Bao, K., Zhang, J., Zhang, Y., Wang, W., Feng, F., He, X. (2023). TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. RecSys 2023. https://arxiv.org/abs/2305.00447, https://doi.org/10.1145/3604915.3608857
- Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., et al. (2024). LoRA Learns Less and Forgets Less. TMLR, 2024. https://arxiv.org/abs/2405.09673
- Fu, J., Yuan, F., Song, Y., Yuan, Z., Cheng, M., Cheng, S., Zhang, J., Wang, J., Pan, Y. (2024). Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights. WSDM 2024. https://arxiv.org/abs/2305.15036, https://doi.org/10.1145/3616855.3635805
- Hou, Y., Mu, S., Zhao, W. X., Li, Y., Ding, B., Wen, J.-R. (2022). Towards Universal Sequence Representation Learning for Recommender Systems (UniSRec). KDD 2022. https://arxiv.org/abs/2206.05941
- Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S. (2019). Parameter-Efficient Transfer Learning for NLP. ICML 2019, PMLR 97:2790–2799. https://arxiv.org/abs/1902.00751
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. https://arxiv.org/abs/2106.09685
- Kang, W.-C., McAuley, J. (2018). Self-Attentive Sequential Recommendation. ICDM 2018. https://arxiv.org/abs/1808.09781
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks (EWC). PNAS. https://arxiv.org/abs/1612.00796, https://doi.org/10.1073/pnas.1611835114
- Krichene, W., Rendle, S. (2020). On Sampled Metrics for Item Recommendation. KDD 2020. https://doi.org/10.1145/3394486.3403226
- Lester, B., Al-Rfou, R., Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP 2021. https://arxiv.org/abs/2104.08691
- Li, J., Zhao, T., Li, J., Chan, J., Faloutsos, C., Karypis, G., Pantel, S.-M., McAuley, J. (2022). Coarse-to-Fine Sparse Sequential Recommendation (CaFe). SIGIR 2022. https://arxiv.org/abs/2204.01839
- Li, X. L., Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL-IJCNLP 2021. https://arxiv.org/abs/2101.00190, https://doi.org/10.18653/v1/2021.acl-long.353
- Liang, Y.-S., Li, W.-J. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning. CVPR 2024. https://arxiv.org/abs/2404.00228
- Mi, F., Lin, X., Faltings, B. (2020). ADER: Adaptively Distilled Exemplar Replay Towards Continual Learning for Session-based Recommendation. RecSys 2020. https://arxiv.org/abs/2007.12000
- Pfeiffer, J., Kamath, A., Rücklé, A., Cho, K., Gurevych, I. (2021). AdapterFusion: Non-Destructive Task Composition for Transfer Learning. EACL 2021. https://arxiv.org/abs/2005.00247
- Rajput, S., Mehta, N., Singh, A., Keshavan, R. H., Vu, T., Heldt, L., et al. (2023). Recommender Systems with Generative Retrieval (TIGER). NeurIPS 2023. https://arxiv.org/abs/2305.05065
- Shi, T., Zhang, Y., Xu, Z., Chen, C., Feng, F., He, X., Tian, Q. (2024). Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems (LSAT). CIKM 2024, short paper. https://arxiv.org/abs/2312.15599, https://doi.org/10.1145/3627673.3679922
- Wang, J., Yuan, F., Cheng, M., Jose, J. M., Yu, C., Kong, B., et al. (2022). TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback. arXiv preprint. https://arxiv.org/abs/2206.06190
- Wang, X., Chen, T., Ge, Q., Xia, H., Bao, R., Zheng, R., et al. (2023). Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA). Findings of EMNLP 2023. https://arxiv.org/abs/2310.14152
- Wang, Z., Zhang, Z., Lee, C.-Y., Zhang, H., Sun, R., Ren, X., et al. (2022). Learning to Prompt for Continual Learning (L2P). CVPR 2022. https://arxiv.org/abs/2112.08654
- Wu, Y., Piao, H., Huang, L.-K., Wang, R., Li, W., Pfister, H., et al. (2025). SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning. arXiv preprint. https://arxiv.org/abs/2501.13198
- Wu, Y., Xie, R., Zhu, Y., Zhuang, F., Zhang, X., Lin, L., He, Q. Personalized Prompt for Sequential Recommendation (PPR). Accepted by IEEE TKDE; arXiv 2022. https://arxiv.org/abs/2205.09666
- Xie, Y., Zhou, P., Kim, S. (2022). Decoupled Side Information Fusion for Sequential Recommendation (DIF-SR). SIGIR 2022. https://arxiv.org/abs/2204.11046
- Yoo, H., Li, T.-W., Kang, S., Liu, Z., Xu, C., Qi, Q., Tong, H. (2026). Continual Low-Rank Adapters for LLM-based Generative Recommender Systems (PESO). ICLR 2026. https://arxiv.org/abs/2510.25093
- Yuan, F., He, X., Karatzoglou, A., Zhang, L. (2020). Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation (PeterRec). SIGIR 2020. https://arxiv.org/abs/2001.04253, https://doi.org/10.1145/3397271.3401156
- Zhang, T., Zhao, P., Liu, Y., Sheng, V. S., Xu, J., Wang, D., Liu, G., Zhou, X. (2019). Feature-level Deeper Self-Attention Network for Sequential Recommendation (FDSA). IJCAI 2019. https://doi.org/10.24963/ijcai.2019/600
- Zhou, K., Wang, H., Zhao, W. X., Zhu, Y., Wang, S., Zhang, F., Wang, Z., Wen, J.-R. (2020). S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. CIKM 2020. https://arxiv.org/abs/2008.07873, https://doi.org/10.1145/3340531.3411954
- Hugging Face PEFT documentation: method overview and TrainableTokens. https://huggingface.co/docs/peft/methods/overview, https://huggingface.co/docs/peft/package_reference/trainable_tokens

Local evidence, read only: `src/cp4285/models/sasrec.py`, `src/cp4285/models/sasrec_learner.py`, `src/cp4285/experiment/model.py`, `src/cp4285/experiment/data.py`, `src/cp4285/experiment/metrics.py`, `configs/pilot.toml`, `docs/CONTINUED-TRAINING.md` and `docs/EXPERIMENT.md`; in the PESO checkout, `docs/PESO-REPLICATION.md`, `replication/peso/category_index.py`, `replication/peso/category_report.py`, `replication/peso/results/category/category_stats.json` and `replication/peso/results/seed42/reports/o2_peso-size-w400-i2.5.json`.
