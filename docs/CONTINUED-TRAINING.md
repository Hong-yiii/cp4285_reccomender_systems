# Continued training for SASRec: proposed protocol

Draft v0.1, 27 September 2026. Scope: Hongyi's neural-model contribution. Update, 28 September: the original SASRec base architecture is now selected and ported to PyTorch; see [SASREC.md](SASREC.md). The A-only implementation uses sequence-wise initial training and fresh-target-only continuation. This document's broader two-domain protocol, dataset choices and final training budgets remain proposals; no real-data result is claimed.

## Research statement

> We extend a SASRec-based next-item recommender with periodic continued training. After initial offline training on domain A, the model is updated from its preceding checkpoint using incoming interaction batches containing controlled proportions of domains A and B. We measure retention on a fixed held-out A task and adaptation to B under a matched update budget.

Shopping and movies are illustrative domains. The contribution is an experimental study and training protocol, not a claim to invent continual learning or a new loss function.

Three choices are distinct:

1. **Continued training:** carry model weights forward and optimize on subsequent batches.
2. **Distribution change:** control the fraction of second-domain examples in those batches.
3. **Domain compatibility:** define identifiers, embeddings, scoring interfaces and shared parameters so both domains are valid inputs to one model.

Only the first is required to make SASRec a continually updated learner. The second supplies our experimental treatment; the third makes a two-domain experiment meaningful.

## Model and training examples

Write all trainable parameters as `Theta`. For the previously proposed two-domain design, `Theta = (theta, phi_A, phi_B)`: `theta` contains the shared causal Transformer and positional parameters, while `phi_d` contains domain-specific item embeddings and scoring parameters. Tie input/output item embeddings within each domain as a proposed default. A known domain selects its scoring interface and candidate set.

An example is `(d, h, i_plus, i_minus)`: domain, observed item-history prefix, observed next item, and a negative sampled from that domain's eligible training catalogue. Negative examples are unobserved alternatives, not confirmed dislikes. Histories preserve real user/session boundaries and order. Domain namespaces prevent accidental ID collisions; unrelated users are never linked merely because numeric IDs match.

Reuse SASRec's sampled binary cross-entropy next-item objective:

`ell_d(Theta) = -log sigmoid(s_Theta(d,h,i_plus)) - log sigmoid(-s_Theta(d,h,i_minus))`.

An implementation may compute many causally masked next-item positions in parallel. Each valid target position counts as one supervised example for loss normalization and mixture accounting. Padding positions contribute neither examples nor loss.

SASRec supplies this prediction task and basic loss; the schedule below is our proposed extension. [Original SASRec, §III-E](https://arxiv.org/html/1808.09781v1)

## Data contract

| Partition | Purpose | Use during updates |
| --- | --- | --- |
| `D_A_init` | Historical data for initial A training | Not reused in the basic continued-training condition |
| `S_A`, `S_B` | Subsequent domain-specific training streams | Consumed in per-domain chronological order |
| `V_A`, `V_B` | Development data for hyperparameter selection | Never used for gradient updates |
| `R_A` | Fixed A retention test representing the original task | Evaluation only |
| `R_B` | Fixed B test for new-domain recommendation quality | Evaluation only |

Prefer complete held-out sessions/sequences for the retention anchor when feasible. Otherwise enforce event-level exclusion so held-out test events do not leak into training targets or prefixes. The same item ID may appear in different partitions; that is not itself leakage. Keep history construction and item-eligibility rules explicit. Select a familiar-item A retention cohort and separately record any excluded cold-start cases.

The update streams use fresh training observations after the initial A period. Explicitly reintroducing examples from `D_A_init` is replay and belongs in a separately labeled condition. No test interactions enter either the update stream or a replay buffer.

For the first controlled experiment, a cycle is a fixed block of optimizer steps, not necessarily a simulated calendar day. Preserve chronological order within each domain. If combining datasets from unrelated periods, state that the interleaving is synthetic. Different mixtures consume different amounts of each stream; they need not reach matching calendar dates. A calendar-matched study would be a separate design.

## Stage 1: initial offline training

Train the A model on `D_A_init`, choose the initial checkpoint using `V_A`, and save `Theta_0`. Initialize any B-only parameters identically across experimental branches, using only the permitted B training catalogue. Adding the B interface must not alter A scores. There are no B gradient updates in Stage 1.

Evaluate the initial A checkpoint to obtain `Q_A_initial = Hit@10(Theta_0; R_A)`.

## Stage 2: periodic continued updates

Let `alpha` be the fraction of supervised next-item examples belonging to B. Proposed pilot values are `0`, `0.5`, and `1`. Keep alpha constant within each run. Every run independently starts from `Theta_0`, including the same B initialization.

Choose batch size `m`, updates per cycle `U`, and number of cycles `T` after checking available training examples and compute. Use the same values across mixture conditions. Stop all conditions at a common feasible budget rather than silently recycling an exhausted stream.

At cycle `t`, the incoming batch distribution is:

`P_t^alpha = (1-alpha) P_A,t + alpha P_B,t`.

Under equal weighting per supervised example, its expected objective is:

`L_t^alpha(Theta) = (1-alpha) E_A,t[ell_A(Theta)] + alpha E_B,t[ell_B(Theta)]`.

This is a sampling-weighted next-item loss, not an additional forgetting penalty. If the sampled minibatch already has the intended mixture, averaging its example losses implements that weighting; do not multiply each example by the mixture coefficient again. For a domain absent from a batch, omit its loss term rather than averaging an empty tensor. Record realized proportions if integer batch sizes require rounding.

The update rule including optimizer state `o` is:

`(Theta_t, o_t) = Update_U(Theta_(t-1), o_(t-1), incoming_batches_t)`.

**Proposed optimizer policy:** at the Stage 1 → Stage 2 transition, initialize a fresh Adam optimizer identically for all branches, retaining the learned model weights. Then preserve its state across every batch and cycle within a branch. This is still continued training: the weights are warm-started. Carrying Stage 1 moments instead would be a valid alternative, but must be applied consistently.

Select the Stage 2 learning rate on development data, fix its schedule across mixture runs, and avoid restarting early stopping or selecting the best test checkpoint separately in each cycle.

```text
initial_weights = train_A_and_select_on_validation()
initial_weights = attach_consistently_initialized_B_interface(initial_weights)

evaluate_frozen_control(initial_weights)

for alpha in [0, 0.5, 1]:
    model = independent_copy(initial_weights)
    optimizer = fresh_adam_for_this_run(model)
    streams = reset_training_stream_cursors()
    evaluate_fixed_tests(model, cycle=0)

    for cycle in 1..T:
        model.train()
        for step in 1..U:
            batch = next_controlled_mixture(streams, alpha, m)
            loss = mean_next_item_loss_over_valid_targets(model, batch)
            update_active_parameters(model, optimizer, loss)

        model.eval()
        evaluate_fixed_tests_without_gradients(model, cycle)
        save_weights_optimizer_and_stream_position()
```

The pseudocode specifies behavior; it is not executable training code. B batches update shared and B-specific parameters; A batches update shared and A-specific parameters. In a domain-absent batch, inactive domain parameters must receive neither weight decay nor stale-momentum updates. Implement this explicitly with excluded/None gradients or suitable parameter groups, not merely a zero-valued loss.

The model is not reinitialized each cycle and is not retrained on the entire accumulated history. No live serving system is needed to execute this offline simulation.

## Evaluation and comparison

At every checkpoint, evaluate the same A histories, relevance targets, candidate IDs and filtering rules in deterministic evaluation mode. Use full A-catalogue ranking if feasible; otherwise fix sampled candidates in advance. B items never enter A's candidate pool in this experiment. Treat held-out test curves as reported outcomes, not feedback for hyperparameter tuning.

| Condition | Updates | Interpretation |
| --- | --- | --- |
| Frozen | None | Evaluation stability and initial-retention reference |
| A-only | `alpha=0` | Continued learning within the original domain |
| Mixed | `alpha=0.5` initially | Continued training with competing domain exposure |
| B-only | `alpha=1` | Sequential switch away from A supervision |

Report both absolute domain scores and:

- `RetentionLoss_A(alpha,t) = Q_A_initial - Q_A(alpha,t)`. Positive means worse than the initial checkpoint; negative means improvement.
- `MixtureEffect_A(alpha,t) = Q_A(alpha,t) - Q_A(0,t)`. Negative means worse than A-only continued training at the same total update budget. This is distinct from losing initial capability.

Report Hit@10 for the next item and Recall@10 for the next items (team decision, 28 September, issue #8). For one relevant next-item target, Recall@10 equals Hit@10. NDCG@10 and MRR@10 are reported alongside; see [EXPERIMENT.md](EXPERIMENT.md#ranking-metrics). Report B quality alongside A so that preserving A by failing to learn B is visible. Compare the initial frozen model with itself over the same test set as a basic evaluator check, then repeat trained comparisons across seeds.

These curves measure retention and adaptation under the stated protocol. They do not guarantee catastrophic forgetting, neural superiority, or malicious behavior in B.

## What the first comparison can establish

With fixed total updates, more B means fewer A examples. The primary comparison measures the practical effect of changing stream composition. To investigate harmful B updates beyond reduced A exposure, add a secondary branch using the mixed run's exact A examples while skipping B updates. That control matches A exposure but intentionally uses less compute; scheduler and optimizer-time handling must be documented.

Randomly initialized B embeddings introduce cold-start learning as well as domain shift. Record this limitation and initialization seeds. Separate domain interfaces also make the request domain known; findings do not automatically transfer to a unified catalogue or unknown-domain inference.

Stored interaction logs cannot reproduce the closed-loop effects of model recommendations changing what people see and click. Describe the experiment as offline continued-training simulation, not a live deployment evaluation.

## Optional retention method, after the baseline

The basic condition has no explicit penalty tying the updated model to the initial model. A examples in the stream still provide supervision for A, but do not guarantee retention.

An extension could add a bounded memory buffer and replay loss, or a precisely specified distillation penalty:

`L_extended = L_incoming + lambda * L_retention`.

Specify buffer size, teacher checkpoint, coefficient and update/sample budget before comparing methods. Adapters instead change trainable parameter structure and require their own freeze/update policy. No mitigation is needed merely to call the initial protocol continued training.

ADER provides precedent for periodically fine-tuning SASRec and then evaluating incoming future windows. Our fixed original-domain retention test and controlled domain mixture are separate experimental choices. [ADER, §3.2 and §4.3](https://arxiv.org/html/2007.12000v1)

## Decisions needed before implementation

- Dataset identities and compatible observed feedback; availability of real interaction order.
- Shared versus domain-specific parameters, including item-vocabulary handling.
- Split boundaries, familiar-item retention cohort, valid targets and negative-sampling rules.
- `m`, `U`, `T`, sequence length, model size, Stage 2 learning rate and seed list.

Recommended first milestone: reproduce A's score after checkpoint reload, then run A-only continued updates with this state lifecycle. Add B only once the evaluator and continuation behavior work.

## Proposed variant: corrupted streams and adapter isolation (24 October 2026)

Agent proposal at the user's request, for team confirmation; it adds a variant without changing the protocol above. It takes the third choice of B in [EXPERIMENT.md](EXPERIMENT.md#the-data-connection-comes-first), explicitly corrupted interactions, and the optional adapter mitigation, and keeps every stage of this protocol. The course frames it as robustness to shill manipulation.

- **Model.** A frozen language model (Llama-3.2-1B, pinned revision) reads a user's history as category names and scores every category name as the next one. Ranking is exact over the closed category set; there is no beam search or sampling. LoRA adapters hold all trainable parameters. Category targets follow from the PESO category study: item targets are too sparse to learn, while category targets drift little between blocks, so the A-only branch is a stable reference.
- **Data.** A is PESO's chronological Instruments blocks with PR 14 categories, the same input as `cp4285 blocks`. Block 0 is `D_A_init`, blocks 1–4 are the stream `S_A`, and each block's last event per user is `R_A`.
- **B as corruption.** Shill profiles are synthetic users injected into an update block. Each pushes one target category, with filler drawn uniformly (random attack), by popularity (average attack), from the most popular categories (bandwagon) or from one segment of real users (segment attack). `alpha` is shill examples over all continued-training examples. Injected users never enter any evaluation set, and their profile lengths match real users'.
- **Conditions.** Frozen, A-only (`alpha = 0`) and shill streams at a few small `alpha` values, all branching from `Theta_0` at a matched update budget. The exposure-matched control drops the shill examples from the same batches.
- **Mitigation: adapter isolation.** A single adapter updated continually is the plain continued-training condition. In the isolated condition each stream trains a new adapter on top of the frozen earlier ones, so an update can be removed after the fact; evaluating with that adapter switched off measures rollback.
- **Outcomes.** `RetentionLoss_A` and `MixtureEffect_A` on the fixed `R_A`, as defined above. The corruption objective is the promoted category's rank and Hit@10 among genuine users, the standard prediction-shift and hit-ratio measures of shilling studies.
- **Off-distribution checks.** The Jensen–Shannon divergence between category mixes (natural block-to-block drift is 0.005–0.023 at level 2), the model's own loss on injected against real profiles, and a classifier's ability to tell injected from real users (near chance means realistic shills).
- **Leakage checks.** No test event in any training input or target; injected users excluded from evaluation; shuffled targets must fall to the popularity baseline; an untrained model's score is reported as the floor for pretraining knowledge.

### Pilot result, 24 October 2026 (one seed, not yet a finding)

Llama-3.2-1B with rank-8 LoRA on PR 14's 30 level-2 categories, seed 42, one epoch per block, on the NUS SoC cluster (about 1.6 GPU-hours). Θ0 scores Hit@1 0.587 and NDCG@10 0.789 on blocks 1–3, between the history rule (0.572, 0.779) and PESO's four-epoch category model (0.597, 0.795). Shills push Keyboards & MIDI (1.3% of test targets) at `alpha = 0.03` in block 1; scores are for genuine users only.

| Condition | Attack | `MixtureEffect_A`, Hit@1 | Pushed category in the top 10 after the attacked block (clean → attacked) | Rank change: attacked block, next block, final state on A | Shill AUC |
| --- | --- | --- | --- | --- | --- |
| Shared adapter | Bandwagon | +0.002 | 62% → 99.8% | −2.0, +0.2, −0.1 | 1.00 |
| Shared adapter | Clone | +0.002 | 62% → 99.7% | −1.5, 0.0, 0.0 | 0.78 |
| Isolated adapters | Bandwagon | −0.004 | 58% → 98% | −1.7, −0.1, +0.1 | 1.00 |
| Isolated adapters | Clone | −0.003 | 58% → 99.6% | −1.7, −0.7, −0.1 | 0.78 |

- A 3% stream lifts the pushed category into nearly every genuine user's top 10, without measurable harm to A (`MixtureEffect_A` stays inside the ±0.004 Hit@1 that persistent user effects already allow on this test set).
- Later clean blocks remove the effect. With a shared adapter one block suffices. With isolated adapters, the later adapters, trained on top of the frozen poisoned one, learn to cancel it: within one block for bandwagon shills, over three for clone shills (−0.7, then −0.4 ranks).
- Rolling the poisoned adapter back at the final state therefore over-corrects: the pushed category falls to 14–16% of top-10 lists against 61% in the clean run, for either attack, and Hit@1 moves by −0.010 to +0.001. A rollback has to come before later updates adapt around the poison, or be followed by retraining them.
- Clone shills, much harder to detect (AUC 0.78 against 1.00), push almost as hard.
- Every attack shifts the block's training-category mix by 0.0019–0.0024 (Jensen–Shannon), inside natural block-to-block drift (0.001–0.012), so aggregate monitoring would not see it.

### Generalisation grid, 25 October 2026 (three domains, three seeds)

The pilot is one seed on one domain. The grid asks whether its pattern holds in two more of PESO's released domains, Video Games and CDs & Vinyl, with seeds 42–44. Each domain gets level-2 categories by PR 14's rule from its own item paths (`cp4285 block-categories`): the first two levels of an item's Amazon `categories` path, with PR 14's text classifier for items that have none.

| Domain | Categories | Items observed / predicted | Classifier holdout accuracy | Block 0 training examples | Test users, blocks 1–4 | Θ0 Hit@1, NDCG@10 on blocks 1–3 |
| --- | --- | --- | --- | --- | --- | --- |
| Instruments | 30 | 50,023 / 3,486 | 0.92 | 90,650 | 1,772–2,289 | 0.591, 0.791 |
| Video Games | 43 | 24,883 / 1,052 | 0.81 | 53,932 | 1,035–1,949 | 0.521, 0.752 |
| CDs & Vinyl | 59 | 39,191 / 0 | n/a | 55,873 | 866–1,078 | 0.346, 0.614 |

On Instruments these labels agree with PR 14's on every observed item and on 94% of predicted ones, so grid numbers differ slightly from the pilot's. Yelp is left out because its tags have no hierarchy.

No pushed category is chosen by hand. The pushed category is the one whose share of block 1–4 test targets is nearest 1.3% (Keyboards & MIDI, Virtual Reality, Soundtracks), and the rare-target arm uses 0.5% (Bass Guitars, Online Game Services, Blue Note Records).

Per domain and seed, the core branches from one `Θ0` are shared and isolated adapters, each clean and under bandwagon and clone shills at `alpha = 0.03` in block 1, plus isolated adapters with the poisoned adapter dropped right after its block (`--rollback-after`). Instruments adds `alpha` 0.01 and 0.05 for every attack and condition. Seed 42 adds the rare target under isolated adapters. All 114 runs finished in about 22.6 GPU-hours on H100 NVL nodes. `replication/peso/grid_aggregate.py` and `grid_checks.py` in the PESO branch checkout recompute every number below from the run folders.

Push and healing at `alpha = 0.03`, mean over three seeds. Rank changes are attacked minus clean for the same condition and cell; negative means the pushed category moved up.

| | Instruments | Video Games | CDs & Vinyl |
| --- | --- | --- | --- |
| Pushed category in the top 10 at the attacked block, clean → attacked | 77% → 96–99% | 38–57% → 77–96% | 9–11% → 32–47% |
| Rank change at the attacked block | −1.4 to −2.0 | −1.8 to −2.2 | −3.1 to −3.5 |
| Rank change one clean block later, shared / isolated | 0.0 to +0.1 / −0.3 to −0.4 | +0.5 / −0.7 to −0.9 | 0.0 to +0.1 / −0.8 to −1.3 |
| Rank change at the final state on A, shared / isolated | 0.0 to −0.1 / −0.3 | +0.1 / −0.2 | −0.1 to +0.1 / −0.3 to −0.6 |
| Shill AUC, bandwagon / clone | 1.00 / 0.76 | 1.00 / 0.71 | 1.00 / 0.63 |
| Category-mix shift of the attacked block (JS) against natural drift between blocks | 0.0020 against 0.0014–0.0131 | 0.0035–0.0038 against 0.0049–0.0162 | 0.0026–0.0035 against 0.0054–0.0089 |

- The attack moves the pushed category up at the attacked block in all 36 seed-branches. CDs & Vinyl has the largest rank change and the most seed spread (SD 1–2 ranks), because its 59 categories leave a pushed category further from the top 10.
- The original task is untouched. Across all 84 attacked seed-branches `MixtureEffect_A` stays within ±0.008 Hit@1, and the paired 95% bootstrap interval excludes zero in 3 of them, about what chance alone gives. Clean updates do not erode A either: `RetentionLoss_A` lies between −0.013 and +0.001 Hit@1, where negative means A improved.

The five predictions:

1. Dose holds on Instruments, the only domain with a dose arm. The rank change at the attacked block is −0.5 to −0.8 at 1%, −1.4 to −2.0 at 3% and −2.0 to −3.0 at 5%. The top-10 rate saturates, since the clean model already puts the category in 77% of top-10 lists. `MixtureEffect_A` stays inside seed noise at every dose.
2. Healing holds in all three domains. One clean block later, isolated adapters keep more of the push than a shared adapter in 15 of 18 seed and attack pairs (0.4–1.3 ranks more on average), and still do at the final state on A in 15 of 18. In Video Games the shared adapter overshoots, leaving the category half a rank below clean.
3. Clone shills push about as hard as bandwagon shills in every domain while a classifier tells them from real users far less well (AUC 0.63–0.76 against 1.00).
4. Rollback timing holds. Dropping the poisoned adapter right after its block ends within 0.07–0.58 ranks of the clean run at the final state, against 1.9–2.4 ranks for a rollback at the end, smaller in 17 of 18 pairs. A late rollback over-corrects: the pushed category ends 1.6–2.1 ranks below clean (16 of 18 pairs), and Hit@1 falls by 0.003 (Instruments), 0.007 (Video Games) and 0.018 (CDs & Vinyl). Skipping block 1's genuine data along with the shills costs nothing measurable later (own-block Hit@1 within ±0.001 of clean).
5. Invisibility holds in Video Games and CDs & Vinyl, where the attacked block's category mix moves less than every natural block-to-block change. In Instruments it moves less than three of the four.

The rare-target arm has one seed. In CDs & Vinyl, Blue Note Records rises 5.6–6.9 ranks at the attacked block and is still 2.7–3.0 ranks up at the final state under isolation, the largest and most persistent effect in the grid. Bass Guitars in Instruments behaves like the common target (−1.6 to −1.9). Online Game Services in Video Games is a poor test: the clean model already ranks it about 8th of 43 for every user, so it moves only 0.6–1.5 ranks.

Limits: three seeds, one model size (Llama-3.2-1B, LoRA rank 8) and one attacked block. Realism is judged by one classifier on four profile features and by the category mix, so a stronger detector could separate clone shills better. The data are Amazon reviews in PESO's released blocks, so a target is a reviewed category, not a purchase.
