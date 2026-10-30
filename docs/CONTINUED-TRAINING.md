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

## Attack catalogue and rollback timing (26 October 2026)

Agent proposal at the user's request, extending the corrupted-stream variant above; the protocol, conditions and outcomes are unchanged. The learner is treated abstractly: a frozen language model with LoRA adapters, updated on a stream of next-category examples. Every corruption is implemented behind one interface in `experiment/corruption.py` and selected with `--corruption` on `cp4285 adapter-blocks` (the shill fill stays on `--attack`); realism is measured per run — the category-mix shift and, where profiles are new or edited, a profile-separability AUC — never assumed. `cp4285 robustness-plan` prints the run matrix as ready-to-launch lines, stage A (core) and stage B (stealth). The catalogue's 189 new runs (21 per domain and seed, three domains, seeds 42–44, stages A and B) completed on the NUS SoC cluster's H100 NVL nodes on 26 October through the October grid's launcher, which skipped the three shared references per domain; run folders live in that checkout's `lm-runs/`. Results for all three seeds are below. The implementation is covered by offline tests on toy blocks (95 passing); the cluster's two pre-existing `test_metadata.py` failures are the missing optional `meta` extra in that environment, unrelated to this catalogue.

| # | Attack | How it is run | What it changes | What it is meant to beat |
| --- | --- | --- | --- | --- |
| 1 | Low and slow | `--attack bandwagon --alpha 0.01 --placement all` | 1% shills in every update instead of one 3% block | undoing a single update; low-α alarms |
| 2 | Hijacked real accounts | `--corruption real-history` | real shoppers' training events gain target items; accounts stay real, and each user's last two events (validation and test) are untouched | new-account detection; single-update undo |
| 3 | Bury a rival | `--corruption demotion --rival-share 0.05` | shills copied from a rival category's real customers, buying the pushed category; the rival's rank is tracked beside the push | boosting-only monitoring |
| 4 | Hidden trigger | `--attack segment --trigger-share 0.1` | segment shills whose filler is the resolved trigger category; the push is measured separately for genuine users whose history contains it | aggregate statistics |
| 5 | Seller miscategorisation | `--corruption relabel --relabel-frac 0.1` | a fraction of a block's items are re-listed under the pushed category: the training-time category map changes, users untouched, evaluation keeps the true labels | every shopper-based detector |
| 6 | Poison the starting data | `--placement init` on the init stage | the corruption enters block 0, so it sits in the base weights before any adapter exists | any rollback |
| 7 | Plain vandalism | `--corruption vandalism` | uniform-noise profiles with no target; `--push` optional and only measured | overall-quality alarms |

Modeling choices these rely on, all open to revision:

- **real-history** keeps each corrupted user's validation and test events fixed, so evaluation labels stay true and only the training signal is poisoned. Its separability AUC asks whether edited profiles stand out from untouched ones; there are no new accounts to find.
- **relabel** splits the category map: training rows are built from the remapped map (what the listings claim), evaluation from the true map (what an audit would record). A re-listed item therefore teaches the wrong association while users' test targets stay correct. The report's `alpha` is the induced share of training examples whose recorded category changed, comparable across corruptions; `--relabel-frac` itself is a share of items.
- **demotion** needs a rival with customers in the attacked block and is refused otherwise. The rival is resolved from `--rival-share` by the same nearest-test-share rule as `--push-share`, so no watched category is picked by hand; the same rule resolves the hidden trigger's `--trigger`, and a segment attack without an explicit `--segment` fills from its trigger.
- Placement `init` corrupts block 0 before Theta_0 exists. A branch from such an init refuses rollback: the corruption sits in the base weights, which is the point of the attack. Its effect is read against a clean-init branch of the same condition.

### Rollback timing

`--rollback-block N` (isolated condition) keeps the attacked block's adapter through block N and drops it there; `--rollback-after` remains its N = attacked-block special case, and the two cannot be combined. Scanning N turns the grid's late-rollback over-correction into a timing curve. The summary adds `drop_effect`: the dropped cell's pushed-category rank against the clean run's state at the same block, so a positive value is the over-correction itself. The weapon reading of the existing grid result: an attacker boosts a rival's category, lets the defender notice and undo the update late, and the undo buries that rival below its natural level with the defender's own defence — a rollback at the end lands 1.9–2.4 ranks below clean, against 0.07–0.58 for dropping immediately, and the sweep adds the intermediate timings (drop after blocks 2 and 3) to find how long a defender can wait before the undo does more harm than the attack.

Summaries also compare the `rival` cells and the trigger split (`push_trigger` against `push_other`) between attacked and clean runs; a hidden trigger is exactly the case where the aggregate push cell moves little while the trigger group's moves a lot. The planner's stage B clean-reference runs carry the same rival and trigger measurement rules, so the summary's (condition, pushed category, rival) key compares like with like.

Rough cost at one seed across the three released domains, from the grid's ~0.2 GPU-hours per run: stage A adds 8 runs per domain-seed (its init and clean references repeat the grid's on purpose; the runner refuses to overwrite them, so a launcher can skip those lines) plus 4 rollback-timing runs already counted, about 5 GPU-hours; stage B adds 13 runs per domain-seed, about 8 GPU-hours. Seeds 43–44 ran the same day; measured training and scoring time for the catalogue's 189 runs is 32.9 GPU-hours.


### Results, seeds 42–44 (26 October 2026)

Seeds 43 and 44 ran on 26 October; two plans hit the 3-hour job limit and were resubmitted. Every catalogue cell now has nine domain–seed pairs, and with the October grid the study holds 303 runs. Every number below is recomputed from the runs' `metrics.json`. A rank change is the pushed (or rival) category's mean rank in the attacked run minus the clean run of the same domain, seed and condition, so negative means the category moved up. Stage-B runs are compared with clean runs that carry the same rival and trigger rules. Counts in brackets are the domain–seed pairs moving up.

**Push by block** (mean of 9 pairs; block 1 is attacked, and low-and-slow attacks every block):

| Corruption | Condition | Block 1 | Block 2 | Block 3 | Block 4 | Top-5 share at block 1 |
| --- | --- | --- | --- | --- | --- | --- |
| clean | shared / isolated | 0 | 0 | 0 | 0 | 4.7% / 4.1% |
| bandwagon 3% | shared | −2.53 (9) | +0.22 | +0.11 | +0.10 (2) | 9.6% |
| | isolated | −2.14 (9) | −0.59 | −0.13 | −0.13 (6) | 7.4% |
| clone 3% | shared | −2.31 (9) | +0.16 | +0.20 | +0.01 (5) | 7.9% |
| | isolated | −2.23 (9) | −0.88 | −0.36 | −0.26 (6) | 5.8% |
| low-and-slow 1% every block | shared | −1.65 (9) | −0.62 | −0.93 | +0.02 (4) | 5.3% |
| | isolated | −1.45 (9) | −1.18 | −0.88 | −0.51 (7) | 5.5% |
| real-history 3% | shared | −4.69 (9) | −0.63 | −0.18 | −0.12 (6) | 36.4% |
| | isolated | −4.80 (9) | −1.83 | −1.14 | −0.80 (9) | 31.2% |
| relabel 10% of items | shared | −7.44 (9) | −0.99 | −0.56 | −0.09 (6) | 92.2% |
| | isolated | −7.68 (9) | −1.94 | −1.55 | −1.18 (9) | 88.6% |
| demotion 3% | shared | −2.00 (9) | +0.32 | +0.12 | +0.07 (3) | 8.3% |
| | isolated | −1.85 (8) | −0.51 | −0.23 | −0.15 (7) | 7.8% |
| hidden trigger 3%, all users | shared | −0.73 (9) | +0.47 | +0.15 | +0.12 (2) | 7.4% |
| | isolated | −0.81 (6) | −0.02 | +0.17 | −0.07 (6) | 6.2% |
| vandalism 3% | shared | −0.03 (4) | +0.09 | +0.15 | +0.04 (4) | 5.7% |
| | isolated | −0.18 (6) | −0.07 | −0.02 | −0.11 (5) | 4.7% |
| poisoned base 3% | shared | +0.47 (3) | +0.60 | +0.41 | +0.50 (1) | 3.9% |
| | isolated | +0.54 (3) | +0.30 | +0.35 | +0.22 (3) | 3.4% |

**Rollback timing** (isolated; drop effect = pushed-category rank right after the poisoned adapter is deleted, minus clean at the same block; positive = buried below clean; mean of bandwagon and clone over three seeds):

| Domain | Drop after block 1 | After 2 | After 3 | At the end | Hit@1 at the end, late rollback |
| --- | --- | --- | --- | --- | --- |
| Instruments | +0.52 | +1.54 | +1.45 | +1.83 | −0.5 points |
| Video Games | +0.22 | +1.37 | +1.54 | +2.07 | −0.7 points |
| CDs & Vinyl | −1.04 | +0.57 | +1.35 | +1.65 | −1.1 points |
| All 18 runs | −0.10 (8 positive) | +1.16 (15) | +1.45 (16) | +1.85 (16) | −0.8 points |

The same runs separate the poisoned adapter from the ones trained after it. Scoring each state with and without A1 gives the pushed category's gain over clean:

| After block | With A1 | Without A1 | A1's own push |
| --- | --- | --- | --- |
| 1 | +2.18 | +0.10 | 2.08 |
| 2 | +0.74 | −1.16 | 1.90 |
| 3 | +0.24 | −1.45 | 1.69 |
| 4 | +0.19 | −1.85 | 2.04 |

"Without A1" comes from the drop runs at blocks 1–3 and from the rollback cell at block 4. A1's push stays near 2 ranks because it is frozen. The adapters trained after it, all with A1 switched on, learn a counter-push that grows block by block. Deleting A1 leaves that counter-push alone, which is the over-correction.

Findings:

- **Every promotion lands in its block.** Bandwagon, clone, low-and-slow, real-history and relabel move the pushed category up at block 1 in all nine pairs under both conditions.
- **Corrupting real accounts beats faking them.** At the same 3% of examples, real-history moves the category 4.7–4.8 ranks against 2.1–2.5 for shills, about twice as far. Edited profiles separate at AUC 0.65–0.78 (mean 0.73), and there are no new accounts to flag.
- **Relabel is the strongest attack.** Re-listing 10% of items moves the category 7.4–7.7 ranks and into 89–92% of top-5 lists at block 1. No profile detector applies. The training-category mix shifts 0.032–0.043 (Jensen–Shannon) against natural drift of 0.001–0.016. It is also the one corruption with a consistent quality cost: NDCG@10 in the tampered block falls 0.3–1.5 points in all 18 runs.
- **One-off effects fade, and the condition decides the residue.** By block 4 a shared adapter is back at clean or slightly below. Isolated adapters keep 0.13 (bandwagon) to 1.18 (relabel) ranks. Paired by domain and seed, isolated minus shared at block 4 is −0.23 (bandwagon, 8/9), −0.27 (clone, 6/9), −0.68 (real-history, 8/9) and −1.10 (relabel, 9/9).
- **Low-and-slow keeps pressure only under isolation.** Isolated adapters end 0.51 ranks up (7/9) with a top-5 share 1.1 points above clean. A shared adapter ends at +0.02 (4/9). Block 4 also receives its own 1% dose, so this measures sustained pressure rather than survival.
- **The later the rollback, the deeper the over-correction, in all three domains.** See the tables above. After a drop at block 3, one more clean block leaves the category 0.50 ranks above clean on average (13 of 18 runs), ranging from 2.78 above to 1.57 below.
- **The hidden trigger moves its segment only.** At block 1 users whose history contains the trigger category move 1.94 (shared) and 2.04 (isolated) ranks, 9/9. Other users barely move (+0.01 and −0.07). The aggregate moves 0.73–0.81. Trigger users are about 38% of genuine test users.
- **Demotion does not bury the rival.** The rival's mean rank changes by at most 0.11 at any block, with no consistent sign, while the pushed category rises 1.9–2.0.
- **Vandalism at 3% changes nothing measurable.** The watched category moves less than 0.2 ranks and own-block Hit@1 stays within ±1.3 points.
- **A poisoned base heals.** Before any update the init state promotes the category by 0.64 ranks on blocks 1–4 (7/9). After four clean blocks it sits 0.22 (isolated) and 0.50 (shared, 8/9) ranks below clean.
- **The original task is untouched.** Own-block Hit@1 stays within ±1.8 points of clean in every cell, with mean shifts under 0.2 points and about half of cells negative. Block-0 retention moves at most 0.5 points on average.

Changes from the seed-42 reading written earlier on 26 October:

- The rollback over-correction is not specific to Instruments and Video Games. CDs & Vinyl seed 42 went −4.35 to −1.15; seeds 43 and 44 over-correct by 2.7–3.7 ranks at the end.
- Low-and-slow on Video Games costing 0.038 Hit@1 was the block-0 cell at seed 42, scored on 1,000 users. Seeds 43 and 44 give −0.3 and +1.8 points.
- "Sustained 1% persists" holds under isolated adapters only, and its block-4 value includes that block's dose.
- Real accounts push about twice as hard as shills, rather than two to three times.

Noise: re-running the same clean configuration changes the pushed-category rank by at most 0.20. Within a domain, seeds spread a cell's rank change by a median of 0.42 ranks. Two random draws of the same 1% shill recipe differed by 1.4 ranks at block 1 (Instruments, seed 42). Means over nine pairs below about 0.3 ranks are within noise.

Limits: one model size (Llama-3.2-1B, LoRA rank 8) and one attacked block; dose only on Instruments; rare targets only at seed 42. The two conditions also differ in capacity (one adapter against five summed), so their contrast measures persistence in this harness. The modelling choices listed above remain agent proposals for team review.

**Provenance.** The catalogue code is commit `d38d27c` and tag **`attack-catalogue-2026-10-26`**. On 29 October 2026 the five source files were hashed in the NUS SoC cluster checkout `/mnt/scratch/w/wyenzhe/peso/cp4285-main/src` and all five SHA-256 values matched the tagged files (`corruption.py`, `robustness.py`, `adapter_blocks.py`, `shilling.py`, `cli.py`). The run `metrics.json` files record the CLI config, the learner description and the Torch version, but no VCS revision, so this is a source-hash match rather than a revision recorded by the artifacts themselves. The 29 October account-strata and dose tooling (`e3a19a3`) is built on top of the tag; it does not change the catalogue code.

## Account strata and critical mass (agent proposal, 29 October 2026)

Implementation of issues #18 and #19. No run has been launched and no result is claimed; the run matrices and costs are proposals for team approval.

**Account quality.** `real-history` can select accounts from a deterministic stratum instead of drawing them uniformly:

- `--user-stratum activity` counts a user's training targets. `--user-stratum reach` averages `log1p` item popularity over the user's distinct training targets, where popularity is the number of users whose training targets contain the item. Reach is interaction reach, not social followers.
- Strata are computed from training targets only; the last two events of every user stay validation and test and are never read.
- `--user-side top|bottom`, with exactly one of `--user-share` or `--user-accounts`, takes the same number of accounts from one ranked list. Ties break on user id and no RNG selects users, so a plan replays exactly.
- `--user-budget accounts` spreads the fixed `alpha` event budget evenly; `events` spreads it in proportion to each account's trainable capacity; `value` shrinks the top-side account count by `--value-k`, so one high-stratum account carries about `k` times a low-stratum account's insertions. The event budget itself is still set by `alpha`, so both sides are event-matched by construction.
- `--eval-strata` adds `push_activity_top`, `push_activity_bottom`, `push_reach_top` and `push_reach_bottom` to every scored cell. The split is a median-rank split of genuine test users, with ties by user id.
- Reports add `selection` (metric, side, budget, `k`, population, accounts, events, mean and maximum insertions, boundary, Spearman correlation between activity and reach) and `separability_auc_same_stratum`, which compares edited profiles with untouched users on the same side so the AUC is not merely activity or reach showing through.

First pass: Instruments, seeds 42–44, isolated, 2 strata × 2 budgets plus a `k` grid and clean references, 22 runs at roughly 0.2 GPU-hours each (about 5.3 GPU-hours). Extend to `activity` only if the `reach` effect is large. This answers "which accounts", not only "how many"; the value invariant is an assumption, not an estimated exchange rate.

**Critical mass.** `--stage dose` emits one attacked branch per dose and seed, with a shared clean reference. Doses are shares of the attacked block's training examples: 0.001, 0.0025, 0.005, 0.01, 0.02, 0.03 and 0.05 in the default plan. The primary endpoint is the mean rank change of the pushed category among genuine users, because top-10 share saturates (the clean Instruments model already has the category in 77% of top-10 lists). Top-5 share, top-1 share, top-10 churn and a normalised gain are reported beside it.

`cp4285 dose-summary` pairs each attacked run with the clean run of the same seed, pushed category and condition, joins on user, and bootstraps the mean difference over users. The pre-registered rule is: the smallest dose whose mean rank change is at least one rank in the promoted direction and whose bootstrap interval lies entirely below zero in at least two runs. A flat or non-monotone curve is reported as "no threshold identifiable", not smoothed away. First pass: Instruments, seeds 42–44, 37 runs (about 7.4 GPU-hours, 9 with overhead); the largest- and smallest-effect domains follow only if the core curve is informative.

Definitions that need team confirmation before any run:

- `reach` or `activity` as the primary account metric, and the `k` grid;
- whether "matched events" means the capacity-proportional split used here;
- the threshold constants (-1 rank, top-5 at 50%) and the two-of-three-runs rule;
- isolated adapters only for the core curve.

Limits: the strata are confounded with history length and item popularity, so report both metrics and the overlap; a category-level threshold need not transfer to item-level recommendation; a "critical number" is a property of this testbed, not a physical constant.

### First results, 30 October 2026

Ran on Instruments, seeds 42–44, isolated adapters, one attacked block. The strata pass used 10% of users at a 3% event dose; the dose pass used the legacy uniform `real-history` selection at 0.1%–5%. No result below is a cross-domain claim.

**Account quality.** Mean rank change of the pushed category against the clean run of the same seed; negative means it moved up. Clean means were 9.79, 9.38 and 8.26.

| Arm | Mean Δrank | SD | Accounts | Insertions/account |
| --- | --- | --- | --- | --- |
| Top reach, matched accounts | −2.92 | 0.44 | 177 | 1.48 |
| Bottom reach, matched accounts | −2.90 | 0.43 | 177 | 1.48 |
| Top reach, matched events | −2.43 | 0.72 | 177 | 1.48 |
| Bottom reach, matched events | −2.97 | 0.71 | 177 | 1.48 |
| Top reach, value k=2 | −1.83 | 1.20 | 89 | 2.94 |
| Top reach, value k=4 | −1.57 | 0.37 | 45 | 5.82 |
| Bottom reach, value k=2 | −2.89 | 0.44 | 177 | 1.48 |
| Bottom reach, value k=4 | −2.90 | 0.45 | 177 | 1.48 |

- **Reach does not order accounts.** Top and bottom are within noise at matched accounts; the seed-level sign flips. A "hub-like" user is not a stronger vehicle on this testbed.
- **Concentrating the same event budget into fewer accounts weakens the attack.** Top-reach k=4 uses 45 accounts at about 5.8 insertions each and moves the category 1.57 ranks, against 2.92 for 177 accounts at about 1.5 insertions each. One high-reach account is not worth four low-reach accounts here.
- The bottom-side value arms keep the full account count by construction, so they are a determinism check, not an independent k contrast. The fair comparison is top k=2/4 against bottom full accounts.

**Critical mass.** Paired user bootstrap (2,000 draws), 1,740 genuine users per run, three seeds.

| Dose | Mean Δrank | 95% CI | Top-5 | Top-10 | Churn |
| --- | --- | --- | --- | --- | --- |
| 0.1% | +0.47 | (+0.13, +0.94) | 0.04 | 0.68 | 0.45 |
| 0.25% | −0.80 | (−1.89, +0.35) | 0.06 | 0.98 | 0.41 |
| 0.5% | −0.22 | (−0.84, +0.66) | 0.04 | 0.88 | 0.45 |
| 1% | −2.13 | (−3.22, −1.55) | 0.10 | 1.00 | 0.41 |
| 2% | −3.79 | (−4.91, −2.79) | 0.49 | 1.00 | 0.45 |
| 3% | −3.85 | (−5.24, −3.01) | 0.55 | 1.00 | 0.48 |
| 5% | −4.45 | (−5.63, −3.12) | 0.81 | 1.00 | 0.56 |

- **The pre-registered threshold is 1%**: mean −2.13 ranks, interval entirely below zero, and three of three seed-level intervals below zero. The 0.25% and 0.5% points are non-monotone and their mean gains are below one rank, so they do not meet the rule.
- **Top-5 reaches 50% at 3%** and 81% at 5%. **Top-10 saturates at 1%**, which is why the threshold is defined on mean rank rather than top-10. The pushed category never reaches rank 1 at any tested dose.
- The low-dose non-monotonicity is expected at this scale: the number of edited examples changes the training length and therefore the run's RNG trajectory. The 0.1% arm edited 9 users on seed 42; that point is not evidence of a backfire.
- These are category-level results on one domain and one attacked block. They do not show that the same 1% threshold holds for item-level recommendation, another domain or a different adapter capacity.

The first dose attempt was discarded: its dumps stored the user's target rank rather than the pushed category's rank. The corrected runs and the `push_rank` dump field are the basis of the table above; the bad folders are kept under `bad-dumps-2026-10-29`.

### Cross-domain replication, 30 October 2026

Video Games (43 categories) and CDs & Vinyl (59) ran the same protocol as Instruments: seeds 42–44, isolated adapters, one attacked block, six doses from 0.25% to 5%, and top/bottom reach strata at 3%. 54 new runs (jobs 1041298–1041303). The clean pushed-category means were 10.50 (VG) and 14.48 (CD), against 9.14 on Instruments, so raw rank changes are not directly comparable.

**Dose.** Relative gain is `(clean_rank − attacked_rank) / (clean_rank − 1)`, averaged per user; top-5 is the share of genuine users with the pushed category in their top five.

| Dose | Ins gain | VG gain | CD gain | Ins top-5 | VG top-5 | CD top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25% | 0.076 | 0.008 | 0.063 | 0.06 | 0.02 | 0.02 |
| 0.5% | 0.000 | 0.089 | 0.146 | 0.04 | 0.05 | 0.03 |
| 1% | 0.237 | 0.161 | 0.239 | 0.10 | 0.06 | 0.04 |
| 2% | 0.434 | 0.301 | 0.399 | 0.49 | 0.08 | 0.13 |
| 3% | 0.429 | 0.431 | 0.417 | 0.55 | 0.25 | 0.13 |
| 5% | 0.503 | 0.531 | 0.526 | 0.81 | 0.46 | 0.24 |

- **The mean-rank threshold is around 1% in all three domains.** The formal rule selects 0.25% on CDs, but that point has one positive seed and its pooled interval includes zero. All three CD seeds are negative from 0.5% upward, so 0.5–1% is the defensible reading. Video Games and Instruments select 1%.
- **Raw rank change is mostly baseline.** CDs move −7.4 ranks at 5% because the clean category sits near rank 14; Instruments move −4.45 from rank 9. Once normalised, the three domains are close at 3–5% (0.42–0.53), though Video Games is lower at 1–2%.
- **Top-5 visibility does not transfer.** Instruments reaches 50% at 3% and 81% at 5%; Video Games reaches only 46% at 5%; CDs 24%. The category is already more dispersed in VG/CD, so a similar relative gain does not produce the same visible placement.
- **The reach advantage is domain-dependent.** On Video Games, top reach is stronger than bottom in all three seeds: −3.91 against −3.23 ranks (differences 0.5–1.0). On Instruments the two are equal (−2.92 against −2.90). On CDs the top mean is higher (−4.77 against −3.34) but the bottom side ranges from 0.00 to −5.74, so the contrast is inconclusive. The Instruments null is not universal.
- **The six `cross-init` runs were unused.** The plans named a fresh init line but every branch pointed at the existing `lm-{domain}-s{seed}-init` checkpoint. They can be deleted; they do not affect any result.

Limits: three seeds, one model size, one attacked block, category targets, isolated adapters. The CD low-dose curve is noisy and the CD reach contrast needs more seeds. The stable part is the relative ordering and the ~1% mean-rank edge, not a universal top-5 threshold.
