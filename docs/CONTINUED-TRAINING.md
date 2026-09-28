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

Evaluate the initial A checkpoint to obtain `Q_A_initial = NDCG@10(Theta_0; R_A)`.

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

Use NDCG@10 as the proposed primary metric and Hit@10 as a secondary metric. For one relevant next-item target, Hit@10 equals Recall@10; they differ only when several future events count as targets (the classical `--targets` option). Report B quality alongside A so that preserving A by failing to learn B is visible. Compare the initial frozen model with itself over the same test set as a basic evaluator check, then repeat trained comparisons across seeds.

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
