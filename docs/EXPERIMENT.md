# Proposed experiment definition

Status: agent proposal for verification, not a protocol already agreed by the team. Update, 28 September 2026: the original SASRec PyTorch A-only pilot has synthetic checks; the [classical comparator](CLASSICAL.md) reports exploratory real-data validation sweeps under a different leave-last-out protocol. No common-protocol classical/neural comparison has run.

For Hongyi's confirmed neural-model contribution, see [MODERN-RECOMMENDER.md](MODERN-RECOMMENDER.md), which formalizes a proposed shared Transformer, the update lifecycle, and the distinction between retention loss and performance relative to A-only continued training.

The detailed proposed execution contract is [CONTINUED-TRAINING.md](CONTINUED-TRAINING.md). It refines the general design here without locking a dataset or training budget.

## Research question

How does changing the composition of continued-training data affect recommendation quality on an original task, and how does this differ between a classical and neural/advanced recommender?

The immediate output should be a short specification that fills these fields:

| Decision | Required detail | Current state |
| --- | --- | --- |
| User task | Who receives which recommendations, from what candidate catalogue? | Open |
| A | Original domain, observed interaction signal, relevance definition | Open |
| B | A legitimate new domain, a chronological shift, or synthetic corruption | Open; two-domain mixing leads the sketch |
| Data connection | Shared real users/items/features, or shared model parameters | Open; essential |
| Updates | Which parameters change, at what cadence, on which examples? | Continued mini-batch training proposed |
| Models | One classical baseline and one neural/advanced model | Required family comparison; exact choices open |
| Primary outcome | Ranking quality on held-out A | Hit@10 for the next item, Recall@10 for the next items (team decision, 28 September, issue #8); selection metric proposed, see [Ranking metrics](#ranking-metrics) |
| Claim boundary | What does this setup represent in a real application? | Open |

## The data connection comes first

For plain matrix factorisation, a user-item score uses that user's factor and that item's factor. If A and B contain entirely different users and items, and updates affect only the factors for B, learning B does not change A's factors or scores. This is an architectural deduction, not a result from this project. Shared regularization, optimizer state, global parameters, or negative sampling can introduce other paths of influence and must be documented. [Foundational MF paper](https://doi.org/10.1109/MC.2009.263)

Never treat user 17 from MovieLens as user 17 from a shopping dataset. Numeric identifiers do not establish shared people. Namespace unrelated IDs.

There are several defensible choices, each answering a different question:

| Choice | What it tests | What must be justified |
| --- | --- | --- |
| Two domains through a shared model | Interference as another domain is learned | Shared parameters/features, identity handling and application motivation |
| Later periods or categories in one dataset | Retention under temporal/category change | Chronological split, available overlap and what constitutes A versus B |
| Explicitly corrupted interactions in one identity space | Sensitivity to controlled training noise | Corruption rule and limits of the synthetic scenario |

The first row most closely matches the supplied sketch. The other rows are alternatives for review, not silent replacements. An MF model with isolated factors can be a useful control if that isolation is intentional; it should not be used to promise a forgetting curve.

## Proposed evaluation

1. Construct initial A training data, validation data, unused A update data, and a held-out A retention test. Where time matters, initial training precedes update data; keep retention-test labels out of every update stream. Explain which period/population the fixed test measures.
2. Save an A-trained checkpoint for each model. Record the initial absolute Hit@10 and the other standard metrics below.
3. Branch each update condition from the same initial checkpoint, with a stated optimizer-state policy. Fix evaluation inputs, relevant items, eligible candidates and previously-seen-item filtering across checkpoints. Freeze sequential input histories if using a sequential model.
4. Apply a matched update budget and evaluate A at fixed checkpoints. Rank all eligible A items exactly for a manageable catalogue; if sampling is necessary, document it and freeze the sampled candidates. Sampling can alter comparative conclusions. [Krichene and Rendle, 2020](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)
5. Report absolute scores and signed change from the initial checkpoint, along with repeated-seed variation for final comparisons. Do not select conditions or tune models on the held-out A test.

## Ranking metrics

Every evaluator reduces its scores to the 1-based rank $r$ of each held-out target and calls `cp4285.experiment.metrics.from_ranks`, so all workstreams report the same keys:

- $\mathrm{Hit@}K = \mathbb{1}[r \le K]$, the reported next-item metric at $K = 10$.
- $\mathrm{NDCG@}K = \mathbb{1}[r \le K] / \log_2(r + 1)$. With one target the ideal DCG is 1.
- $\mathrm{MRR@}K = \mathbb{1}[r \le K] / r$, plus uncut MRR.
- `recall@K` is kept as an alias of `hit@K` for older run files. With one target they are equal. Per-user Recall@10 for several targets belongs to the next-items task.

Reports give Hit@5, Hit@10, Hit@20, NDCG@5, NDCG@10, NDCG@20 and MRR@10, each averaged over users. Checkpoint and model selection use validation Hit@10, and NDCG@10 breaks ties (`selection_key`). That rule was proposed on PR #9 under issue #8 and still needs team confirmation.

The shared function fixes the formulas only. Candidates and masking stay protocol-specific. The classical evaluator ranks the first `n_candidates` items (the base domain or the whole catalogue) and excludes items the user has seen. The neural evaluator ranks the whole fixed catalogue and keeps seen items eligible, because a target can be a repeat review. Scores from the two protocols are not comparable.

With few candidates the cutoff matters. In PESO's Instruments data with PR 14's level-2 subcategories, 10 of 30 categories cover 97% of test targets, so Hit@10 saturates near 1 for any sensible ranking. There, compare Hit@1, Hit@5, NDCG@10 and MRR@10, and always report a popularity baseline.

Adding B items to the evaluation candidate pool is a separate catalogue-expansion experiment. It can worsen A ranking without changing A's scoring function, so it should not be folded into the first retention result.

## Comparison conditions

| Condition | Purpose |
| --- | --- |
| Frozen A checkpoint | Checks stable evaluation; anchors retention without further learning |
| A-only continued training | Measures changes from continuing updates within A |
| A/B mixed continued training | Tests stream composition under a fixed update budget |
| Optional mitigation | Later test of replay or adapters under an explicit resource budget |

Define `p_B = B examples / all continued-training examples`. A pilot might use `p_B = 0, 0.5, 1.0`, once the data connection is valid. Hold batch size and update steps fixed. Higher B share necessarily reduces A exposure under this design; describe the result as the effect of stream composition, not proof of harmful B gradients alone. Isolating that mechanism would need additional exposure-matched controls.

The meeting's “200%” could mean `B/A = 2`, equivalent to a B share of two-thirds. It cannot mean a 200% fraction of the total stream. Record the denominator on every plot.

Use fresh, unused A data for the A update stream where available. Reusing stored initial A examples is already replay; a separate replay condition must specify what extra memory/selection policy it adds.

A proposed corrupted-stream variant (24 October 2026, agent proposal): shill profiles as B, a language model with adapters as the neural learner and adapter isolation as the mitigation. It keeps this design's conditions and outcomes; see [CONTINUED-TRAINING.md](CONTINUED-TRAINING.md#proposed-variant-corrupted-streams-and-adapter-isolation-24-october-2026). Extended on 26 October into a seven-corruption attack catalogue — including corruptions of real activity that no shopper-based detector sees, a poisoned starting state no rollback reaches, and a rollback-timing sweep that measures the defender's own undo as a weapon; see [CONTINUED-TRAINING.md](CONTINUED-TRAINING.md#attack-catalogue-and-rollback-timing-26-october-2026).

## Scope and interpretation

The first pilot establishes that preprocessing, updates and the metric work and that the intended shared parameters can change. It need not show degradation. Extend the same valid protocol to both model families before comparing them.

If B represents legitimate new preferences, also measure learning B before claiming a mitigation improves the retention/adaptation tradeoff. A frozen model can retain A by refusing to adapt. If B is corruption, define a corruption objective instead.

Adapters remain optional. A claimed benefit requires stating whether the trunk, item embeddings, original head and adapters are frozen or trained. FAISS is optional retrieval infrastructure, not the neural model. [FAISS documentation](https://faiss.ai/)

For error and ethical analysis, a useful proposed slice is users/items with sparse versus frequent interactions: does the average conceal loss for less represented groups? Offline data does not by itself establish authenticity, human welfare, or adversarial intent.
