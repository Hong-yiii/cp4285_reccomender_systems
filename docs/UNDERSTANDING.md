# Project understanding

Status: reconstructed from the 23 September meeting and supplied diagram, pending team verification. Transcript timestamps refer to blocks, not exact utterance times. The transcript is machine-generated and contains errors.

## How the idea developed

| Meeting stage | Interpretation | Evidence |
| --- | --- | --- |
| Original motivation | Recommend authentic items despite many synthetic or near-duplicate items. Authenticity and suitable labels were difficult to define. | 00:08:39–00:13:53 |
| Proposed proxy | Use one domain initially, then introduce data from another domain in increasing amounts. Shopping and movies were examples. | 00:15:07–00:16:30 |
| Unresolved objection | Why would these domains enter one recommender, and how would their different fields/identities connect? | 00:16:30–00:20:38 |
| Clearer experiment | Compare continued training on A-only data with continued training on mixtures including B; evaluate the original task. | 00:24:15–00:25:36; 00:43:36 |
| Intended contribution | Implement a classical and neural/advanced model, then investigate and explain their behavior. A new architecture was not agreed. | 00:31:02–00:37:07 |
| Possible extension | Domain-specific adapters or heads might reduce interference. This was conditional on scope/time. | 00:27:28; diagram |
| Late alternative | Ordinary chronological updates on one dataset might also expose loss of earlier performance. The group left room to revise after reading papers. | 00:59:47–01:04:48 |

The strongest shared direction is an empirical study of recommendation quality during continued learning. The transcript does not establish a final choice between cross-domain mixing and a chronological shift.

## What the sketch adds

The sketch explicitly puts the changed data **after initial training, during continued learning**, and says to evaluate on Dataset 1. It separates data/evaluation, a basic baseline, a neural baseline, and integration. The adapter box is phrased as a question.

Interpret “all evals on Dataset 1” as the intended primary retention evaluation. If the project also claims successful adaptation to legitimate B data, an additional B evaluation would be needed; this is a proposed refinement.

## Current decision status

| Item | Status |
| --- | --- |
| Classical baseline plus neural/advanced model | Strong meeting agreement; also matches public course expectations |
| Initial training followed by continued updates | Strong meeting agreement |
| Evaluate original-task quality over updates | Strong meeting agreement |
| Dataset and evaluation work belong together | Strong meeting agreement |
| Prepare critique materials before the full study | Explicit immediate priority at 00:56:18 |
| Shopping versus movies | Illustrative, not selected |
| MF, two-tower, transformer, recurrent model | Candidates; no final selection |
| Hit@10 / Recall@10 | Decided 28 September (issue #8): Hit@10 for next item, Recall@10 for next items; NDCG@10 breaks ties only |
| Mixtures such as 20%, 50%, 100%, 200% | Examples with no agreed denominator |
| Neural model will be more resilient | Hypothesis at 00:22:40 |
| New data will necessarily cause forgetting | Not established; explicitly questioned at 01:03:17 |
| Adapters will fix the problem | Optional hypothesis, not a commitment |
| Exact related-work paper | Unconfirmed; likely ADER, see SOURCES.md |

## Likely roles

These are proposed roles inferred from 00:57:55–00:59:47, not confirmed assignments.

Update, 25 September: Hongyi explicitly confirmed that his contribution is the modern deep-learning recommender. The other allocations remain provisional. See MODERN-RECOMMENDER.md for the follow-up validation and proposal.

| Workstream | Status |
| --- | --- |
| Neural/advanced model | Hongyi; confirmed by the user on 25 September |
| Classical model | Provisional teammate allocation; verify with the team |
| Dataset selection and evaluation | Provisional teammate allocation, with additional dataset support |
| Integration and presentation | Shared/unconfirmed |

Personal teammate details remain in the local-only source record rather than this public summary.

## Technical interpretations to correct

- **FAISS is a vector-search library.** It can index model embeddings; some indices require training. It does not specify a neural recommender architecture. The meeting itself raises this at 00:50:42. [Official documentation](https://faiss.ai/)
- **A changed input history is different from updated model weights.** This project needs to identify the actual parameter updates. Session state, in-context behavior, and continued training should not be conflated.
- **The proposed measurement is ranking quality.** Calling it DoS, AI-slop detection, counterfeit detection or a successful attack would require additional definitions and evidence.
- **Disjoint MF factors may not interfere.** If B updates no factors used by A, A's scores need not change. See EXPERIMENT.md for the assumptions behind this deduction.

At the time of the meeting reconstruction, the supplied materials did not evidence a completed implementation, selected dataset, measured baseline or finished deck. For subsequent A-only implementation and critique preparation, see the current [context](../CONTEXT.md) and [runbook](IMPLEMENTATION.md). Teammates' later work still needs confirmation.
