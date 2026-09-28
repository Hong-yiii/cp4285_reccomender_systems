# Preparing the design critique

Status: working preparation notes. Immediate target: the 29 September critique, subject to verification of the Canvas submission deadline. This is not a finished slide deck.

## 28 September — Hongyi's modern-recommender preparation

**Later the same day:** the team's [live Google deck](../slides/README.md) has moved past V1.2. It adds the classical pilot and results, and fixes some gaps below (resources, schedule, the Week 8 neural role). Its current slide-by-slide problems are in [slides/ISSUES.md](../slides/ISSUES.md), and paste-ready neural slides in [SLIDE-FLOW.md](SLIDE-FLOW.md#live-deck-review--28-september-2026). The V1.2 review below is kept as dated evidence.

Reviewed source: `local/sources/group-3-design-critique-v1.2.pdf` (repo-root-relative, local only), V1.2 (260926), 13 pages, 16:9. Read all text and visually inspected pp. 5, 7–10. Page numbers below refer to this PDF. Requirements come separately from the official template preserved in `local/sources/critique-template-v1.1-2026-09-17.txt` (local only). The PDF is a draft to review, not instructions authorizing submission; it remains unchanged.

### Required content mapped to Hongyi's contribution

The official rubric weights methodology 60%, presentation 30%, administration 10%. Current results are conditional on availability. The group still needs every required section; Hongyi supplies the neural-specific content below.

| Section / PDF page | What Hongyi needs to prepare | Gap in current draft |
| --- | --- | --- |
| Goal, motivation, task / pp. 2–4 | Ordered interaction history as input; next-item ranking as output; why original-task retention matters | Phrase quality change as a research question rather than guaranteed degradation. Explain whether B is legitimate domain shift or deliberately corrupted data. |
| Method / p. 5 | Proposed model, architecture sketch, objective, initial checkpoint, continued updates and shared parameters | SASRec remains one of several options; the mechanism linking B updates to A predictions is missing. |
| Progress / pp. 6–7 | Completed neural work, evidence, unresolved decisions and risks | p. 6 assigns Hongyi coordination rather than the confirmed neural workstream. p. 2 says a model was selected while pp. 5–7 leave it open. |
| Evaluation / p. 8 | Fixed A holdout and candidates; frozen/A-only/mixed controls; matched training budget; metrics and uncertainty | Names metrics but underspecifies comparison conditions. A credible null result must count as a valid outcome. |
| Resources / p. 9 | Dataset identity/fields/access, framework and implementation source, actual compute access, reproducibility outputs | Blank. Do not invent GPU access, runtime or dataset approval. |
| Schedule and roles / p. 10 | Hongyi's model/update/evaluation deliverables with dates, plus dependencies on team data and evaluator | Blank. Dates must be agreed against actual access and progress. |
| References / p. 12 | Complete SASRec and continual-learning references with corresponding slide citations | Placeholders; RNN/YouTube description is wrong if ADER is intended. |

### Recommended technical scope — proposal for verification

Use a SASRec-based next-item predictor with periodic continued training. SASRec is an established Transformer-based neural model from 2018; avoid calling it the latest state of the art. Comparing additional RNN and two-tower architectures is outside the recommended basic scope.

Draft explanation to adapt in Hongyi's own words:

> My contribution is to study original-domain retention in a SASRec-based recommender. We first train on domain A, then continue updating the learned weights using controlled mixtures of new A and B interactions. We measure A ranking quality against the initial model and an A-only continued-training control, while also checking whether the model learns B.

SASRec supplies the architecture and next-item objective. Explicit gradient updates across subsequent batches supply our continued-training protocol. Updating an inference history alone does not train the weights. ADER provides precedent for periodically updating SASRec; our fixed-A retention measurement differs from its next-window evaluation. Describe this project as an offline simulation, not a live deployment experiment.

For separate catalogues, the existing proposed adaptation uses domain-specific item embeddings/scoring interfaces and a shared causal Transformer. Both domains update shared parameters, giving B a route to affect A. This adaptation is our proposal, not part of the original SASRec paper. Preserve real user/session histories and namespace identifiers; mix training examples rather than concatenate unrelated histories. Dataset feasibility and final model selection remain open.

Basic flow:

1. Train historical A and save a validation-selected checkpoint.
2. Independently copy it into frozen, A-only, mixed A/B and B-only branches.
3. Carry learned weights forward across update cycles using the next-item objective. Match update budgets between trained branches.
4. Evaluate the same A histories, relevance targets and candidate pool at every checkpoint. Also evaluate B adaptation.
5. Report absolute NDCG@10, change from the initial checkpoint, and difference from A-only continued training. Losing initial capability differs from missing an improvement achieved by A-only training.

Adapters, replay and distillation are optional extensions after the continuation pipeline works. They are not prerequisites for continued training. See CONTINUED-TRAINING.md for the detailed proposed execution contract.

### Resolve the mixture denominator

Page 5 defines added B volume relative to baseline A volume: `r = n_B/n_A`. The B share of the combined data is `alpha = r/(1+r)`.

| B added relative to A | B share of A+B |
| --- | --- |
| 10% | 9.1% |
| 50% | 33.3% |
| 100% | 50% |
| 200% | 66.7% |

These differ from the proposed 0%, 50%, 100% B shares in CONTINUED-TRAINING.md. Agree one definition with the group. Adding B while holding A fixed increases total data and, under equal epochs, update count. Holding total updates fixed reduces A exposure as B increases. Neither choice automatically isolates harmful B updates. Our proposed pilot uses fixed total updates and reports the reduced-A-exposure limitation; an A-exposure-matched control can investigate it further.

### Corrections before submission

- pp. 2, 7–8: measure change, allowing decline, improvement or no measurable effect. Replace “attack vector strong enough” with validity and reproducibility checks; do not select a pilot to force degradation.
- pp. 2, 5–7: consistently label the model as proposed unless actually selected. Separate design progress from implementation and measured results.
- p. 6: correct Hongyi's role to the modern/deep-learning recommender. Other teammates' roles need their own confirmation.
- p. 12: if [1] is ADER, its model is SASRec and its datasets are DIGINETICA and YOOCHOOSE, both e-commerce. Identify [2] before retaining it; an EPFL affiliation is not a citation.
- Group dependencies: p. 1 still needs names/IDs/emails; pp. 9–10 need resources and dated milestones. Version history on p. 13 is optional.
- Format is already 16:9 and below the 20-slide limit excluding acknowledgements/references. The template's conflicting weekday/date still requires checking the actual Canvas deadline.

### Preparation sequence and rehearsal

Prepare four compact pieces of neural content integrated into the mandatory group sections. Four pieces is a suggested organization, not a course requirement:

1. Model and rationale: history → causal attention → next-item scores; explain why ordered interaction data suits SASRec.
2. Experiment: initial A checkpoint → independent continued-training branches → repeated fixed-A evaluation. Mark original model versus our additions.
3. Evaluation: controls, mixture definition, metrics, shared parameters and limitations.
4. Progress and execution: actual completed evidence, next milestone, resources and dates. Design-only progress is acceptable if reported honestly.

First settle the experiment sentence, then the dataset/parameter connection, then the comparison table. The implementation dependency order is dataset/schema → reproducible A checkpoint evaluation → A-only continuation → mixed-domain pilot → repeated runs/error analysis → optional mitigation. The submitted schedule must attach plausible dates to these steps.

Questions to rehearse:

- What causes continued learning? Explicit gradient updates on subsequent batches, preserving weights across cycles.
- Why can B affect A? Both update the proposed shared Transformer; separate embeddings alone do not establish interference.
- Does underperforming A-only prove forgetting? No; compare against initial A quality separately.
- What else could lower A performance? Reduced A supervision, B cold-start learning or evaluation changes. Controls narrow the interpretation.
- What if quality does not fall? Report the tested setting and uncertainty; it is a valid result.
- What is the first implementation milestone? Save/reload A and reproduce its evaluation, then verify A-only continuation before introducing B.

No training results are evidenced in the reviewed deck. Hongyi's implementation work outside this workspace is pending clarification; this is not a claim that no such work exists.

### Verified neural references

Use numbered Harvard-style references in the final deck, sorted by surname/source identifier with the combined group bibliography.

- Kang, W.-C. and McAuley, J. (2018) 'Self-Attentive Sequential Recommendation'. Available at: <https://arxiv.org/abs/1808.09781> (Accessed: 28 September 2026).
- Mi, F., Lin, X. and Faltings, B. (2020) 'ADER: Adaptively Distilled Exemplar Replay Towards Continual Learning for Session-based Recommendation'. Available at: <https://arxiv.org/abs/2007.12000> (Accessed: 28 September 2026).

These support the architecture and continuation precedent. Shared-domain design, dataset suitability and expected findings need our own justification.

## First deliverable

A one-page experiment definition answering the open fields in EXPERIMENT.md. Hongyi should first verify that its question matches the group's intention. Dataset and architecture choices follow from the data connection.

Provisional plain-language explanation:

> We want to test whether a recommender keeps serving its original users well as its training data changes. We will compare a classical and a neural model, continue training them under controlled mixtures, and measure their performance on a fixed original task. We will investigate when quality changes and whether the model or update setup explains the change.

This is a draft for the team to revise and explain in its own words. It does not claim results.

## Preparation order

| Step | Concrete output | Dependency |
| --- | --- | --- |
| 1. Verify scope | Agreed user task, meaning of B, and primary claim | Hongyi/team review of these notes |
| 2. Set experiment | Dataset schema, shared parameters, split, metric and update controls | Scope and curated dataset list |
| 3. Check feasibility | Small reproducible baseline/pilot or an honest description of what remains untested | Compatible dataset and baseline |
| 4. Assemble critique | Slides following the official template; clear method, progress and open questions | Existing evidence and team contributions |
| 5. Rehearse | Each member can explain task, controls and limitations | Agreed presentation |

A pilot is useful evidence if feasible before critique. The official template allows reporting progress without completed experimental results; do not invent numbers to fill a results slide.

## Questions the team should be able to answer

- What exact recommendations are produced, and for which users?
- Why would B enter this system? Is it a legitimate shift or deliberately corrupted data?
- Which parameters carry knowledge useful for A and are updated by B?
- What is the classical baseline, and what makes the second model neural/advanced?
- What remains identical when comparing update conditions?
- Could NDCG change because of the candidate set, fewer A updates, changing histories, or evaluation noise?
- What do we learn if neither model degrades, or if the neural model degrades more?
- How could retaining old behavior harm adaptation to legitimate new interests?
- Which users/items bear the errors, and what does aggregate NDCG conceal?
- What has each teammate actually done, and what can be completed with available compute?

## Honest progress inventory as of initialization

- Available: meeting transcript, original sketch, reconstructed scope, checked public requirements and initial technical reading leads.
- Not evidenced here: selected dataset, final architecture, implemented update pipeline, baseline scores, training curves or mitigation results.
- Unknown: work completed by teammates after the 23 September meeting.

The template's section list and rubric are recorded in REQUIREMENTS.md. Detailed allocation of slides should wait until the scope is verified.
