# Dataset assessment for the continued-training experiment

28 September 2026. Documentation and primary-paper review, not an empirical audit of downloaded interactions. No datasets have been selected by the user in this assessment. Scope: the neural workstream, with classical-model compatibility considered where it affects the group comparison.

## Recommendation

Prefer Amazon Reviews 2023 Electronics and Movies_and_TV as the provisional primary pair for the proposed category-mixture experiment. They offer a common collection/schema and event type while varying product domain. Verify usable chronological sequences and user overlap before locking them. Electronics versus Clothing is also possible but has no demonstrated advantage for this project merely because a related paper uses an Amazon pair.

DIGINETICA/YOOCHOOSE are suitable session recommendation resources. Treat their mixture as a separate cross-dataset/retailer stress test, with collection differences as part of the treatment. For a simpler chronological continual-learning pilot, use one of them over successive time windows; this is closer to ADER's use of the data. It answers a different question from changing category mixture.

## Verified facts and source interpretation

- Amazon's official schema provides reviewer ID, product/parent-product ID, rating and review timestamp. These are reviews, not complete browsing, purchase or viewing histories. [Dataset documentation](https://amazon-reviews-2023.github.io/)
- The official deduplicated 5-core ID files have approximately 15.5M Electronics ratings and 7.4M Movies_and_TV ratings. They retain `user_id,parent_asin,rating,timestamp`. Prebuilt last-out and absolute-time splits exist. [Processing documentation](https://amazon-reviews-2023.github.io/data_processing/5core.html)
- ADER uses SASRec in separate experiments on DIGINETICA and YOOCHOOSE, with weekly and daily update windows respectively. It does not train on DIGINETICA and inject YOOCHOOSE. Its evaluation predicts the next window, whereas our proposed original-domain test stays fixed. [ADER §§3.2, 4.1–4.3](https://arxiv.org/html/2007.12000v1)
- YOOCHOOSE's click file records session ID, timestamp, item ID and category. A session identifier is not a persistent user identifier. [Dataset authors' paper](https://dbs-home-page.s3.amazonaws.com/dbs/papers/recsyschallengeyc2p.pdf)
- ADER's implementation reads `train-item-views.csv` for DIGINETICA and `yoochoose-clicks.dat` for YOOCHOOSE through separate preprocessing paths. Do not assume compatible integer IDs or timestamp conventions. [Authors' repository](https://github.com/doublemul/ADER)
- CCTL, one reference already in these project notes, uses Amazon Books and Movies & TV, including a shared-user population. It focuses on improving target-domain CTR prediction using source information. This supports cross-domain motivation, not a reproduction claim for our continual-retention experiment or this particular Amazon release. The teammate's exact intended paper has not been identified. [CCTL §3.1.1](https://arxiv.org/html/2306.16425v1)

## Comparative judgment

The following are design judgments based on the documented structures, not measured dataset findings.

| Dimension | Amazon Electronics / Movies & TV | DIGINETICA / YOOCHOOSE |
| --- | --- | --- |
| Prediction task | Next reviewed item, or next positively rated item after an explicit rating filter | Next item within a browsing session |
| Neural fit | Suitable if enough ordered review histories survive the split | Natural session-sequence input, with model and sampling settings adapted to short sessions |
| Interpretation of shift | Category change within a common platform and event representation | Change of dataset/retailer, catalogue, traffic and collection conditions |
| Identity | Measure reviewer intersection using original IDs; measure item overlap as well | No established cross-dataset user/item identity mapping; keep namespaces distinct |
| Persistent-user MF fit | More natural, particularly if shared users remain sufficiently active | Requires special treatment for unseen sessions; session IDs cannot simply substitute for returning users |
| Main risk | Sparse histories, review-time proxy, compute/candidate scale, domain-specific populations | Dataset provenance confounds, session cold start, mapping/chronology mismatch |

Two Amazon domains still differ in users, popularity, sequence length and time coverage. They do not isolate semantic category distance. An ID-only SASRec does not explicitly read the words “Electronics” or “Movies”; it receives learned item representations and interaction patterns.

## Mechanism and group-comparison implications

The neural mechanism remains the proposed shared Transformer with compatible domain item interfaces. B can affect A through shared parameter updates even without common users. Interleave valid examples, preserving each real history. Never fabricate cross-user sequences or force unrelated item IDs to coincide.

Classical comparison needs a separate parameter-sharing audit. In ordinary dot-product MF, with completely disjoint users and items, domain-restricted negative sampling, fixed A candidates and no updates/decay to A factors, B training has no direct route to change A's scores. Global shared parameters, cross-domain negative sampling, regularization or optimizer behavior can change this conclusion. A flat MF result may reflect structural isolation rather than superior resistance to forgetting.

For Amazon, if genuine shared reviewers occur, retaining the same user factor across domains gives B updates a route to affect that user's A recommendations. This is a valid proposed design but the overlap and post-split histories must be measured. Do not invent links or silently restrict to a different cohort for each model. If using a shared-user cohort, keep the evaluation cohort comparable and report the selection bias. The shared neural encoder does not itself require such overlap.

For session data, consider item-kNN or session-kNN as a classical comparator, subject to team agreement, or specify a legitimate inference procedure for MF on unseen sessions using only the observed prefix. Training a user vector on a held-out next item would leak the answer.

## Feasibility gate before final selection

1. Inspect both official files and record version, checksum, schema and timestamp units. Prefer ID-only files for the first prototype. Choose a bounded cohort/time window that fits actual compute, preserving each selected history rather than randomly dropping rows.
2. Count users, items, unique interactions and sequence lengths after the intended preprocessing and temporal split. Report how many histories support initial training, later A updates and held-out prediction. A globally 5-core dataset does not guarantee five usable training events per user at the initial cutoff.
3. Measure user overlap and item overlap explicitly, including parent-product deduplication and repeated/tied timestamps. Decide whether real overlapping users share factors. Never match independently reindexed integer IDs across domains.
4. Define the feedback target. All review events can support next-reviewed-item prediction, but low ratings are not endorsements. A positive-rating-only target is another choice that requires refiltering and recounting histories. A review timestamp is not a purchase or viewing timestamp.
5. Build initial-training and subsequent-update pools in chronological order within each domain. Document common calendar bounds if claiming time alignment. Keep validation/test events out of update targets and prefixes. Use a fixed A retention anchor as defined in CONTINUED-TRAINING.md and a separate B adaptation evaluation.
6. For a strict temporal study, derive eligibility/filtering decisions from the allowed training period. Using the provided globally filtered 5-core files is acceptable for a labeled offline benchmark pilot, but conditions the population on full-history activity and should be disclosed.
7. Define vocabulary growth/cold-start handling. Begin with a familiar-item A evaluation cohort and count excluded cases. Size the catalogue and evaluator before promising full-catalogue ranking. If candidates are sampled, fix them across runs and do not compare their scores directly with full-catalogue results.
8. Check available B supervised targets against each requested update budget. More B than the full available stream requires shrinking the A budget, changing the ratios, or explicitly declaring resampling. Repeating B is not additional unique incoming data.
9. Run A training/save/reload evaluation, then A-only continuation before introducing B. Verify B learns under mixed/B-only conditions. Select data based on suitability and feasibility, not on which pair forces the desired decline.

## Ratio and framing corrections

The proposed 10%, 50%, 100%, 200% must specify a denominator. If these mean `n_B/n_A`, the B fractions of the combined stream are 9.1%, 33.3%, 50%, 66.7%. A 200% fraction of a combined stream is impossible. Fixed A volume plus extra B changes total data/update count under equal epochs. Fixed total updates changes A exposure. The earlier proposal in CONTINUED-TRAINING.md uses the latter and documents the limitation.

An A-only dataset is a baseline, not proof of a clean or stationary environment. Different-domain reviews/clicks are not inherently contamination, adversarial behavior or low-quality data. Use “second-domain interactions” and “controlled distribution shift” unless there is a specified attack objective and attacker-controlled input. Measure performance change and allow improvement or no measurable effect.

## Suggested slide wording

> We propose Amazon Reviews 2023 Electronics as the original domain and Movies & TV as the second domain. We construct timestamp-ordered reviewer histories and study next-item recommendation. Following initial Electronics training, we continue updating a shared SASRec-based model with controlled mixtures of subsequent interactions. We evaluate Electronics retention on a fixed held-out task and Movies & TV adaptation. Final selection depends on sequence coverage, identity overlap and compute feasibility.

If using all ratings, explain “next-item” here as the next reviewed item. This draft describes a proposal, not downloaded-data findings or a team-approved choice.
