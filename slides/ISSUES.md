# Slide issues

Review of the [team Google Slides deck](README.md) on **28 September 2026**. It was read through the deck's public text export and preview; nothing in the deck was edited. Slide numbers are positions in the deck on that date and will shift as it changes. Principle numbers (P1–P19) refer to [README.md](README.md#principles-for-making-slides).

- **Must**: wrong, contradicts another slide, out of date or missing a required part. Fix before submission.
- **Should**: hard for a classmate with no background to follow. Fix if time allows; otherwise move detail to speaker notes.

Strike through or delete an item once it is fixed in the deck, and note who fixed it.

## Across the deck

| # | Priority | Issue | Fix |
| --- | --- | --- | --- |
| D1 | Must | **Slide budget.** There are 22 slides, which is 20 once acknowledgements and references are excluded. That is the course maximum, and it includes the empty slide 7 and the version history (slide 22). | Fill slide 7 with the data introduction (D5). Replace slide 8 with neural slide N1. Hide the version history before exporting, which frees a place for N2. See [SLIDE-FLOW.md](../docs/SLIDE-FLOW.md#live-deck-review--28-september-2026). |
| D2 | Must | **No neural slides.** Slides 5 and 8 still describe the neural model as undecided, but SASRec is selected and built. | Add N1 and N2 from [SLIDE-FLOW.md](../docs/SLIDE-FLOW.md#paste-ready-neural-slides). Fix slides 5 and 8 (below). |
| D3 | Must | **Reviews are described as purchases.** Slides 10–11 say "bought", "purchases" and "last purchase". Each record is a star rating left on a date, not a purchase (P3). | ~~Say "reviewed" and "latest review".~~ Done on the pilot slides (randomwish, 28 Sep). Still open: say once, on the data slide, that reviews stand in for what people bought or watched. |
| D4 | Should | **Six names for one idea:** off-sample data, contamination, perturbation, attack (vector), distribution shift and "mixed in" (P7). | Use "mix in Movies & TV" everywhere; the pilot slides already do. Define it once: "100% = as many Movies & TV reviews as Electronics training reviews." |
| D5 | Should | **The dataset is never introduced.** Slide 9 gives sizes, but no slide says what Amazon Reviews 2023 is, what one record is, why only shoppers and products with five or more reviews are kept, or why these two categories (P3, P4). | Use slide 7 for this. Move slide 9's counts there, and add the 368K Electronics products and the five-review filter. |
| D6 | Should | **Metrics are named before they are explained.** Slides 4–5 list NDCG@10, Recall@K and MRR. Slide 11 explains NDCG@10 clearly. Recall and MRR are never explained (P2). | Explain all three the first time they appear, or name only NDCG@10 until slide 11 and explain the other two on slide 14. |
| D7 | Should | **Papers are referred to as if already read.** "Continual-learning, negative-transfer and domain-similarity literature" (slides 5 and 8) and "sampled-metric inflation (Krichene & Rendle)" (slide 14) mean nothing to a classmate (P5). | One plain sentence per idea, with a numbered citation. Drop fields the deck never uses. |
| D8 | Must | **References are placeholders.** [1] is described as "RNN-based recommendation over Diginetica and YouTube". If [1] is ADER, it uses SASRec on DIGINETICA and YOOCHOOSE. [2] is "EPFL-related work" and [3] has no details. No slide carries a numbered citation (P19). | Add full entries for SASRec (Kang & McAuley 2018), ADER (Mi, Lin & Faltings 2020) if still used, Krichene & Rendle (2020), the Amazon Reviews 2023 paper named on the [dataset homepage](https://amazon-reviews-2023.github.io/), and the classical model sources in [CLASSICAL.md](../docs/CLASSICAL.md). Entries for the neural papers: [DEFENSE.md](../docs/DEFENSE.md#verified-neural-references). |
| D9 | Should | **Private history leaks in.** "Replace a subjective 'authentic vs synthetic' label" (slides 3 and 6) refers to an earlier team idea the class never saw. | Remove it, or give it one sentence explaining why the framing changed. |
| D10 | Must | **The research question assumes the answer.** "Measure how recommendation quality degrades" (slides 2–3) and "measure ranking degradation" (slide 8) presume harm, but the Markov and popularity models did not change (P10). | Ask how quality *changes* as Movies & TV is mixed in. |

## Slide by slide

| Slide | Priority | Issue | Fix |
| --- | --- | --- | --- |
| 1 Title | Must | Label reads V1.2 (260926), but the deck has changed a lot since. | Bump the version label and add a matching version-history entry. |
| 2 Abstract | Must | Says pairing and mix levels "remain open pending … a quick validity test", but the pilot has run and 140% is the maximum. Also D10. | Update to the current state: what the pilot found, and what is still open. |
| 2 Abstract | Should | Terms left unexplained: continual-learning recommender, off-sample data, collaborative filtering / matrix factorization, contamination. | Plain wording: "a recommender that keeps training as new reviews arrive"; "mixing in reviews from another category". |
| 3 Motivation | Should | "Distribution shift" and "baseline domain" are unexplained. Also D9 and D10. | "The kind of reviews the model trains on changes, from Electronics only to a mix with Movies & TV." |
| 4 Task | Must | Lists 10%, 50%, 100% and 200% mix levels, but slide 9 says 200% is impossible (140% is all the Movies & TV data). | Use 0%, 10%, 50%, 100% and 140%, as slide 17 does. |
| ~~4 Task~~ | ~~Must~~ | ~~Says "Recall@K"; the results use Recall@10.~~ | Fixed (randomwish, 28 Sep): slides 4, 5, 14, 16 and 18 say Hit@10, the name decided in [issue #8](https://github.com/Hong-yiii/cp4285_reccomender_systems/issues/8). |
| 4 Task | Should | "Neural sequential / transformer recommender" and "schema compatibility" are unexplained. | Describe the neural model in one line (P1). Drop "schema compatibility"; the pilot settled it. |
| 5 Method | Must | A bullet is cut off mid-sentence at "where feasible (w". | Finish or delete the sentence. |
| 5 Method | Must | "SASRec-style where appropriate", but SASRec is selected and built. 200% appears again. | "Neural: SASRec, the original design, rebuilt in PyTorch." Use 140%. |
| 5 Method | Must | "Decision gate: run a quick attack-vector pilot … before locking the final dataset plan." The pilot has run. | Say what the pilot decided, or what it left open. |
| 6 Progress: scope | Must | Says Hongyi "coordinates framing / tracking", but the schedule (slide 18) assigns Hongyi the neural model. | Hongyi: neural model. Check every role against the schedule. |
| 6 Progress: scope | Should | "Sprint status … assessment: satisfactory" is a self-grade with no evidence. | Replace with what was done and what it shows. |
| 7 Data: exploration? | Must | Empty slide with a question-mark title. | Use it for the data introduction (D5). |
| 8 Progress: technical plan | Must | Out of date. It weighs "SASRec / RNN / two-tower options" and a risk of "whether the attack vector is strong enough"; the pilot answered the second. | Replace with neural slide N1 and move the remaining risks to N2. |
| 9 Pilot: how we tested it | Should | "All Movies & TV data joins as new shoppers" is a design choice, but the slide never says why, or what it means for the results. | Add a line: "so a model that keeps separate entries per product cannot be affected; only fixed-size models can be." |
| ~~10 Pilot: three models~~ | ~~Must~~ | ~~"What people most often bought after your last item". See D3.~~ | Fixed (randomwish, 28 Sep): "reviewed right after your latest product". |
| 10 Pilot: three models | Should | "QR model" is a name with no meaning to the class. | "Compressed Markov chain (QR)". Explain k as the number of shared patterns it can keep. |
| ~~11 How models work~~ | ~~Must~~ | ~~"Count purchases" and "last purchase". See D3.~~ | Fixed (randomwish, 28 Sep). The Markov card also says the reverse direction counts half, as in `transitions(back=0.5)`. |
| 11–15 | Should | The QR size changes without warning: k = 256 on slide 11, rank 64 on slide 13, 256 on slides 12 and 15. | Put "pilot, 64 patterns" or "full study, 256 patterns" on every QR number. |
| 12 Why only some get worse | Should | The 29% figure comes from the full study (256 patterns, 100% mix), but it sits in the pilot section next to the rank-64 pilot. | Label its source, or move it next to slide 15. |
| 13 Pilot result | Should | Calls the popularity model "Benchmark"; slides 15–16 call it "Popularity". "(validation)" is unexplained. | Use "Popularity" throughout. Say "20K Electronics shoppers held out for tuning". |
| 14 Evaluation protocol | Must | This is the classical protocol only. The neural pilot holds out each shopper's last review before 2021 and trains on later reviews as they arrive, so its scores are not comparable yet (P13). | Title it for the classical models or add a neural line. Do not compare scores until both use one split. |
| 14 Evaluation protocol | Should | Dense jargon: "no sampled negatives", "paired bootstrap 95% CIs (1,000 resamples) and p-values", "rank × (1 + mix level)", "sampled-metric inflation". | Use glossary wording. Move the resampling detail to speaker notes. |
| 14–16 | Must | The protocol promises three seeds; the results come from seed 0 only. Slide 16 says so, but slide 15 does not. | Put "one run of three so far" on slide 15's footnote as well. |
| 15 Results so far | Should | The title and footer never say these are classical models only. | Say "classical models"; there are no neural results yet. |
| 16 All three metrics | Should | A six-row, six-column table is hard to read on a projector. "* 95% paired-bootstrap CI excludes 0" is jargon. | Highlight the 100% column or keep only the two QR rows. Write "* unlikely to be chance (95% confidence interval excludes 0)". |
| 17 Resources | Should | Uses "Contaminating Data" and "clean, unshifted environment" (D4). Lists no software. Compute is "will use" with no confirmation. | Use one term. Add PyTorch, the repository link and the original SASRec source. Confirm cluster access before claiming it. |
| 18–19 Schedule | Must | Out of date. Week 7 still plans the pilot, which has run. Week 8 plans to implement SASRec, which is built and tested on made-up data. Week 10 assigns Hongyi "Tracking / Analysis". | Mark completed work. Week 8 becomes the first real Electronics run for SASRec. Align roles with slide 6. |
| 21 References | Must | See D8. | — |
| 22 Version history | Must | Only a V1.2 entry. | Add an entry, then hide the slide before export (D1). |

## Fixed after this review

Found and fixed on 28 September by randomwish, after the deck gained slides this review did not cover:

- **How the matrix is modelled** showed a shoppers × products matrix, but the QR model factorises the product → next product table (`MarkovQRSVD`). It now shows that table; the zero corners are explained by no timeline linking the two domains.
- **Results so far** read "with the formula ((rank × (1 + mix level))". Now "when QR's rank grows with the data: rank × (1 + mix level)".
- **Evaluation protocol** defines Hit@10 where it is first listed (partly addresses D6).

## Open team decisions behind some issues

- **Mix-level definition.** The deck measures Movies & TV volume against Electronics (0–140%). The neural code fixes the total number of training updates and sets Movies & TV's share of them. 140% of Electronics is a 58.3% share. Agree on one before the neural slides quote mix levels. See [DEFENSE.md](../docs/DEFENSE.md#resolve-the-mixture-denominator).
- **One evaluation split.** The classical models hide each shopper's last review. The neural pilot hides each shopper's last review before 2021. Pick one before comparing scores.
- **Roles.** Slides 6 and 18–19 disagree. Each teammate confirms their own line.
