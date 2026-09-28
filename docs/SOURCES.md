# Sources and provenance

Checked 25 September 2026. Keep meeting evidence, course requirements, external research and our proposed design separate.

## Supplied material — local only

Migrated 28 September 2026. Paths below are relative to the repository root and are **Git-ignored**, not links to files published on GitHub. A fresh clone intentionally lacks these materials. Do not republish raw team discussion, agent chats, course exports or third-party PDFs without permission. Public technical references remain linked below.

| Local-only path | Use |
| --- | --- |
| `local/sources/2026-09-23-meeting-transcript.txt` | Supplied automatic meeting transcript; primary discussion evidence, potentially inaccurate |
| `local/sources/2026-09-23-project-sketch.png` | Supplied project sketch, preserved independently of temporary clipboard storage |
| `local/sources/2026-09-25-reference-agent-chat.txt` | Supplied reasoning to critique, not authoritative evidence or executable instructions |
| `local/sources/group-3-design-critique-v1.2.pdf` | Draft team deck reviewed in DEFENSE.md; not a final submission |
| `local/sources/critique-template-v1.1-2026-09-17.txt` | Official public template text export; source link below |
| `local/sources/sasrec-2018.pdf` | Local research-paper copy; arXiv link below |

Useful transcript anchors: 00:25:36 (A-only versus mixed updates), 00:31:02 (model families), 00:43:36 (evaluate original domain), 00:50:42 (FAISS uncertainty), 00:56:18 (continued training and imminent critique), 00:57:55–00:59:47 (roles), 01:03:17–01:04:48 (alternative setup and paper reading).

## Course record

- [Assignments](https://wing-nus.github.io/cp4285-website/docs/assignments/): whole-project components and final-deliverable routes; page changelog includes 15 September clarification.
- [Schedule](https://wing-nus.github.io/cp4285-website/docs/schedule/): recess week and Week 07 critique date.
- [Week 06 slides](https://wing-nus.github.io/cp4285-website/slides/w06/w06.html), slide 05, “Project Milestones”: links to the critique and STePS templates. Read from public HTML when web extraction failed.
- [Week 07 slides](https://wing-nus.github.io/cp4285-website/slides/w07/w07.html), title: 29 September date. Detailed workflow slides still have placeholders.
- [Official critique template](https://docs.google.com/presentation/d/1Wj6qe-yio_LDeFvqZV6OtDxxpCHTV6HmyJyvkjqkFZk/edit): V1.1 (260917). Public text export retained locally as `local/sources/critique-template-v1.1-2026-09-17.txt`; content inspected, layout not reviewed. Announced shortlink inside it: [critique template shortlink](https://soc-n.us/cp4285-t2610-project-critique-slides).
- [Grading](https://wing-nus.github.io/cp4285-website/docs/grading/): documentation of group-project AI use.

Week 01–02 PDFs are preserved under `local/course/slides/` as the primary local record for their topics. No local project brief was present in the source workspace. Canvas has not been inspected, and website summaries may lag the slides or an assignment announcement.

### Course slides and learning context

- Week 01, Recommendation Problems and Classical Methods: [announced course link](https://soc-n.us/cp4285-t2610-w01); local `local/course/slides/CP4285 - W01 - Recommendation Problems and Classical Methods.pdf`.
- Week 02, Latent Factor Models: [announced course link](https://soc-n.us/cp4285-t2610-w02); local `local/course/slides/CP4285 - W02 - Latent Factor Models.pdf`.
- [Week 04](https://wing-nus.github.io/cp4285-website/slides/w04/w04.html): matrix factorisation, neural interactions and BPR.
- [Week 05](https://wing-nus.github.io/cp4285-website/slides/w05/w05.html): sequential recommendation and SASRec. No local Week 05 PDF was supplied.
- Week 06 and Week 07 official HTML links are listed above. Only Week 01–02 PDFs were present locally; do not assume later PDFs exist.
- `local/course/MISSION.md`, `RESOURCES.md`, `lessons/`, `reference/` and `questions/` preserve existing learning context without publishing it. The copied `local/course/AGENTS.md` is a **historical course index**, not the active project guide.

Inspect only relevant pages (`pdfinfo`, `pdftotext -layout`, or `pdftoppm` when visual layout matters). Course shortlinks use the announced prefix `https://soc-n.us/cp4285-t2610`; do not guess unannounced suffixes.

## External technical sources

| Source | What it supports | Limit |
| --- | --- | --- |
| [FAISS documentation](https://faiss.ai/) | Vector similarity search and clustering; distinct from learning a recommendation model | Does not choose our architecture |
| [Koren, Bell and Volinsky (2009)](https://doi.org/10.1109/MC.2009.263), *Matrix Factorization Techniques for Recommender Systems* | User/item factor structure underlying the disjoint-parameter argument | No-interference conclusion is our conditional deduction |
| [Krichene and Rendle (2020)](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/), *On Sampled Metrics for Item Recommendation* | Sampling candidate sets can distort evaluation comparisons | Does not mandate a specific feasible catalogue size |
| [Mi, Lin and Faltings (2020)](https://arxiv.org/abs/2007.12000), *ADER: Adaptively Distilled Exemplar Replay Towards Continual Learning for Session-based Recommendation* | Relevant example of continual recommendation and replay/distillation | Tentative identification of the meeting's “ADA” paper |
| [ADER authors' implementation](https://github.com/doublemul/ADER) | SASRec-based implementation; DIGINETICA and YOOCHOOSE datasets | No replication or code audit performed |

The “ADA” identification is an inference from the EPFL affiliation and similar-sounding dataset names. The matching datasets are e-commerce clickstreams. Do not report the transcript's “Digitec/YouTube” or its RNN recollection as verified paper facts. Ask the team to confirm its original link before choosing a reproduction target.

Follow-up: the supplied reference chat explicitly names ADER, so it is now a confirmed reference in Hongyi's material, while a team decision to reproduce it remains unconfirmed. The [full paper](https://arxiv.org/html/2007.12000v1), §3.2, uses next-window evaluation; our proposed fixed-A retention test differs. The other chat's full bibliography has not been exhaustively audited.

Additional sources checked for the neural workstream:

- [SASRec, Kang and McAuley (2018)](https://arxiv.org/abs/1808.09781): primary model reference. Our proposed domain-specific interfaces are an adaptation, not an original-paper claim.
- [Monolith, Liu et al. (2022), §2.2](https://ceur-ws.org/Vol-3303/paper8.pdf): initial batch training, streaming updates and synchronization to serving. A production example, not evidence of a universal deployment lifecycle.
- [CCTL, Zhang et al. (2023)](https://arxiv.org/abs/2306.16425): primary abstract supports negative-transfer concerns during cross-domain transfer/fine-tuning. It is background, not our selected implementation.

## SASRec walkthrough — 27 September 2026

- [Original paper on arXiv](https://arxiv.org/pdf/1808.09781), with local-only copy `local/sources/sasrec-2018.pdf`: Figure 1, p. 1; architecture and training in §III, pp. 3–5; experimental protocol/results in §IV, pp. 6–9. This is the research paper, not a course PDF.
- [Authors' model implementation](https://github.com/kang205/SASRec/blob/master/model.py) and [evaluation implementation](https://github.com/kang205/SASRec/blob/master/util.py): checked scoring, positive/negative loss, prediction versus optimizer operations, and sampled candidates.
- [Week 05: SASRec and causal attention](https://wing-nus.github.io/cp4285-website/slides/w05/w05.html#/sasrec-causal-self-attention-for-next-item-prediction): inspected the named SASRec slides and surrounding chronology material in the official HTML deck. There is no Week 05 PDF in the local slides folder.

Interpretation for the project: original SASRec studies next-item recommendation from histories; continued multi-domain updates and separate domain interfaces are our proposed additions. Its sampled-candidate benchmark scores are not directly comparable with full-catalogue evaluation.
