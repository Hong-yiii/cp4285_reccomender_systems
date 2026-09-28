# Slides

**The team deck lives in Google Slides: [CP4285 Project Design Critique, Group 3](https://docs.google.com/presentation/d/1u8LnXJsb2AvCALW7iLqudzy6rRwcIICo70Bf3ripuHI/edit).** It is the canonical, team-owned deck. Edit it there; this folder keeps no copy.

- Known problems in the current deck, slide by slide: [ISSUES.md](ISSUES.md).
- Paste-ready neural slides and the full neural narrative: [SLIDE-FLOW.md](../docs/SLIDE-FLOW.md#live-deck-review--28-september-2026).
- Submission format, rubric and the unresolved deadline: [REQUIREMENTS.md](../docs/REQUIREMENTS.md). In short: 16:9 PDF through Canvas, at most 20 slides excluding acknowledgements and references.
- Questions to rehearse: [DEFENSE.md](../docs/DEFENSE.md).
- Do not commit exported deck PDFs or screenshots. The deck contains team names and IDs.

The six-slide HTML narrative that used to live here was retired on 28 September 2026. Its story and diagrams remain in [SLIDE-FLOW.md](../docs/SLIDE-FLOW.md); the HTML is in Git history (`git show 79108ff:slides/index.html`).

## Who the deck is for

The class reads this deck, and it is submitted as a PDF, so every slide must work without a speaker. Assume the reader:

- takes CP4285, and knows what a recommender does and what a ranked top-10 list is;
- has **not** used SASRec or any other model we use;
- has **not** seen Amazon Reviews 2023 or how we process it;
- has **not** read the papers we cite.

Easy to follow does not mean vague. Every claim still says what was measured, on what data, compared with what, and how sure we are.

**The classmate test:** hand a slide to someone from another group. Without help, can they say in one sentence what it claims and why they should believe it? If not, revise it.

## Principles for making slides

### Explain before you use

1. **Describe, then name.** Write "a model that reads a shopper's past products in order and guesses the next one (SASRec [1])", not "SASRec" alone. Once explained, the short name is fine.
2. **Explain every technical term where it first appears, or replace it with plain words.** Use the [glossary](#plain-wording-for-terms-we-use) below. If a term needs more than one line, it belongs in the speaker notes, not on the slide.
3. **Introduce the data before any result uses it.** Say what one record is: a shopper's star rating of a product, with a date. It is not a purchase, so write "reviewed", not "bought". Say which two categories we use and why, and how many shoppers and reviews each has. Call them Electronics and Movies & TV, not A and B.
4. **Explain each data step by its purpose.** "We keep only shoppers and products with at least five reviews, so everyone has a history to learn from", not "5-core".
5. **A citation says where an idea came from; it does not explain it.** Give one sentence of what we take from each paper. "Prior work kept retraining this model on new data [2]; we do the same, but keep testing on the old data."
6. **Carry abstract ideas with one running example**, such as one shopper who reviewed a phone, then a case, then a cable. Reuse the same example across slides.
7. **One word per thing, on every slide.** Say shopper (not user, reviewer or customer), product (not item), and "mix in Movies & TV" (not contamination, perturbation, attack or off-sample data).

### Rigour in plain words (methodology is 60% of the grade)

8. **The title is the claim, not the topic.** "Only size-limited models got worse" tells the reader what to see; "Pilot results" does not. Someone reading only the titles should follow the argument.
9. **Mark each claim as built, measured or planned.** "The neural model runs end to end on made-up test data" is built. "The compressed model loses 12% at a 100% mix" is measured. "Movies & TV training for the neural model" is planned. Never let a plan read as done.
10. **Ask about change, not guaranteed damage.** Improvement or no change is a valid result; the Markov line is flat. Keep "degrades" out of the research question.
11. **Every number answers: out of what, compared with what, and which way is better.** "0.0110, down 12% from 0.0125 with no Movies & TV; higher is better." When a scale looks surprising, say why. An NDCG@10 of 0.012 looks tiny, but it comes from ranking all 368K products.
12. **Every result carries its fine print and one limit.** Fine print means data, number of shoppers, number of runs and confidence interval, in a footnote line. The limit is one plain sentence on what the result does not show. For example: "Movies & TV shoppers are treated as new people, so only models with a fixed size can be affected."
13. **Real numbers only, compared like with like.** Never show numbers from made-up test data or "expected" curves as results. An empty results slide is better than an invented one. Do not put classical and neural scores side by side until both use the same split, candidates and metric.

### Readable visuals

14. **One idea, one visual, little text.** Use a chart, a diagram or three big numbers, not all three. Keep text blocks to four lines or fewer.
15. **Charts read without a legend.** Label lines at their ends and say which direction is better on the axis. Use one highlight colour for the thing being discussed and grey for context. Keep each category's colour the same on every slide.
16. **Every big number has a caption, and every arrow means one thing.** Say what the number counts ("of Electronics shoppers also reviewed Movies & TV"). In a flow diagram, arrows carry data forward. Arrows converging on "same test" mean shared evaluation, not merged models.

### One deck, four authors

17. **One owner per slide.** Put the owner's name in the speaker notes. Comment on a teammate's slide rather than rewriting it.
18. **Say the same thing everywhere.** Mix levels, model names, model sizes, roles and dates must match across slides. Avoid hard-coded slide numbers in cross-references, or recheck them after every reorder.
19. **Every citation resolves.** Each numbered reference on a slide has a complete entry on the References slide, and each entry is cited somewhere.

## Plain wording for terms we use

Use these on first appearance, or instead of the term. Check them against the source before relying on them for a new claim.

| Term | Plain wording |
| --- | --- |
| Review / rating | One record: a shopper gave a product 1–5 stars on a date. Not a purchase or a viewing. |
| Amazon Reviews 2023 | A public collection of Amazon product reviews, split into product categories; we use Electronics and Movies & TV. Cite the paper named on the [dataset homepage](https://amazon-reviews-2023.github.io/). |
| 5-core | The version of the data that keeps only shoppers and products with at least five reviews each. |
| Mix level, e.g. 100% | How much Movies & TV data is added, relative to the Electronics training data; 100% means an equal amount. |
| Popularity model | Recommends the same most-reviewed products to everyone. The simplest yardstick. |
| Markov chain | Recommends what people most often reviewed right after your latest product. |
| QR model / rank k | A compressed Markov chain: it squeezes the "what comes next" table into k shared patterns. QR names the maths used; k is fixed unless stated. |
| SASRec | A neural model that reads a shopper's past products in order and predicts the next one [Kang & McAuley 2018]. |
| Self-attention | When guessing, the model decides how much each earlier product should count. It can look only at earlier products, never later ones. |
| Leave-last-out | For each shopper, hide their latest review for the final test and their second-latest for tuning; train on the rest. |
| Validation vs test | Validation: hidden data used to choose settings. Test: hidden data used only to report the result, never to choose anything. |
| NDCG@10 | Rank all products for each shopper and find the hidden one. Score 1 if it is first, 0.63 if second, less further down, 0 outside the top 10. Average over shoppers. Higher is better. |
| Recall@10 | Share of shoppers whose hidden product appears in their top 10. |
| MRR | Average of 1 ÷ the hidden product's rank: 1 if first, 0.5 if second, and so on. |
| Full ranking | We rank every product, not the right one plus a small random sample, which would inflate scores [Krichene & Rendle 2020]. |
| Checkpoint | A saved copy of the model at one point in training. |
| Continued training | Keep training an already-trained model on newer data, starting from its saved state. |
| Seed / run | The random starting point of a run. Repeating with different seeds shows how much results vary by chance. |
| 95% confidence interval | The range the true change plausibly lies in, estimated by re-drawing the test shoppers many times. If it excludes 0, the change is unlikely to come from which shoppers happened to be tested. |

## Before exporting the PDF

- [ ] Every item in [ISSUES.md](ISSUES.md) is fixed or deliberately accepted by the team.
- [ ] The classmate test passes: someone outside the group read the PDF and could restate each slide's claim.
- [ ] At most 20 slides excluding acknowledgements and references. Hide the version-history slide and any empty placeholder slides before export.
- [ ] No template instruction text, truncated sentences or "?" titles remain.
- [ ] Title, abstract, motivation, task, method, progress, evaluation, resources, schedule/roles, acknowledgements and references are all present.
- [ ] Version label and version history reflect the submitted content.
- [ ] Each speaker can explain every slide they own and answer the rehearsal questions in [DEFENSE.md](../docs/DEFENSE.md).
- [ ] File → Download → PDF, then open the PDF and check 16:9, fonts and charts.
