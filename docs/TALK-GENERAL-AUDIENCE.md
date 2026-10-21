# Talk: teaching a recommender new tricks without forgetting old ones

A 12-minute talk on the LLM workstream, for a college-educated listener who is not a maths or computer-science major. It follows the [slide principles](../slides/README.md#principles-for-making-slides):
- describe, then name;
- one running example;
- "shopper" and "product" throughout;
- each title is the claim;
- each claim is marked **built**, **measured** or **planned**.

Numbers come from [PESO-REPLICATION.md](PESO-REPLICATION.md) and the committed reports. Status is as of 21 October 2026; every claim below is built or measured.

**Running example.** One shopper reviewed a beginner guitar, then a pack of strings, then a clip-on tuner. What should we show them next?

---

## Slide 1. Online shops must keep learning without forgetting their regulars

- **Visual:** a timeline of months. New products and new shoppers arrive each month, and the recommender gets an update arrow each month.
- **On the slide:** *Retraining from scratch every month is slow and expensive, so shops update their recommender instead. Does an update make it worse for the shoppers it already knew?*
- **Note:** in our data the updates are four later time periods, not literal months. The talk says "month" for simplicity.

**Say:** Think of a shop assistant who learns this season's stock every month. That's useful, unless they start forgetting what the regulars like. AI systems do this too, and badly enough that it has a name: *catastrophic forgetting*. Our project asks how much of it happens in product recommenders as they're updated. We're careful not to assume the answer: getting worse, staying the same and getting better are all possible.

## Slide 2. We change one thing at a time and keep the test fixed

- **Visual:** one saved model branching into versions that differ in exactly one setting. Every arrow converges on *the same hidden test reviews*.
- **On the slide:** *Same starting model · same test shoppers · one difference per comparison.*

**Say:** It's like testing a recipe. If you change the flour and the oven at the same time, you learn nothing. So every comparison we make differs in one setting only, and every version is graded on the same hidden reviews it never trained on. A review here means a shopper gave a product 1–5 stars on a date; it's not a purchase. Our team does this with simpler models on Amazon Electronics and Movies & TV. This talk covers the newest kind of recommender: one built on an AI language model.

## Slide 3. A language model can recommend by writing a product's "address"

- **Visual:** the shopper's history as text (guitar → strings → tuner). An arrow leads into the model, which writes four short codes, like *department → aisle → shelf → spot*.
- **On the slide:** *Each product gets a four-part code. The model writes the next product's code one part at a time.*

**Say:** Language models write text one word at a time. To make one recommend products, the PESO paper we built on (ICLR 2026) gives every product a four-part code, like an address. The first part picks a broad group, say tuning gear. Each later part narrows it, down to one exact product. The model reads the shopper's history as text and writes the address of the product to show next.

We use the paper's own data: Amazon reviews of musical instruments, 53,509 products. The model is Meta's Llama 3.2, with about a billion adjustable numbers.

**Built:** we reran the authors' public code end to end.

## Slide 4. Updates are cheap: sticky notes on a textbook

- **Visual:** a thick textbook (frozen) with small sticky notes attached (trained), plus a separate dictionary of product words (also trained).
- **On the slide:** *Each update rewrites the sticky notes (about 6 million numbers), not the textbook (about a billion).*

**Say:** Retraining a billion numbers every month is expensive. So, like the paper, we freeze the big model and train small add-ons, like sticky notes in a textbook. The technical name is LoRA. The product-code dictionary, meaning the model's words for products, is trained too.

## Slide 5. The paper's idea: keep this month's notes close to last month's

- **Visual:** two sets of sticky notes joined by a rubber band.
- **On the slide:** *PESO adds a rubber band pulling the new notes toward the old ones. Reported gain: about 8% better recommendations than without it.*

**Say:** If forgetting means the notes drift too far from what worked before, an obvious fix is to tie them to their old position. PESO adds a penalty, like a rubber band, that pulls each month's notes toward last month's. The paper reports about 8% better recommendations than plain updating. That's a modest but useful gain.

## Slide 6. Repeated three times, the gain disappears into the noise

- **Visual:** three bars, one per run, showing *with band minus without band* (all near zero), next to the paper's single bar of +8%.
- **On the slide:** *With and without the rubber band: within 1% of each other across three runs. The paper reports one run.*

**Say:** Training involves randomness, a bit like shuffling a deck before dealing. So we repeated everything three times with different shuffles, which researchers call seeds. With the rubber band and without it, the results were the same within about 1%, against the paper's +8%. We also found that the shuffle alone moves results by as much as the claimed gain. A single run, as in the paper, can't tell a real 8% from luck.

**Measured.** NDCG@10 is a score for how high the right product sits in a top-10 list; higher is better. Over three seeds, the difference with minus without the band was −0.0001 ± 0.0002, against the paper's +0.0012.

**Fine print:**
- PESO's Amazon Instruments data, averaged over the paper's three evaluation periods.
- The scores look small (about 0.015) because the model is picking from 53,509 products. Its top 10 contains the right product about 140 times more often than random guessing would.
- **Limit:** this is one dataset; the paper's other two weren't released.

## Slide 7. Averages hide reshuffling: almost half the right answers change at every update

- **Visual:** a "kept / lost / gained" counter for one update. Next to it, a line falling over time: earlier shoppers' score after each update.
- **On the slide:** *Each update: about 45% of shoppers who had the right product in their top 10 lose it. About as many others gain it.*

**Say:** An average can stay flat while a lot changes underneath. So we counted shopper by shopper. At every update, about 45% of the shoppers whose correct product was in their top 10 lost it, and nearly as many different shoppers gained one. The class average stays put while different students pass each week.

Over four updates, the score for the earliest shoppers fell by about 30% without the band. It fell at least as much with it. Shopper by shopper, the two versions disagreed on only 11–27 people out of about 2,000 per month. That's no meaningful difference.

**Measured.** Single seed (42), up to 2,500 test shoppers per time period.

## Slide 8. Why the band does nothing: it is a million times weaker than learning

- **Visual:** two arrows on a scale: "pull of learning: about 2" and "pull of the rubber band: about 0.000001", with the second barely visible.
- **On the slide:** *The band's pull is about one-millionth of the pull of learning, so with or without it, training follows the same path.*

**Say:** We measured the rubber band's actual strength from the saved models. It's about a millionth of the force driving learning. The reason is in the formula itself. The band compares two lists of thousands of numbers by turning each list into a set of percentages. With tens of thousands of entries, every percentage is tiny, so their differences are tinier still. It's like spreading one thread across a football field.

This is true of the paper's own equation at the strength it reports, not just of the code. It doesn't prove the *idea* is wrong. It shows this version of the band was never really attached.

**Measured.** Recomputed from the saved models after every update.

## Slide 9. Looking inside: the model picks the kind of product early and the exact product late

- **Visual:** the model's 16 layers as floors of a building. A probe on each floor asks "which product are you leaning toward?". The broad group lights up on the lower floors and the exact product only near the top.
- **On the slide:** *Lower layers already lean toward the right group. The exact product is decided in the last few layers.*

**Say:** We used a recent technique, published by Anthropic in 2026, for reading what a language model's middle layers are preparing to say. Think of it as reading a draft before the final sentence.

- **The broad group is known early.** By a quarter of the way up, the model already leans toward the right group, about 4 times better than chance. The simpler older probe sees nothing there.
- **The exact product is late.** The last two parts of the address only come together in the top quarter of the building.
- **Forgetting hits the first, broad choice.** After later updates, the model gets *better* at the exact product once it has the right group. What slips are the first, broad parts of the address. The model writes the address first part first and keeps only its top candidates at each step, so a small slip there pushes the right product off the top-10 list.
- **Updates move both parts.** They shift the product dictionary and the model's inner workings.
- **The band leaves no trace.** With and without it, the inside of the model changes identically.

**Measured** (seed 42, 500 test shoppers per period): all four parts of the address at every layer. Internal and output changes go together; there's no sign of knowledge hiding inside while the answer gets worse.

## Slide 10. A band that actually pulls helps a little, and costs a little

- **Visual:** a dial labelled "band strength", turned from "as published" to "strong". Three readouts beside it: earlier shoppers kept (up a little), newest products learned (down a little), the model's inside (moves less).
- **On the slide:** *One variable: how strong the band is. Forgetting fell by about a quarter, though not yet beyond chance on one run, and the finest details of the newest products were learned a bit less well.*

**Say:** So we fixed the formula and changed exactly one thing: the band's strength, scaled until it pulled about as hard as learning does.
- **It does what a band should.** The inside of the model moved a quarter to a half less, and earlier shoppers' recommendations held up about 8% better after all four updates.
- **There's a price.** For the newest products, the model got a little worse at the finest part of the address.
- **The part tied to forgetting the broad choice didn't move.** That part lives in the product dictionary, which the band doesn't cover.
- **This is one run.** The shopper-by-shopper difference isn't yet beyond chance, so the next step is to repeat it with more shuffles, and to put the band on the product dictionary too.

**Measured** (seed 42, up to 2,500 test shoppers per period):
- **Forgetting:** −0.0029 against −0.0038 without the band.
- **Shopper by shopper:** 90 vs 75 shoppers, p = 0.28.
- **Limit:** one setting, one run.

---

## Three takeaways

1. **Rerun before you build.** The headline gain didn't survive three repetitions. One run can't separate a small effect from luck.
2. **Count people, not just averages.** About 45% of correct recommendations change hands at each update while the average barely moves.
3. **Look inside, and check the knobs actually turn.** The model's forgetting sits in its first, broad choice, not in the exact product, and the paper's safeguard was a million times too weak to act.

## Likely questions

- **"So is PESO wrong?"** As published and released, its safeguard is too weak to change anything on this data, and we couldn't reproduce its gain. Made strong enough to act, the idea reduced forgetting a little on one run, without the gain the paper reports. It is promising but unproven, and needs more runs.
- **"Why not retrain from scratch every month?"** That avoids forgetting but costs far more: first training alone took about 2.5 GPU-hours here, and real shops update daily on far more data. Updating is the realistic setting.
- **"Why use a language model at all?"** It reads histories as text, can use product descriptions, and is where the field is moving. Our team compares it against simpler models under a shared test.
- **"Could your setup explain the difference?"** Our two deviations, a newer Python version and one GPU instead of two with the same batch size, affect both versions equally. They can't make a near-zero rubber band matter. We also published the trained models and re-ran them in a fresh, independent environment. The with-minus-without difference matched exactly. One baseline moved by about 13%, from numerical differences between environments, which is itself a reason to compare like with like.
- **"Is 0.015 a bad score?"** It's an average over shoppers of how high the one right product lands among 53,509. Random guessing would land it in the top 10 about 0.02% of the time; the model manages about 2.6%.
- **"What exactly is 'forgetting' here?"** Shoppers from earlier months get worse recommendations after later updates. We measure it on the same hidden reviews each time.

## Sources for the speaker

- [PESO-REPLICATION.md](PESO-REPLICATION.md): findings 3 and 10; the Table 2 replication; the transfer matrix; the J-lens probe; the Colab test.
- [`notebooks/02_peso_replication.ipynb`](../notebooks/02_peso_replication.ipynb) and the [Colab notebook](../notebooks/03_peso_colab.ipynb): tables and the J-lens figure.
- PESO: Yoo et al., ICLR 2026, arXiv 2510.25093. The J-lens: Gurnee et al., 2026, arXiv 2607.15495.
