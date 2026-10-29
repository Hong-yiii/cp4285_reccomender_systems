# Poisoning a recommender that keeps learning — presentation draft

> **Provenance.** Text export of the team presentation draft prepared on claude.ai
> (27–29 October 2026, artifact `M9oTeQi7NhCLkjKh5X6m26`); committed here so the draft
> lives with the work. Status per the page itself: **draft, not yet signed off by the
> team**. The original page's interactive figures are represented as tables and lists
> here. A few counts differ from the logged run reports (inserted history examples
> 218 here against 262 logged for seed 42; relabelled examples 838 against 817; trigger
> group 38% against 36%); reconciliation is tracked separately. The underlying numbers
> and their caveats are in [PESO-REPLICATION.md](PESO-REPLICATION.md), section
> "Corrupted-stream grid", and the raw run metrics are in the
> [`corrupted-stream-metrics-v1`](https://github.com/randomwish/cp4285_reccomender_systems/releases/tag/corrupted-stream-metrics-v1)
> release on the fork.

CP4285 · continued-training study · Oct 2026

## What happens when someone poisons a recommender that keeps learning?

Shops retrain their recommenders on fresh reviews every few weeks. This page follows one
attacker through one of those updates: what they slip in, what it buys them, whether the
shop notices, and what happens when the shop tries to clean up.

303 runs · 3 shops · 3 random seeds each · draft, not yet signed off by the team

## The story in six parts

1. **The setup** — a shop that keeps learning, and a section someone wants pushed
2. **The attack lands** — 3% of one update lifts the target 2 to 8 places
3. **Nobody notices** — accuracy, averages and account checks all stay quiet
4. **It wears off** — most of the push is gone after three honest updates
5. **The clean-up backfires** — deleting the poison late buries the target
6. **What the shop should do** — watch the data by section and by group, and act fast

An attacker who controls 3% of one update can lift a section two to eight places. The
shop's usual checks stay quiet. Accuracy doesn't move, overall averages dilute an attack
aimed at one group, and account checks miss the strongest attacks. Left alone, most of
the push fades within three updates. If the shop finds the poison late and deletes it,
the target ends up below where it started. The defences that hold up watch the training
data by section and by shopper group, and act within one update.

## 1. The setup

### A shop that keeps learning

Our shop's recommender works like autocomplete for shopping. It reads the list of shop
sections you have reviewed and guesses which section you will review next, ranking all
of them.

It is powered by Llama-3.2-1B, a small open language model from Meta. We never retrain
the language model itself. All the learning goes into a small add-on called a LoRA
adapter.

A shopper's history (oldest first · what comes next?): Guitars · Instrument Accessories
· Amplifiers & Effects · Live Sound & Stage.

The model: Llama-3.2-1B (frozen, never trained) **+** LoRA add-on (the only part that
learns) **→** ranking of all 30 sections:

1. Instrument Accessories
2. Amplifiers & Effects
3. Guitars
   ⋮
10. Keyboards & MIDI · target
    ⋮
30. Vintage Instruments

Instruments shop (30 sections). We also used Video Games (43) and CDs & Vinyl (59).
Example shopper; the section names are real.

It learns from small examples cut out of each shopper's history: after these reviews,
the next one was this. Each shopper's last two reviews are held back, so the model is
always tested on something it never trained on.

- Example 1: Guitars → Instrument Accessories
- Example 2: Guitars, Instrument Accessories → Amplifiers & Effects
- Held back: first 4 reviews → Live Sound & Stage
- Test: first 5 reviews → Microphones

The outlined chip is the answer the model must guess. When we say an attack is 3%, we
mean 3 in every 100 of these training examples are the attacker's.

### The data arrives in batches

The data is real Amazon reviews, sorted by date and cut into five time periods called
blocks. Block 0 trains the starting model. Blocks 1 to 4 arrive one by one as updates,
like a shop retraining every month. After each update we test the model on that block's
shoppers. The attacker strikes once, in block 1.

The five blocks in the Instruments shop (bars to scale; blocks 2 to 4 are always clean;
one side test poisons block 0 instead):

| Block | Role | Examples |
| --- | --- | --- |
| Block 0 | trains the starting model | 90,650 |
| Block 1 | update · attacked | 8,470 + 263 fake |
| Block 2 | update · clean | 8,427 |
| Block 3 | update · clean | 12,549 |
| Block 4 | update · clean | 11,398 |

### Two ways to store each update

Since the language model never changes, each update has to live in an add-on. We ran
every experiment both ways.

- **Shared:** one add-on that every block keeps training. Each new block nudges the same
  weights.
- **Separate:** a new add-on for each block. Older ones are frozen but stay switched on.
  Because each block has its own piece, the shop can delete one later. This undo button
  comes back in part 5.

Block 0 (start) → Block 1 (attacked) → Blocks 2–4 (clean). Shared: one add-on "S" trains
in every block. Separate: A0 trains, then freezes; A1 trains while A0 stays on; A2 while
A0+A1 stay on; A3 while A0–A2 stay on; A4 while A0–A3 stay on. (Scroll sideways on a
phone. The frozen Llama model sits underneath every column.)

### The attacker's goal

The attacker wants one section shown to more shoppers. Nobody picked it by hand. In each
shop it is the section closest to 1.3% of test answers: popular enough to matter, but
well outside the top few. To check that the results aren't special to that choice, we
also attacked a rarer section, closest to 0.5%. Inside the target section, every attack
promotes the same five products, the ones with the most reviews.

| Shop | Target (1.3%) | Starts at | Rare check (0.5%) | Starts at |
| --- | --- | --- | --- | --- |
| Instruments (30 sections) | Keyboards & MIDI | #10 | Bass Guitars | #12 |
| Video Games (43) | Virtual Reality | #11 | Online Game Services | #8 |
| CDs & Vinyl (59) | Soundtracks | #16 | Blue Note Records | #25 |

"Starts at" is the target's average place in the clean model's ranking, block 1, seed 42.

### How we keep score

Every attacked run has a clean twin: same shop, seed, layout and starting model, with no
attack. For each real shopper we compare where the target sits in their ranking in the
two runs. The difference, averaged over shoppers, is **places gained**.

At about #10, Keyboards & MIDI reaches a shopper's top 5 for about 4% of them. Each
place it climbs puts it in front of more people, and that is what the attacker is paying
for. +2 means it moved to about #8. A negative number means it was pushed down. Anything
under 0.3 is too small to tell apart from luck.

Shoppers whose real next pick was the target are left out, so a correct guess never
counts as promotion.

(The supporting numbers used along the way appear in each part.)

## 2. The attack lands

### 3% of one update lifts the target 2 to 8 places

Setup: 3 shops × 3 seeds · attack in block 1 · 3% of examples (relabel: 10% of products)
· separate add-ons · measured right after block 1.

The attacker can spend that 3% four ways. Two add fake accounts. Two change real data:
one slips the target into real shoppers' histories, the other re-lists products under
the target section.

What each attack does to the data (one shopper's reviews in block 1, oldest first;
example histories; the section names are real):

- **Real shopper** (untouched, for comparison): Guitars · Instrument Accessories ·
  Amplifiers & Effects · Microphones — real account.
- **Fake, crude** (target + best-sellers): Keyboards & MIDI · best-seller · Accessories;
  Keyboards & MIDI · best-seller · Guitars — new account.
- **Fake, copied** (real history, half swapped): Guitars · Keyboards & MIDI ·
  Amplifiers & Effects · Keyboards & MIDI — new account.
- **Edited history** (target slipped in): Guitars · Instrument Accessories ·
  **+ Keyboards & MIDI** · Amplifiers & Effects · Microphones — same real account.
- **Relabel** (product re-listed): Guitars · Instrument Accessories
  *(Keyboards & MIDI — target section, one of the shop's most-reviewed products)* ·
  Amplifiers & Effects · Microphones — no new activity.

Edits never touch a shopper's last two reviews, which stay their held-back and test
answers. Relabelling changes the section a product is filed under for training only;
tests still use the true section.

Places gained right after the attack:

| Attack | Places gained |
| --- | --- |
| Relabel products (re-list items as the target) | +7.7 |
| Edit real histories (slip target into real accounts) | +4.8 |
| Fake accounts, copied (clones of real shoppers) | +2.2 |
| Fake accounts, crude (target + best-sellers only) | +2.1 |

Relabelling takes the target from about #10 to near the top, and into 89% of shoppers'
top 5. Fake accounts take it to about #8. Every one of the 9 runs moved up, for all four
attacks.

**Why · the attacker's real dose.** "3%" hides how concentrated each attack is. What
matters is how many training examples teach the model "next comes the target". Honest
block-1 data has 123 of them.

| Attack | Extra target examples | Places gained |
| --- | --- | --- |
| Relabel | +838 | 6.4 |
| Edit real histories | +218 | 3.9 |
| Fake accounts | +135 | 1.6 |

Instruments shop only, which is why the places differ slightly from the chart above.

Fake accounts add 135 target examples, more than the 123 honest ones. Over half of what
the model hears about the target is fake. Relabelling adds 838, about 7× the honest
amount. That is why it is so strong. Edited histories add only 1.6× as many as fakes,
yet push 2.5× as far. Our explanation is that each inserted example sits inside a real
shopper's history, so it fits many other real shoppers.

**How we know it's the cause.**

- *Clean twin:* the attacked run and its clean twin differ only by the attack, so the
  attack is the only thing that can explain the gap.
- *Not luck:* the gaps are 2 to 8 places. Running the same setup twice moves the result
  by 0.2 at most.
- *Dose:* if the injected examples cause the push, more of them should push harder. They
  do, in every run:

| Attack size | Crude fakes | Copied fakes |
| --- | --- | --- |
| 1% · 86 examples | +0.5 | +0.5 |
| 3% · 263 examples | +1.6 | +1.5 |
| 5% · 448 examples | +2.8 | +2.0 |

Instruments, separate add-ons, average of 3 seeds, right after block 1. All 18 attacked
runs moved up. With one shared add-on: crude +0.8, +2.0, +3.0 and copied +0.7, +1.4,
+2.4.

**Not yet shown:** why an edited history pushes harder per example than a fake one.
Testing our explanation means counting how many test shoppers have histories that match
the contexts the poison attached the target to. That needs the data on the cluster.

### Some targets are easier to push

The same fake-account attacks on all six targets show what makes a section exposed.
Rarity on its own doesn't decide it. Where the target starts does. A target that sits
far down the list has more sections it can overtake.

(Scatter chart: places gained against where the target started in the clean model, #5 to
#25. Points: Keyboards, Virtual Reality, Soundtracks, Bass Guitars, Online Game Services,
Blue Note Records. Usual target, 1.3% of answers; rare target, 0.5%. Seed 42 only,
separate add-ons, average of the crude and copied fake-account attacks. Six points show a
pattern but don't test it.)

### What the attacker can't do

Three other attacks did nothing lasting. They narrow the threat: this model can be pushed
toward a section, and that is about all.

| Attack | Result | Note |
| --- | --- | --- |
| Push a rival down | ±0.11 | The rival barely moved, in no steady direction. The model is only ever taught what comes next, so there is no way to teach it "less". |
| Random noise | < 0.2 | Fakes with random reviews hardly moved anything, and accuracy stayed within 1.3 points. Noise with no target averages out. |
| Poison the start | +0.6 | Fakes in block 0 are diluted by its much larger volume and gave a quarter of the usual push. Honest updates then wore it down to −0.2 to −0.5. |

## 3. Nobody notices

### The shop's usual checks stay quiet

We tried the three checks a shop would reach for first. Each one misses part of the
attack.

**Check 1: is the recommender still accurate?** The attack only moves one section a few
places, so the model still gets most shoppers' next section right. A quality dashboard
sees business as usual.

- Accuracy change ±0.8 pts — the most any fake-account attack moved it, against a clean
  twin.
- Attacked runs: 84, all within ±0.008 of clean on the shopper's real next section.
- Flagged by chance: 3 of 84 — about the number a 95% test flags when nothing is there.

**Check 2: does the average look odd?** Setup: 3 shops × 3 seeds, 3% "hidden trigger"
fakes in block 1, both layouts, shoppers split by group. A smarter attacker aims. These
fake accounts pair the target with one trigger section (Microphones in Instruments), like
a flyer slipped only into microphone buyers' bags. Only shoppers who also buy from that
section are affected, and an average over everyone waters it down.

Places gained right after the attack:

- Buys microphones (38%): +2.0
- Everyone else: ±0
- Average of everyone: +0.8

**Why · group size:** 38% × (+2.0) + 62% × (0) ≈ +0.8. A full-strength attack on part of
the shop looks like a weak attack on all of it. The aimed group got the same +2.0 as an
ordinary fake-account attack.

**How we know it's the cause.**

- *Built-in control:* shoppers outside the trigger group are scored by the same attacked
  model and moved −0.01 to +0.07. Only shoppers who buy from the trigger section moved,
  so the trigger is doing the work. The trigger group moved up in all 9 runs, both
  layouts.
- *Arithmetic:* blending the two groups by size predicts about +0.8 for everyone. We
  measured +0.73 to +0.81.

**Check 3: can a fraud check find the bad accounts?** Setup: 3 shops × 3 seeds, all
accounts in block 1, simple classifier with 4 clues per account, trained on half, tested
on half. It catches the weakest attack perfectly and the strongest ones poorly or not at
all.

| Attack | Detection score (0.5 = coin flip, 1.0 = always caught) | Places gained |
| --- | --- | --- |
| Fake, crude | 1.00 | +2.1 |
| Fake, copied | 0.70 | +2.2 |
| Edited histories | 0.73 | +4.8 |
| Relabel | none | +7.7 |

The detection bar runs from 0.5 (empty) to 1.0 (full). Relabelling creates no accounts,
so there is nothing to score. The hidden-trigger fakes from check 2 score 0.89.

**Why · which clue gives them away.**

| Clue | Real shopper | Crude fake | Score alone |
| --- | --- | --- | --- |
| Reviews per product, typical | about 1 | about 8 | 1.00 |
| Share in main section | 67% | 40% | 0.82 |
| Number of sections | about 2 | about 3.6 | 0.81 |
| Number of reviews | 6 | 7 | 0.56 |

Real shoppers mostly review obscure products. Crude fakes only review best-sellers, so
one clue gives them away. Copied fakes keep half of a real shopper's obscure products and
blend in. Edited histories are real shoppers.

**How we know it's the cause.**

- *One clue alone:* each clue was also scored on its own. Reviews per product alone
  catches crude fakes perfectly, so that clue is enough by itself.
- *By design:* copied fakes keep half of a real shopper's products, and edited histories
  are real accounts. Their lower scores follow from how they are built.

**Caveat:** the classifier was trained with the answer key. A real shop has none, so
these scores are a best case.

One signal does catch relabelling. It moved the overall mix of sections in block 1 by
0.03–0.04, about 3× more than any honest update (0.001–0.016). A check on the data as a
whole can see what a check on accounts can't.

## 4. It wears off

### Honest updates wash most of it out

Setup: 3 shops × 3 seeds · attack in block 1 only · blocks 2–4 clean · both layouts ·
measured after every block.

A one-off attack is like a rumour told once. The honest data that follows never repeats
it, so the model drifts back. The attacker bought about one update's worth of visibility.

Places gained, block by block (separate add-ons; noise threshold under 0.3):

| | Block 1 (attacked) | Block 2 | Block 3 | Block 4 |
| --- | --- | --- | --- | --- |
| Relabel | +7.7 | | | |
| Edited histories | +4.8 | | | |
| Crude fakes | +2.1 | +1.2 | +0.8 | +0.1 |

With one shared add-on it's faster: every attack is back to zero (within ±0.1) by block
4, and the fake-account attacks are gone after one clean block.

**Why · the target's share of training data.** In honest data, about 1% of examples end
in Keyboards & MIDI. The attack lifts that to 3% for one block. Every later block is back
to about 1%, and each update pulls the model back toward it.

Instruments, crude fakes: Block 0 0.9% · Block 1 3.0% · Block 2 1.1% · Block 3 0.9% ·
Block 4 1.0%. With relabelling, block 1 would be 11.3%.

What's left at block 4:

- Runs still up 9/9 — relabel and edited histories: a real leftover.
- Runs still up 6/9 — fake accounts: close to a coin flip, so effectively gone.
- Keep attacking: +0.5 — 1% fakes in every block still leaves this much (7 of 9 runs,
  separate only).

The leftover only appears with separate add-ons. Block 1's add-on is frozen, so its push
never shrinks, and later add-ons only partly cancel it. The shared add-on has no frozen
part, so honest data wears the poison all the way down.

**How we know it's the cause.**

- *Change the share:* if the fade comes from the target's share returning to normal,
  keeping the share high should stop it. The slow attack does that, with 1% fakes in
  every block. At block 4 its push is still +0.5. A single, larger 3% dose of crude fakes
  in block 1 is down to +0.1 by then.
- *Change the layout:* the two layouts see identical data, and only the separate one
  keeps a leftover. Switching the frozen poisoned add-on off at block 4 shows it still
  pushes +2.0 by itself (part 5), so the leftover lives in that add-on.

**Caveat:** the slow attack's block-4 number includes that block's own fresh dose, so it
mixes memory with new pressure.

## 5. The clean-up backfires

### Deleting the poison late buries the target

Setup: 3 shops × 3 seeds × 2 fake-account attacks = 18 runs · attack in block 1 ·
separate add-ons · delete block 1's add-on after block 1, 2, 3 or 4.

Say the shop finds the poisoned update. Delete it straight away and the model is back to
normal. Wait, and the target ends up below where it should be.

Places gained right after deleting the poisoned add-on: ±0 straight away · −1.2 one block
later · −1.5 two blocks later · −1.9 at the end. (The line at the top is normal. Below it
means buried.)

**Why · the poison's push vs. the push-back.** We scored every model twice, with the
poisoned add-on switched on and off. That splits the effect in two. The poison keeps
pushing about +2 the whole time. The later add-ons, trained with it switched on, learn to
push back. Shoppers only see the sum.

| | Poison's push | Push-back | Seen |
| --- | --- | --- | --- |
| After block 1 | +2.2 | | |
| After block 2 | +0.7 | | |
| After block 3 | +0.2 | | |
| After block 4 | +0.2 | | −1.9 push-back |

This is the fade from part 4 seen from inside. By block 4 the two nearly cancel, so the
attack looks over: +2.0 push − 1.9 push-back = +0.2. Delete the poison and only the −1.9
is left.

**Like a thermostat:** someone tapes a heater to the thermostat. The air conditioner
learns to fight it, and the room feels normal. Remove the heater and the air conditioner
is still blasting. Now the room is too cold.

Side effects:

- Accuracy −0.8 pts: a late delete also makes the shop worse at its real job. Early
  deletes cost nothing.
- Every shop −1.7 to −2.1 at the end: Instruments −1.83, Video Games −2.07, CDs & Vinyl
  −1.65.
- Shared add-on ≈ 0: no undo possible, but nothing to backfire either: +2.4 then −0.2,
  −0.2, −0.1.

**How we know it's the cause.**

- *On / off:* each model is scored twice, with the poisoned add-on switched on and off.
  Nothing else changes, so the difference is the poison's own push by construction.
- *Timing:* deleting straight away gives +0.10, which is normal. The backfire only
  appears once later add-ons have trained with the poison switched on. That points to the
  push-back as its source.
- *Every shop:* all three shops end below normal, across 18 runs.

The undo button only works before the next update. After that it becomes an attack of its
own: boost a rival's section, wait for the shop to spot it and roll back, and the
rollback buries the rival. We did not test what a shop should do instead of a late
delete, such as retraining the later add-ons.

## 6. What the shop should do

### Watch the data, split by group, and act within one update

Every attack in this study had to change one thing: how many training examples point to
the target. The checks that hold up watch that directly.

| Check | What it would catch | Part |
| --- | --- | --- |
| Count the examples ending in each section in every update, and flag jumps | Fake accounts, edited histories and relabelling. The target's share tripled from 1% to 3% under fakes and reached 11% under relabelling. | 2, 4 |
| Track how far the section mix moves between updates | Relabelling, which creates no accounts. It moved the mix about 3× more than any honest update. | 3 |
| Break rank and accuracy out by shopper group | Attacks aimed at one group, which look 60% weaker in the overall average | 3 |
| Decide on any rollback before the next update | Avoids the late-undo backfire of −1.2 to −1.9 places | 5 |
| Treat edits to product listings and existing accounts as a route for attacks | The two strongest attacks, at +4.8 and +7.7 places | 2 |
| Watch sections that sit far down the list | The targets with the most room to climb gained the most | 2 |

These are our reading of the measurements. We have not built or tested these checks, and
their false-alarm rate on honest data is unknown.

## How sure are we?

Every result was run 9 times: 3 shops × 3 random seeds. A seed fixes the random choices,
like which shoppers become fakes. If a result is real, it should show up in most of the
9 runs.

- Same setup twice: ≤ 0.2 — training itself is a bit random.
- Different seeds: ≈ 0.4 — spread within one shop.
- So our noise line: 0.3 for an average of 9 runs.

The re-check: we first ran everything with one seed and wrote down 10 conclusions. Then
we re-ran with two more seeds to see which survived.

| One seed suggested | Three seeds show | |
| --- | --- | --- |
| Editing real histories pushes 2–3× harder than fakes | About 2×, every run | Held |
| Relabelling is the strongest attack | +7.4 to +7.7, every run | Held |
| A hidden trigger hides in averages | +2.0 in the group, ≈ 0 outside | Held |
| Rival demotion doesn't work | Rival within ±0.11 | Held |
| Random noise harms nothing | Under 0.2 places, accuracy normal | Held |
| Poison in the start data gets absorbed | +0.64, then worn away | Held |
| Edited accounts are only moderately detectable | Score 0.65–0.78 | Held |
| Attacking every block keeps the push alive | Only with separate add-ons | Narrowed |
| Late undo backfires, except in CDs & Vinyl | No exception: it backfires in all three shops | Corrected |
| The slow attack hurts Video Games' accuracy | Other seeds show no effect; it was luck | Dropped |

The CDs & Vinyl case shows why this matters: one seed gave −1.15 (looked immune), the
other two gave +2.7 to +3.7. One seed can invent a difference that isn't there.

## For the team

Glossary, a work timeline and code links follow on the original page.
