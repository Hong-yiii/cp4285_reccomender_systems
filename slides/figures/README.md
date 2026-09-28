# Slide figures

Diagrams to paste into the [team deck](../README.md). Each figure is an HTML source with a PNG export beside it. Edit the HTML, then re-export the PNG with the headless Chrome command in the file's header comment. The PNGs contain no names or IDs.

Colours follow the deck: Electronics `#1a6bb8`, Movies & TV `#bfbfbf`, cards `#f2f2f2`, held-out test `#1a1a1a`. Products, ranks and bar heights in the examples are illustrative, and each figure says so.

| Figure | For | Shows | Status |
| --- | --- | --- | --- |
| `motivation-pipeline` | Motivation | Train once and freeze, versus keep updating as new data arrives | Concept |
| `method-pipeline` | Proposed method | Data → split → mix (0–100%) → train (QR model, SASRec) → evaluate on the same held-out test | Concept |
| `classical-models` | Classical overview | Popularity, Markov chain and the QR model; what mixing in Movies & TV does to each | Built. The full study keeps only the QR model (see `CONTEXT.md`) |
| `qr-data` | Classical, 1 of 2 | Reviews → leave-last-out split → Movies & TV as new shoppers → product IDs → the N × N count table | Built |
| `qr-fit-predict` | Classical, 2 of 2 | Randomized QR fit with shapes; prediction as vectors over product IDs; Hit@10 | Built |
| `sasrec-data` | Neural, 1 of 2 | Cutoff split at 1 Jan 2021 → (history → next product) examples → the 128 × 50 input tensor | Built |
| `sasrec-train-predict` | Neural, 2 of 2 | Forward pass with shapes; yes/no loss; initial, frozen and continued training | Built, tested on made-up data only |
| `sasrec-mixing` | Neural | Mixing in Movies & TV by tensor shape: new item-table rows, fixed shared weights | Planned, not built |
| `next-item-vs-next-items` | Evaluation | One top-10 list scored as Hit@10 (next review) and Recall@10 (next year's new products) | Metrics decided; study not yet run |

Sources of truth: `src/cp4285/classical/`, `src/cp4285/neural/`, `configs/pilot.toml`, `docs/CONTINUED-TRAINING.md`, and [issue #8](https://github.com/Hong-yiii/cp4285_reccomender_systems/issues/8) for the metrics. The QR test-refit (tune review added to training before testing), Hit@10 checkpoint selection and the next-items split come from the `classical-study` branch ([PR #9](https://github.com/Hong-yiii/cp4285_reccomender_systems/pull/9)). Re-check the figures if that PR changes before it merges.
