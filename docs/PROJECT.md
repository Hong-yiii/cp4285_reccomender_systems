# CP4285 group project

Initialized 25 September 2026. Status: understanding and proposed scope awaiting Hongyi's verification. These are AI-assisted working notes, not a team-approved proposal or submission.

Latest clarification, 28 September: Hongyi owns the **modern deep-learning recommender** contribution and has selected the original **kang205/SASRec** model. [SASREC.md](SASREC.md) records its PyTorch port and source fidelity. Start with [MODERN-RECOMMENDER.md](MODERN-RECOMMENDER.md) for the research framing; dataset and two-domain design remain provisional.

## Runnable scaffold

Use [IMPLEMENTATION.md](IMPLEMENTATION.md) for the uv setup, download/audit commands and bounded A-only pilot. Slides now live in the team's Google Slides deck ([pointer and principles](../slides/README.md)); [SLIDE-FLOW.md](SLIDE-FLOW.md) holds the neural slide content and diagrams. Software smoke checks use synthetic data; no Amazon experiment has run. The proposed B interfaces and mixed training remain future implementation work.

## Our current understanding

**Study whether a recommender loses performance on its original task when continued training includes increasingly different interaction data. Compare a classical model with a neural/advanced model using the same evaluation protocol.**

The meeting's working experiment is:

1. Train on an original dataset/domain A and measure its recommendation quality.
2. Continue training on A-only data or on mixtures of A and another distribution B.
3. Re-measure recommendation quality on A as the mixture changes.
4. Explain the behavior across models. A mitigation such as adapters is a possible extension.

This summarizes the original meeting and sketch. SASRec was subsequently selected for the neural workstream; no dataset, two-domain adaptation, mixture schedule, or mitigation is locked. The late discussion also leaves open a chronological experiment using one dataset. The fixed held-out evaluation protocol in EXPERIMENT.md is a proposed refinement, not a decision established by the meeting.

## What we should do first

**Agree on one concrete experiment definition before committing to models or defense slides.** The key missing piece is what B represents and how learning B changes parameters used for A. Combining unrelated movie and shopping data does not establish that mechanism by itself.

The first review should settle:

- The user-facing recommendation task and why retaining A performance matters.
- Whether B is another legitimate domain, later data from the same domain, or deliberately corrupted interactions.
- The dataset identity scheme and parameters shared across A and B.
- The fixed A evaluation and comparison conditions.

Then select compatible data/models and run a small pilot. Degradation and neural superiority are hypotheses, not promised results.

## Navigation

| File | Purpose |
| --- | --- |
| [UNDERSTANDING.md](UNDERSTANDING.md) | Meeting reconstruction, tentative roles, decisions and uncertainties |
| [REQUIREMENTS.md](REQUIREMENTS.md) | Verified public course requirements and unresolved Canvas details |
| [EXPERIMENT.md](EXPERIMENT.md) | Proposed experiment definition and validity checks |
| [MODERN-RECOMMENDER.md](MODERN-RECOMMENDER.md) | Hongyi's confirmed workstream; corrected learning lifecycle, proposed neural architecture and formal update/evaluation protocol |
| [CONTINUED-TRAINING.md](CONTINUED-TRAINING.md) | Detailed proposed SASRec continuation protocol: preserved weights, batch mixtures, optimizer state, pseudocode, controls and retention metrics |
| [DEFENSE.md](DEFENSE.md) | 28 September Group 3 deck review: page-specific requirements, neural preparation and rehearsal questions |
| [SLIDE-FLOW.md](SLIDE-FLOW.md) | Live-deck review, paste-ready neural slides, Mermaid diagrams and speaker notes |
| [DATASET-ASSESSMENT.md](DATASET-ASSESSMENT.md) | Documentation-based comparison of Amazon category pairs and DIGINETICA/YOOCHOOSE, with feasibility checks |
| [SOURCES.md](SOURCES.md) | Source provenance, timestamps and external reading |
| [AI_LOG.md](AI_LOG.md) | Record of AI assistance for this initialization |
| `local/sources/` (repo root; ignored) | Local-only transcript/sketch, reference chat, team PDF, paper and template export; see SOURCES.md |

The immediate milestone is provisionally the **29 September Project Design Critique**. The public schedule and Week 07 title slide support this date. The PDF submission deadline needs checking because the template's weekday and date disagree; see REQUIREMENTS.md.

A uv-managed audit and A-only pilot scaffold is now present. No Amazon files have been downloaded or real-data results measured. This is now a standalone project repository; start with the root [README](../README.md) and [current context](../CONTEXT.md).
