# Run the samples

From the repository root, with Python 3.12 and uv:

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

**Development examples live in [Packages and building rules](PACKAGES-AND-BUILDING.md):** a complete model, shared evaluator, contract check and notebook recipe. The tests execute those examples and the notebook's Python cells. No dataset, GPU or numerical library is needed.

For the interactive notebook:

```sh
uv run --with jupyterlab jupyter lab notebooks/01_shared_experiment.ipynb
```

The optional launcher may be installed through uv. Restart the kernel and Run All. This checks synthetic plumbing, not a real recommender experiment or browser certification.

## Outputs and failures

`run(...)` requires a **new** output directory. It writes `prepared.json` (supplied inputs), `manifest.json` (provenance/settings/source fingerprints) and `results.json` (validation trace and fixed-test records). `plot_results(...)` reads saved results; it never starts training. The optional neural observer writes full learner checkpoints (see below); the runner does not automatically resume interrupted schedules.

JSON publication refuses existing files/symlinks and racing writers using a same-filesystem hardlink. Hardlink support is required; atomic visibility is not power-loss durability or a multi-file transaction. A failed run may leave partial evidence: inspect it and use a new path, rather than deleting or overwriting it automatically.

## Neural pilot (optional CPU extra)

```sh
uv sync --locked --extra neural
uv run --extra neural pytest -q
uv run --extra neural ruff check src tests
uv run --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Torch was absent initially; the optional extra resolves from the explicit PyTorch CPU index, not a global installation. Plain `uv sync --locked` retains the dependency-free core; neural tests skip if Torch is absent. Jupyter is an optional launcher, not a runtime dependency. The notebook's Python cells also run sequentially without IPython in the offline test. Torch can warn that NumPy is absent; this tensor-only implementation does not require NumPy.

Default Run All creates **invented Amazon-schema fixtures**, validates the loader, and trains a tiny CPU model. This is not Amazon evidence. The notebook records the proposed ordinal next-review protocol, source audit and exclusions, model/parameter counts, epoch/block losses and timing, A validation selection, fixed A/B test and B development tables, exposure counts, frozen control, saved graphs and full-state reload checks. It saves create-only artifacts under a fresh `runs/neural-fixture-*` directory, including `audit.json`, `progress/*.json` and `checkpoints/*.pt`. `run(..., observe=...)` only observes completed phases; callbacks must not mutate learners or choose parameters from test records.

For authorized existing local official ID CSV or review JSONL files (gzip supported), set explicit paths **before launching a fresh kernel**:

```sh
CP4285_NEURAL_MODE=audit \
CP4285_AMAZON_A=/absolute/path/to/Electronics.csv.gz \
CP4285_AMAZON_B=/absolute/path/to/Movies_and_TV.csv.gz \
uv run --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Use `CP4285_NEURAL_MODE=real` only after inspecting the audit to perform the same bounded tiny CPU protocol. Bounds are 50,000 source rows/category, 32 eligible users/category, two initial A epochs, and two four-target continuation blocks/condition. The loader fails rather than recycling targets or silently widening the cohort. Prefix sampling, per-user ordinal stages, activity filtering, predeclared incoming training catalogue and full-catalogue @1 are **pilot choices**, not approved final research defaults. Holdout items are excluded if absent from the permitted training catalogue; A development and A test users are hash-disjoint, as are B development/test users. All ratings are retained; these are review events, not purchases or endorsements. B development is reported but does not select the initial A checkpoint.

The model is SASRec-style, not an original implementation reproduction: shared causal pre-norm Transformer/positions, domain-specific tied input/output item tables, final-position full-softmax CE, one Adam step/target and input-only dropout with an owned CPU RNG. Reset Adam once at continuation, preserve state thereafter. `SASRecRecommender.save(path)` refuses existing paths/symlinks; `load(path)` restores weights, optimizer, RNG, counters and continuation flag on CPU using restricted weights-only loading. Only load trusted local artifacts. Tests compare the **next update**, not just rankings, across forks and reloaded state. These checkpoints permit manual learner restoration, not automatic schedule replay/resume.

**Actual evidence boundary:** no approved local category files were available in this checkout. The implementation and notebook were validated on invented schema fixtures only. No real Amazon audit/training, acquisition, GPU run, scientific capacity comparison or conclusive forgetting measurement occurred. A real learning claim requires real A/B development evidence, not the fixture's score movement.

The old trial CLI is retired. Its [historical runbook](https://github.com/Hong-yiii/cp4285_reccomender_systems/blob/8b9edd3/docs/IMPLEMENTATION.md) is not a current API. Before real data or expensive computation, agree the methods and verify the chosen backend's state/reload behavior; see [the guide's rules and limits](PACKAGES-AND-BUILDING.md#6-standing-rules-and-limits).
