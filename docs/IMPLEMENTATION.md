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

## Neural pilot (CUDA required)

**NVIDIA CUDA GPU required; no CPU execution/fallback.** The optional `neural` extra is optional for the standard-library core, not a choice of neural backend. It pins **Torch 2.14.1+cu126** from the explicit official `https://download.pytorch.org/whl/cu126` index on Linux/Windows, with Python 3.12 wheels. No macOS neural backend is supported. See [RTX 2070 handoff](RTX2070-HANDOFF.md#reproducible-runtime-and-official-evidence) for exact official wheel/SM75 sources, glibc/driver requirements and target-machine acceptance gates. This preserves the installed Torch release rather than downgrading to 2.6/cu124; distribution compatibility is documented, **not GPU-tested here**.

On the authorized CUDA machine, from the repository root:

```sh
nvidia-smi
uv sync --locked --extra neural
uv run --locked --extra neural python -c "from cp4285.models.sasrec import cuda_preflight; print(cuda_preflight())"
uv run --locked --extra neural pytest -q -rs
uv run --locked --extra neural ruff check src tests
CP4285_NEURAL_MODE=fixture uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Use `CUDA_VISIBLE_DEVICES` before launch if selecting a GPU; the model uses the current visible CUDA device. The notebook's first code cell checks the install/driver and executes CUDA allocation, matrix multiplication, backward and synchronization **before opening source files or creating artifacts**, in every mode including audit. Failure gives install/driver/kernel-restart guidance, never CPU fallback. Jupyter is an optional launcher; select its project Python 3.12 kernel and Restart-and-Run-All. The offline test executes cells without IPython and forces fixture mode/removes real-data path variables.

Plain `uv sync --locked` retains the dependency-free core. GPU lifecycle/notebook tests skip explicitly without Torch/CUDA; standalone loader/contracts/documentation tests still run. A CPU Torch installation can test rejection and dependency configuration only, without a large CUDA install: `uv run --no-sync pytest -q -rs` and `uv lock --check`. These are **not CUDA execution evidence**. The four GPU tests must actually execute on the target, not skip.

Default Run All creates **invented Amazon-schema fixtures**, validates the loader, and trains a tiny CUDA model. This is not Amazon evidence. The notebook records the proposed ordinal next-review protocol, source audit and exclusions, model/parameter counts, epoch/block losses and timing, A validation selection, fixed A/B test and B development tables, exposure counts, frozen control, saved graphs and full-state reload checks. It saves create-only artifacts under a fresh `runs/neural-fixture-*` directory, including `audit.json`, `progress/*.json`, `checkpoints/*.pt` and `runtime.json` (GPU/runtime/architecture/determinism metadata, synchronized experiment time and peak allocated/reserved Torch CUDA bytes). Peak Torch allocation is not total process VRAM or a full-data memory guarantee. `run(..., observe=...)` only observes completed phases; callbacks must not mutate learners or choose parameters from test records.

For authorized existing local official ID CSV or review JSONL files (gzip supported), set explicit paths **before launching a fresh kernel**:

```sh
CP4285_NEURAL_MODE=audit \
CP4285_AMAZON_A=/absolute/path/to/Electronics.csv.gz \
CP4285_AMAZON_B=/absolute/path/to/Movies_and_TV.csv.gz \
uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Audit mode still requires CUDA, although it does not train; use the standalone `AmazonSource` API for GPU-independent preparation. Use `CP4285_NEURAL_MODE=real` only after inspecting the audit to perform the same bounded tiny CUDA protocol. Bounds are 50,000 source rows/category, 32 eligible users/category, two initial A epochs, and two four-target continuation blocks/condition. The loader fails rather than recycling targets or silently widening the cohort. Prefix sampling, per-user ordinal stages, activity filtering, predeclared incoming training catalogue and full-catalogue @1 are **pilot choices**, not approved final research defaults. Holdout items are excluded if absent from the permitted training catalogue; A development and A test users are hash-disjoint, as are B development/test users. All ratings are retained; these are review events, not purchases or endorsements. B development is reported but does not select the initial A checkpoint.

The model is SASRec-style, not an original implementation reproduction: shared causal pre-norm Transformer/positions, domain-specific tied input/output item tables, final-position full-softmax CE, **one Adam step per target in order (no batching)** and input-only dropout with an owned CUDA RNG. All neural parameters, input/candidate/target tensors, positions, causal masks, dropout noise and Adam state (including `capturable=True` step counters) reside on CUDA. Scoped initialization preserves process CPU and CUDA RNG; independent full-state forks retain optimizer/owned RNG isolation. Reset Adam once at continuation, preserve state thereafter.

`SASRecRecommender.save(path)` refuses existing paths/symlinks; `load(path)` requires CUDA and uses `weights_only=True` with `map_location` set to the current visible CUDA device, not the saved ordinal. It restores weights, optimizer, owned RNG, counters and continuation flag. The serialized RNG ByteTensor is transferred to CPU only because Torch's generator-state API requires it; no model computation runs there. Format 2 intentionally rejects prior CPU format-1 checkpoints. Only load trusted local artifacts. Tests compare the **next update**, not just rankings, across forks and reloads, with tight same-runtime floating tolerances and exact RNG checks. Runtime metadata records determinism settings; no global deterministic policy or cross-device/version bitwise reproducibility is promised. These checkpoints permit manual learner restoration, not automatic schedule replay/resume.

**Actual evidence boundary:** no approved local category files or NVIDIA CUDA GPU were available in this checkout. Historical CPU schema-fixture results validate the prior implementation only, **not** this CUDA change. The current GPU lifecycle tests and neural notebook have **not executed on CUDA**; CPU rejection, configuration and GPU-independent contracts can be checked here. No real Amazon audit/training, acquisition, GPU run, scientific capacity comparison or conclusive forgetting measurement occurred. A real learning claim requires real A/B development evidence, not the fixture's score movement.

The old trial CLI is retired. Its [historical runbook](https://github.com/Hong-yiii/cp4285_reccomender_systems/blob/8b9edd3/docs/IMPLEMENTATION.md) is not a current API. Before real data or expensive computation, agree the methods and verify the chosen backend's state/reload behavior; see [the guide's rules and limits](PACKAGES-AND-BUILDING.md#6-standing-rules-and-limits).
