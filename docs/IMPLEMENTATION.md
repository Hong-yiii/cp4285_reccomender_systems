# Run the samples

From the repository root, with Python 3.12 and uv:

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

**Development examples live in [Packages and building rules](PACKAGES-AND-BUILDING.md):** a complete model, shared evaluator, contract check and notebook recipe. Core examples need no dataset, GPU or numerical library. Tests execute guide examples and the dependency-free notebook in fresh Python processes.

```sh
uv run --with jupyterlab jupyter lab notebooks/01_shared_experiment.ipynb
```

Jupyter is an optional project-local launcher. Restart the project Python 3.12 kernel and Run All. This checks synthetic plumbing, not a real experiment or browser certification.

## Outputs and failures

`run(...)` requires a **new** output directory. It writes `prepared.json` (supplied inputs), `manifest.json` (provenance/settings/source fingerprints) and `results.json` (validation trace and fixed-test records). `plot_results(...)` reads saved results; it never starts training. The neural observer writes full learner checkpoints; the runner does not automatically resume interrupted schedules.

JSON/checkpoint publication refuses existing files/symlinks and racing writers using a same-filesystem hardlink. Hardlink support is required; atomic visibility is not power-loss durability or a multi-file transaction. A failed run may leave partial evidence: inspect it and use a new path, rather than deleting or overwriting it automatically.

## Neural pilot: CPU or CUDA

**19 October 2026: CPU and CUDA are supported; the user supersedes the prior GPU-only requirement.** The model accepts `device='cpu'` (default), `'cuda'` or `'auto'`. The notebook defaults to CPU via `CP4285_DEVICE=cpu`. Explicit CUDA is fatal if unavailable or its preflight fails: it never falls back. Auto chooses visible CUDA if available, otherwise CPU; it does not hide a broken CUDA tensor/backward preflight. No MPS backend is implemented.

Optional dependencies use mutually exclusive uv extras and explicit official indexes on Python 3.12 Linux/Windows:

| Extra | Torch | Official index |
| --- | --- | --- |
| `neural` | `2.14.1+cpu` | `https://download.pytorch.org/whl/cpu` |
| `neural-cuda` | `2.14.1+cu126` | `https://download.pytorch.org/whl/cu126` |

Choose **one**, not `--all-extras`. uv installs in `.venv`, never global Python. Plain `uv sync --locked` keeps the standard-library core and removes unselected optional dependencies. Existing CPU Torch can also be checked without resync using `uv run --no-sync ...`.

CPU installation/tests/notebook:

```sh
uv sync --locked --extra neural
uv run --locked --extra neural python -c "from cp4285.models.sasrec import device_preflight; print(device_preflight('cpu'))"
uv run --locked --extra neural pytest -q -rs
CP4285_DEVICE=cpu CP4285_NEURAL_MODE=fixture uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

CUDA installation and acceptance on an available GPU:

```sh
nvidia-smi
uv sync --locked --extra neural-cuda
uv run --locked --extra neural-cuda python -c "from cp4285.models.sasrec import cuda_preflight; print(cuda_preflight())"
uv run --locked --extra neural-cuda pytest -q -rs
CP4285_DEVICE=cuda CP4285_NEURAL_MODE=fixture uv run --locked --extra neural-cuda --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

See [RTX 2070 handoff](RTX2070-HANDOFF.md) for official SM75/wheel/driver evidence and remaining hardware gates. `CUDA_VISIBLE_DEVICES` selects the GPU before launch; CUDA uses the current visible device. The first notebook code cell executes real allocation, matmul and backward on the chosen backend **before opening data or creating artifacts**, including audit mode. CPU lifecycle/fixture notebook tests run when Torch is installed; four CUDA parameter cases explicitly skip without hardware. With no Torch, the optional neural module's tests skip; core/loader/documentation tests still run. CPU passes are not CUDA passes.

### Observable notebook and bounded real inputs

Default Run All invents Amazon-schema fixtures and trains a tiny model. This is not Amazon evidence. The notebook shows source/cohort audit, model/parameter counts, epoch/block losses and progress, A validation selection, fixed A/B test and B development tables, exposures, frozen control, graphs and reload/fork checks. Create-only private evidence lives under fresh `runs/neural-{mode}-*`: `audit.json`, `acquisition.json`, `progress/*.json`, `checkpoints/*.pt`, `runtime.json` and shared runner files. Runtime includes device/Torch/thread/determinism metadata, wall time and cumulative peak process RSS (not incremental model memory); CUDA additionally records peak Torch allocated/reserved bytes (not total process VRAM).

For authorized official ID CSV or review JSONL files (gzip supported), set paths before a fresh kernel:

```sh
CP4285_DEVICE=cpu CP4285_NEURAL_MODE=audit \
CP4285_AMAZON_A="$PWD/data/amazon2023-prefix/Electronics.csv" \
CP4285_AMAZON_B="$PWD/data/amazon2023-prefix/Movies_and_TV.csv" \
uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Audit does not train. Change only `CP4285_NEURAL_MODE=real` after approving the audit. Optional adjacent `.source.json` acquisition sidecars are checked against full local file SHA256/size and embedded in prepared/run provenance. Loader fingerprints separately hash the **scanned prefix**, not the full local file. No acquisition is hidden in the notebook.

Real/audit bounds: 50,000 rows/category, **256 lexical eligible users/category**, two initial A epochs, two four-target continuation blocks/condition. The original 32-user real audit produced zero A/B test targets; the parent explicitly approved **one pre-metric data-coverage adjustment to 256**, without changing catalogue/split/budget rules. Fixture mode retains 32. If bounds cannot support the protocol, fail and ask; never recycle targets, choose users using metrics or silently widen bounds. [Scratchpad](RESEARCH-SCRATCHPAD.md) records the failure, approval, provenance and actual results.

The protocol retains all ratings, predicts the next reviewed product, freezes training-only known catalogues and hash-disjoint dev/test users, and reports full-catalogue @1. Prefix/activity/cohort bias, unknown-item exclusions, fixed anchors containing future-update context, A-before-B block order, eight continuation targets/branch and one seed are serious limits, not approved final benchmark defaults. B development is diagnostic only; it does not select the initial A checkpoint. No tuning on retention/adaptation tests.

### Model/state policy

SASRec-style, not an original reproduction: shared causal pre-norm Transformer/positions, domain-specific tied item tables, final-position full-softmax CE, **one Adam step/target in order (no batching)**, owned device RNG for input-only dropout. All model/input/candidate/target/mask/noise tensors and Adam state reside on the chosen backend. CUDA uses `capturable=True` (including step counters on GPU); CPU uses non-capturable Adam. Scoped initialization restores process CPU/selected CUDA RNG. Full-state forks own independent weights, optimizer, counters and dropout RNG. Adam resets once at continuation and persists thereafter.

`SASRecRecommender.save(path)` is create-only. `load(path, device='cpu'|'cuda'|'auto')` uses trusted `weights_only=True`, maps tensors to the requested backend and requires **the same backend as saved**. CPU/CUDA generator algorithms are not interchangeable, so cross-backend full-state loading is rejected, not labelled equivalent. CUDA ordinals can remap within CUDA. Format 3 intentionally rejects formats 1/2; no compatibility layer. RNG ByteTensor metadata is CPU-transferred for Torch's state API. Tests compare the next update across forks/reloads, not just ranking outputs. Same-runtime tight numerical tolerances/exact RNG checks are not a cross-device/version bitwise reproducibility promise. Only load trusted local files; checkpoints restore learners, not interrupted schedules.

### Actual evidence, 19 October 2026

The authorized real CPU notebook completed on x1-dev using existing project Torch `2.14.1+cpu`, one Torch thread and 42,640 parameters. Official Electronics/Movies & TV CSV prefixes were acquired by the parent: two HTTP 206 first-8-MiB ranges, pinned upstream revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`, **16 MiB received total, not full datasets**. The loader scanned 50,000 rows/category. Audit: A/B catalogues 1,216/1,230; A development/test 15/8, B 2/4; 512 initial A targets.

Private executed notebook: `runs/neural-real-cpu-executed-20261019.ipynb`; run: `runs/neural-real-1b6e83b494e1/`. Sequential headless Python cell execution completed (not browser/Jupyter-kernel certification). Two A epochs reduced mean training CE 9.5242 → 6.9119, but A development NDCG@1 stayed zero; first tied-best epoch 1 selected. **All 36 retention-A/development-B/adaptation-B records have NDCG/Hit/Recall@1 = 0**, frozen unchanged. B-only development remains zero. Training loss reduction does not establish development learning; these floor scores cannot show successful retention or catastrophic forgetting. No tuning/sweep followed.

Measured runner phase 1.528 s, audit/setup/run 2.498 s, cumulative peak RSS 358,662,144 bytes (~342 MiB). These are this tiny run's process-level measurements, not scale guarantees or full acquisition/import/end-to-end wall time. Reload/fork next-update check passed. A same-protocol repeat executed successfully in an actual Jupyter kernel: `runs/neural-real-cpu-jupyter-8dc5844cc7ca.ipynb`, run `runs/neural-real-e97f8b3da603/`. Five cells (1–5), no errors, two displayed SVGs, identical prepared inputs/validation/all 36 records, reload PASS; runner 1.160 s, audit/setup/run 2.338 s and ~366 MiB peak process RSS. Independent audit inspected both runs and the separate 22-pass/four-CUDA-skip test transcript; no findings, accepted engineering/CPU feasibility only. Browser UI, CUDA execution, real GPU throughput/VRAM and RTX acceptance remain unvalidated. See the scratchpad and AI log for commands, exact limits and audit evidence.

The old trial CLI is retired; its [historical runbook](https://github.com/Hong-yiii/cp4285_reccomender_systems/blob/8b9edd3/docs/IMPLEMENTATION.md) is not a current API. Agree a separately justified development-learning criterion and larger scientific budget before scaling.
