# RTX 2070 session handoff — draft neural pilot

## Status and goal

User wants to run observable neural continued-training research on real Amazon data on an RTX 2070. Start with `AGENTS.md`, `CONTEXT.md`, `docs/IMPLEMENTATION.md`, then `notebooks/02_neural_continual_learning.ipynb`.

**This draft is runnable on CPU, not CUDA-ready.** An RTX 2070 is a Turing GPU (compute capability 7.5), normally with 8 GB VRAM. The tiny default model should be modest in memory, but CUDA compatibility, VRAM use and throughput have not been measured. Do not promise a full-data run fits. The `neural` extra explicitly installs CPU Torch; moving this checkout to a GPU machine alone does not enable acceleration.

Existing evidence: 17 passing CPU tests, Ruff and fresh-interpreter Pyright clean, independent review completed. Executed fixture notebook and artifacts live in ignored `runs/` on the development machine and are **not included in Git/this PR**. Reproduce them with default notebook Run All. No real Amazon audit, download, training, GPU validation or meaningful B-development learning is established.

## Next agent: do these in order

1. Identify the new machine, OS, Python 3.12 environment, driver and GPU with `nvidia-smi`. Preserve unrelated work. Do not copy the development machine's environment or credentials. No driver/system upgrades without authorization.
2. Read `pyproject.toml`: `neural` resolves through the explicit `pytorch-cpu` index. Add a minimal documented, reproducible CUDA installation option suitable for the actual OS/driver and RTX 2070. Preserve the offline CPU path. Choose a PyTorch build that supports Turing; verify a CUDA tensor operation on the card rather than assuming `torch.cuda.is_available()` alone proves compatibility. Never modify global Python.
3. Add explicit device selection to `SASRecRecommender` and notebook settings. In `src/cp4285/models/sasrec.py`, audit **all** allocations: model, history/candidate/target tensors, positions, causal mask and dropout noise. Keep owned RNG independent of process RNG and branches. A CPU-owned RNG with explicit noise transfer is a simple option. Preserve fork, score purity, inactive-domain parameter and optimizer rules. Do not blindly replace every `cpu` string.
4. Make checkpoint loading explicitly portable across CPU/GPU. Preserve weights-only loading, create-only writes, optimizer and RNG state, counters and continuation policy. Check same-device next-update equivalence; compare CPU/GPU scores with appropriate numerical tolerances, not bitwise equality. Record device, GPU, Torch/CUDA versions and determinism policy in artifacts. Do not claim cross-device bitwise reproducibility.
5. Run existing CPU tests first, then small CUDA tests for scoring, one real backward/update, branch independence, save/reload and optimizer-state device placement. Verify frozen controls remain unchanged. Keep offline notebook tests forced to fixture mode regardless of parent environment.
6. Launch Jupyter and run the invented fixture on CUDA. Observe timing and peak allocated/reserved memory. The current model uses one optimizer step per supervised target and per-query evaluation; GPU utilization may be poor. Do not silently batch updates: batching changes optimizer-step semantics. If batching is needed, declare the protocol change and test target accounting and loss normalization.
7. Ask for the actual local A/B category file paths if unavailable. No datasets are tracked or bundled. Start with `CP4285_NEURAL_MODE=audit` and `CP4285_AMAZON_A/B`. Audit the bounded cohort, candidate sizes, time/tie rules, exclusions and fresh-target availability. Read the notebook's disclosed ordinal/future-inventory assumptions; this is not a strict global temporal benchmark. Do not silently widen sampling, recycle streams or change splits to obtain convenient results.
8. After the audit passes, run a tiny real pilot using `CP4285_NEURAL_MODE=real` and explicit CUDA selection added above. Establish A learning and B development learning before scaling. Current fixture settings are two initial epochs, two four-target continuation blocks, NDCG@1, one seed/width—not a final research budget. Choose a meaningful real-data cutoff/budget on development evidence **before inspecting test curves**. Report retention and adaptation together; improvement, stability and decline are valid.
9. Save a newly executed notebook with outputs and create-only private artifacts. Update context/AI log and this PR with exact commands, hardware, tests, dataset provenance, actual versus unvalidated claims, and remaining limits. Do not commit datasets, original user IDs, checkpoints or private run outputs. Do not run large sweeps without an agreed budget.

## CPU baseline / notebook instructions already available

```sh
uv sync --locked --extra neural
uv run --extra neural pytest -q
uv run --extra neural ruff check src tests
uv run --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Default mode is invented fixture. For local real-data preparation only:

```sh
CP4285_NEURAL_MODE=audit \
CP4285_AMAZON_A=/absolute/path/to/Electronics.csv.gz \
CP4285_AMAZON_B=/absolute/path/to/Movies_and_TV.csv.gz \
uv run --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

These commands currently use CPU Torch. Do not present them as CUDA installation instructions. The next agent must supply verified platform-specific CUDA commands after inspecting the RTX machine.
