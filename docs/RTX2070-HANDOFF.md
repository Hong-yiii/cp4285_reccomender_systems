# RTX 2070 session handoff — CUDA-required neural pilot

## Status and evidence boundary

The user explicitly requires **NVIDIA CUDA execution, with no CPU execution/fallback**, for `SASRecRecommender` and `notebooks/02_neural_continual_learning.ipynb` (including notebook audit mode). The standard-library experiment/counting packages and standalone Amazon loader remain GPU-independent. Read `AGENTS.md`, `CONTEXT.md`, `docs/IMPLEMENTATION.md`, then the notebook.

CUDA device placement, owned CUDA dropout RNG, scoped initialization, CUDA Adam state, trusted portable checkpoint loading and an early real tensor/backward preflight are implemented. **GPU tests and notebook execution have NOT run on this development VM.** Prior 17-test CPU/fixture evidence belongs to the old implementation, not CUDA validation. No real Amazon audit/training, measured GPU throughput/VRAM, or meaningful B development learning is established. Generated historical CPU artifacts remain ignored and are not included in Git.

## Reproducible runtime and official evidence

The `neural` extra pins **`torch==2.14.1+cu126`** to the explicit official **`https://download.pytorch.org/whl/cu126`** index. This keeps the installed development environment's Torch release rather than downgrading to 2.6 merely to obtain CUDA. Only Linux/Windows have this neural dependency; macOS has no supported CUDA execution path. The core still installs without Torch. No claim of a complete security audit or indefinite support follows from this version choice.

Primary sources checked for this change:

- [NVIDIA compute-capability table](https://developer.nvidia.com/cuda-gpus): GeForce RTX 2070 is **7.5 (Turing / SM75)**.
- [Official cu126 Torch wheel index](https://download.pytorch.org/whl/cu126/torch/): exact `torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl` and `torch-2.14.1+cu126-cp312-cp312-win_amd64.whl` entries establish Python 3.12 Linux x86_64 / Windows AMD64 availability; `uv.lock` records their hashes. Linux needs glibc >=2.28 ([PyTorch prerequisites](https://pytorch.org/get-started/locally/)). These are distribution facts, **not an executed hardware test**.
- [PyTorch maintainer compatibility statement](https://discuss.pytorch.org/t/soumitra-dutta-oxford-how-do-i-install-pytorch-with-the-right-cuda-version-so-my-gpu-works/224733): cu126 binaries support SM52 through SM90, encompassing SM75. The [2.10 build recipe](https://github.com/pytorch/pytorch/blob/v2.10.0/.ci/manywheel/build_cuda.sh) independently includes `7.5` in cu126's architecture list; it is historical corroboration, not proof about a newer wheel. The installed wheel's architecture list and real forward/backward still need checking on the target.
- [NVIDIA CUDA 12.6 release notes, Tables 2–3](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html): CUDA 12.x **minor-compatibility floor** is Linux 525.60.13 / Windows 528.33; CUDA 12.6 GA's corresponding toolkit drivers are Linux 560.28.03 / Windows 560.76. These are different concepts, not a guarantee that the minimum driver runs every wheel/library feature. [Minor compatibility caveats](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) include PTX and newer driver features. Use a maintained driver compatible with the actual GPU/runtime; do not change drivers/system software without authorization. `nvidia-smi`'s CUDA field is driver capability, not `torch.version.cuda`.
- [Official previous-version instructions](https://pytorch.org/get-started/previous-versions/) also list Torch 2.6.0/cu124 on Linux/Windows, but that older alternative is **not** this project's installation path. There is no dual backend.

## Target-machine acceptance: run in order

From the repository root in the target machine's ordinary account, with Python 3.12 and uv. Commands below use a POSIX shell (Linux, or an appropriately configured WSL CUDA environment); native Windows uses the same `uv` commands but its own environment-variable syntax. Do not copy `.venv`, credentials or private sessions between machines.

```sh
nvidia-smi
uv sync --locked --extra neural
uv run --locked --extra neural python -c "import json; from cp4285.models.sasrec import cuda_preflight; print(json.dumps(cuda_preflight(), indent=2))"
uv run --locked --extra neural pytest -q -rs
uv run --locked --extra neural ruff check src tests
uv run --locked --extra neural ruff format --check src tests
CP4285_NEURAL_MODE=fixture uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

1. Confirm GPU identity, available VRAM, OS/driver and runtime. The preflight really allocates on CUDA, multiplies tensors, backpropagates and synchronizes. Its failure is fatal and actionable, not a request to use CPU. Select a GPU with `CUDA_VISIBLE_DEVICES` before launching; the model uses the current visible CUDA device. On RTX 2070 inspect the reported capability `[7, 5]` and compiled architectures for `sm_75`.
2. The four GPU tests **must execute**, not skip. They cover causal scoring/actual updates, exact one-step-per-target counts, inactive A table versus shared B updates, unchanged process CPU/CUDA RNG, pure scoring, independent full-state forks, same-runtime checkpoint **next-update** equivalence, all Adam tensors on CUDA and frozen controls. The offline notebook test forcibly uses invented fixture inputs even under hostile parent real-data variables. Tight floating tolerances allow CUDA numerical variation; cross-device/version bitwise equality is not promised. Investigate failures; do not simply loosen tolerances.
3. Restart the notebook kernel and Run All in fixture mode. Verify GPU/runtime/determinism metadata, synchronized phase wall times and `runtime.json` peak allocated/reserved bytes. The peak measures Torch allocations during the experiment, not all process VRAM. The tiny architecture is not a full-data memory/throughput guarantee. Per-target updates and per-query scoring can underutilize a GPU; **do not introduce batching**, which changes optimizer-step semantics.
4. Checkpoints are create-only format 2, restricted `weights_only=True`, remapped to the **current required CUDA device**, independent of the saved CUDA ordinal. Adam uses `capturable=True` so step counters as well as moments stay on CUDA. Serialized RNG state is a CPU ByteTensor by Torch API; converting this metadata is not CPU model execution. Old CPU format-1 checkpoints are intentionally rejected. Only load trusted local files; these checkpoints restore learners, not interrupted runner schedules.

## Authorized real-data path

Ask for approved local A/B files if unavailable; none are bundled. Audit before training:

```sh
CP4285_NEURAL_MODE=audit \
CP4285_AMAZON_A=/absolute/path/to/Electronics.csv.gz \
CP4285_AMAZON_B=/absolute/path/to/Movies_and_TV.csv.gz \
uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Audit mode still requires CUDA/preflight but does not train. The standalone `AmazonSource` API and its tests need no GPU. Inspect cohort, candidate sizes, exclusions/timestamp ties and fresh-target budgets; the bounded ordinal/future-inventory protocol is not representative sampling or strict calendar-time deployment. After audit approval, repeat the same command with `CP4285_NEURAL_MODE=real`. No downloads, split changes, stream recycling, silent bound widening or expensive sweeps are authorized. Establish A and B development learning before scaling; never tune on retention/adaptation tests.

Save new executed notebook outputs and create-only artifacts privately; record exact GPU/driver/runtime, commands, test output and actual versus unvalidated claims in context/AI log. Do not commit datasets, original user IDs, checkpoints or private run outputs. Independent read-only GPU structural review found no static blockers. Target-machine execution remains the outstanding acceptance gate; static review is not CUDA execution evidence.

## Checks on a machine without CUDA

Normal core checks need no Torch/GPU:

```sh
uv sync --locked
uv run --locked pytest -q -rs
uv run --locked ruff check src tests
```

Neural tests explicitly skip without Torch or usable CUDA. If this checkout already has CPU Torch installed, validate only rejection/configuration without syncing CUDA wheels:

```sh
uv run --no-sync pytest -q -rs
uv run --no-sync pytest -q tests/test_sasrec.py -k 'rejection or rejects_cpu'
uv lock --check
```

This deliberately uses the existing environment only for negative-path/core checks. It is **not** installation or execution evidence for the pinned CUDA build. A CUDA sync was not run on the development VM to avoid a large, unusable wheel download.
