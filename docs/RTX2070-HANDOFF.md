# CPU pilot and RTX 2070 acceptance handoff

## Current requirement and evidence boundary

**19 October 2026: the user supersedes GPU-only execution. CPU and CUDA are supported; CPU is the model/notebook default.** This VM has 16 vCPU and 15 GiB RAM; its GPU is busy elsewhere. Model `device='cpu'|'cuda'|'auto'` and notebook `CP4285_DEVICE` select a backend. Explicit CUDA is fatal if unavailable or its real preflight fails, never a silent fallback. Auto chooses available CUDA, otherwise CPU. Core/counting/standalone Amazon preparation remain GPU-independent.

The authorized real **CPU** pipeline completed: existing Torch `2.14.1+cpu`, one thread, 42,640 parameters, bounded official Electronics/Movies & TV prefixes (16 MiB acquired total), 50k rows/category scanned. Original 32-user audit had zero test targets; parent approved exactly one pre-metric adjustment to 256 lexical users with all splits/catalogue rules, two initial epochs and eight continuation targets/condition unchanged. Audit gives A dev/test 15/8, B 2/4; catalogues 1,216/1,230. Mean initial CE 9.5242→6.9119, but A development and every retention-A/development-B/adaptation-B @1 record stayed zero. **Development learning, successful retention and catastrophic forgetting are not established.** No tuning followed. Runner time 1.528 s, audit/setup/run 2.498 s, cumulative peak process RSS ~342 MiB; not full acquisition/import/end-to-end time or a scaling guarantee.

Private notebook `runs/neural-real-cpu-executed-20261019.ipynb`, run `runs/neural-real-1b6e83b494e1/`; headless sequential Python cells completed. An actual Jupyter-kernel repeat also passed: `runs/neural-real-cpu-jupyter-8dc5844cc7ca.ipynb`, run `runs/neural-real-e97f8b3da603/`; identical inputs/results, five kernel cells, two SVG displays. Independent final/follow-up audit found no findings, accepted engineering/CPU feasibility only. Browser UI remains untested. Reload/fork next-update equivalence passed. [Scratchpad](RESEARCH-SCRATCHPAD.md) records provenance, scientific decision, commands and independent audit status. **CUDA hardware tests/notebook have not executed on this VM.** CPU results do not validate CUDA, wheel `sm_75` support, GPU memory or throughput.

## Project-local installation: choose one backend extra

Python 3.12 Linux/Windows, uv, ordinary account; never copy `.venv`, credentials or private sessions between machines. Extras are intentionally mutually exclusive, so do not use `--all-extras`.

CPU:

```sh
uv sync --locked --extra neural
uv run --locked --extra neural python -c "from cp4285.models.sasrec import device_preflight; print(device_preflight('cpu'))"
uv run --locked --extra neural pytest -q -rs
CP4285_DEVICE=cpu CP4285_NEURAL_MODE=fixture uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

`neural` pins **`torch==2.14.1+cpu`** to the explicit official **`https://download.pytorch.org/whl/cpu`** index. `neural-cuda` instead pins **`torch==2.14.1+cu126`** to **`https://download.pytorch.org/whl/cu126`**. Both remain optional for the standard-library core. Plain `uv sync --locked` unselects Torch; CPU lifecycle checks need the CPU extra (or existing compatible project Torch via `uv run --no-sync`). No macOS/MPS backend is claimed. No security audit or indefinite support follows from version selection.

## CUDA reproducibility and official compatibility evidence

Primary sources previously checked for the CUDA distribution (compatibility documentation is **not hardware execution evidence**):

- [NVIDIA compute-capability table](https://developer.nvidia.com/cuda-gpus): GeForce RTX 2070 is **7.5 (Turing / SM75)**.
- [Official cu126 Torch wheel index](https://download.pytorch.org/whl/cu126/torch/): exact `torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl` and `torch-2.14.1+cu126-cp312-cp312-win_amd64.whl` entries establish Python 3.12 Linux x86_64 / Windows AMD64 availability; `uv.lock` records hashes. Linux requires glibc >=2.28 ([PyTorch prerequisites](https://pytorch.org/get-started/locally/)).
- [PyTorch maintainer compatibility statement](https://discuss.pytorch.org/t/soumitra-dutta-oxford-how-do-i-install-pytorch-with-the-right-cuda-version-so-my-gpu-works/224733): cu126 binaries support SM52–SM90, encompassing SM75. The [2.10 build recipe](https://github.com/pytorch/pytorch/blob/v2.10.0/.ci/manywheel/build_cuda.sh) includes `7.5` for cu126; historical corroboration, not proof about a newer wheel. Check the installed architecture list and actual forward/backward on target.
- [NVIDIA CUDA 12.6 release notes, Tables 2–3](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html): CUDA 12.x **minor-compatibility floor** Linux 525.60.13 / Windows 528.33 differs from CUDA 12.6 GA's corresponding toolkit drivers Linux 560.28.03 / Windows 560.76. Neither is a universal wheel-feature guarantee. [Minor compatibility caveats](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) include PTX/newer driver features. Do not change system drivers without authorization. `nvidia-smi` CUDA is driver capability, not `torch.version.cuda`.
- [Official previous-version instructions](https://pytorch.org/get-started/previous-versions/) list older Torch 2.6/cu124, but that is not this project's CUDA installation path. Both project backend extras retain Torch 2.14.1.

## Target-machine CUDA acceptance: run in order

POSIX commands (native Windows needs its own environment-variable syntax):

```sh
nvidia-smi
uv sync --locked --extra neural-cuda
uv run --locked --extra neural-cuda python -c "import json; from cp4285.models.sasrec import cuda_preflight; print(json.dumps(cuda_preflight(), indent=2))"
uv run --locked --extra neural-cuda pytest -q -rs
uv run --locked --extra neural-cuda ruff check src tests
uv run --locked --extra neural-cuda ruff format --check src tests
CP4285_DEVICE=cuda CP4285_NEURAL_MODE=fixture uv run --locked --extra neural-cuda --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

1. Confirm GPU, free VRAM, OS/driver and runtime. Preflight allocates CUDA tensors, multiplies, backpropagates and synchronizes before notebook source/artifacts. Failure is fatal for explicit CUDA. Use `CUDA_VISIBLE_DEVICES` before launch. RTX 2070 should report `[7, 5]` capability and `sm_75` compiled architecture.
2. **Four CUDA parameter cases must execute, not skip.** CPU counterparts run independently. Coverage: causal scoring/actual learning, per-target Adam steps, inactive A table versus shared B updates, process CPU/CUDA RNG preservation, score purity, independent full-state forks, checkpoint next-update equivalence, device-local tensors/Adam state, frozen control and fixture notebook isolation under hostile real-mode parent variables. Tight same-runtime tolerances, exact RNG/purity; investigate failures rather than relaxing tests.
3. Restart-and-Run-All fixture notebook on the target Jupyter project kernel. Check device/runtime/determinism metadata, completed phase timing, RSS and CUDA peak allocated/reserved bytes. Torch peaks are not total VRAM. Per-target/query processing can underutilize CUDA; **do not add batching**, which changes optimizer-step semantics.
4. Format-3 trusted `weights_only=True` checkpoints require **same-backend full-state restoration**. CPU/CUDA RNG algorithms differ; cross-backend load fails clearly. CUDA ordinal remapping is supported. CUDA Adam uses `capturable=True` so step counters/moments stay on GPU; CPU uses non-capturable Adam. RNG state metadata is CPU-transferred for Torch's API. Old formats 1/2 rejected. Checkpoints restore learners, not runner schedule replay.

## Authorized real-input path and next scientific decision

The already acquired bounded source files/provenance stay ignored. Source revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`, official `benchmark/0core/rating_only/{Electronics,Movies_and_TV}.csv`, two first-8-MiB HTTP 206 ranges trimmed to complete lines, **not full datasets**. Notebook verifies optional `.source.json` SHA256/size sidecars and embeds acquisition provenance.

```sh
CP4285_DEVICE=cpu CP4285_NEURAL_MODE=audit \
CP4285_AMAZON_A="$PWD/data/amazon2023-prefix/Electronics.csv" \
CP4285_AMAZON_B="$PWD/data/amazon2023-prefix/Movies_and_TV.csv" \
uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

Audit mode works on either backend without training; standalone loader needs no Torch. Real mode uses the same approved bounds. Inspect exclusions, known-catalogue holdouts and fresh-target counts; do not silently widen rows/cohorts or recycle streams. Current real CPU floor scores do not justify a forgetting comparison. Agree a development-only learning criterion and compute budget before further experiments; no sweeps/tuning against retention/adaptation tests. Prefix/activity selection, fixed future-context anchors, A-before-B mixed block order, eight continuation targets/branch, tiny filtered holdouts and one seed remain confounds/limits.

Save executed notebooks/checkpoints/results privately and create-only. Do not commit original IDs, datasets, weights or private run outputs. Parent owns independent final review and PR #12 publication; this worker does not stage/commit/push. Historical GPU-only notes in AI log remain dated history, not current instructions.
