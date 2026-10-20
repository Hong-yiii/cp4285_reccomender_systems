# CPU/CUDA neural pilot — shared research scratchpad

## Task and authority

The user supersedes the GPU-only requirement: support explicit CPU, CUDA and automatic device selection, default the notebook to CPU, and execute a bounded real Amazon pilot on **this** x1-dev VM (16 vCPU, 15 GiB RAM; GPU busy elsewhere). Dataset acquisition is authorized, initially bounded to approximately 200 MB combined; the parent owns acquisition and PR #12 publication. This worker is the sole source/document/scratchpad writer. No staging, commit, push, global install, driver changes, test-set tuning or silent protocol changes.

The main deliverable is `notebooks/02_neural_continual_learning.ipynb`, plus a privately saved executed real notebook and create-only run evidence. Invented fixture evidence must not be called Amazon evidence. Scientific stability/non-learning are valid outcomes.

## Acceptance checklist

- [x] Model supports `cpu`, `cuda`, `auto`; explicitly requested CUDA fails rather than falling back.
- [x] Device placement, owned RNG, process RNG preservation, continuation/forks/checkpoints implemented and CPU-tested; CUDA hardware remains unvalidated.
- [x] Optional nonconflicting CPU/CUDA uv extras/index configuration; no global install.
- [x] CPU lifecycle/notebook tests execute; unavailable CUDA checks explicitly skip.
- [x] LSP, Ruff, full tests, lock/config and no-staged-files checks recorded.
- [x] Official bounded source provenance, transfer sizes and local hashes recorded; data remain ignored.
- [x] Existing protocol audited; one explicit parent-approved coverage adjustment, no silent scientific changes.
- [x] Bounded real CPU notebook executed and privately saved; actual timing/memory/progress and A/B development/retention outcomes recorded.
- [x] Current docs/context/AI log/RTX handoff/ignored PR description supersede GPU-only requirement; historical notes stay historical.
- [x] Final independent auditor `936e686a` checked source, actual executed notebook/results/provenance and scientific claims: no findings; engineering + bounded real-CPU feasibility only.
- [x] Parent reviewed implementation, independently reran final checks and confirmed private artifacts remain excluded; publication to draft PR #12 follows acceptance.

## Chronological work and evidence

### 19 October 2026 — orientation

- Initial working tree was clean. Confirmed hostname `x1-dev`, account `hongyilin`, KVM and VM private address against `/AGENTS.md`.
- Read `CONTEXT.md`, project/global guides, implementation/building/RTX handoff, SASRec implementation, loader, lifecycle/config tests, both runtime/package settings and ignored PR description.
- `free -h`, `nproc`, `df -h /`: 15 GiB RAM, 16 vCPU, 181 GiB available disk. Existing project virtual environment has Torch `2.14.1+cpu`, CUDA unavailable. No dependency installation performed.
- Existing GPU-only code requires reversal at the shared model/device/RNG/optimizer/checkpoint boundary, not per-caller CPU workarounds. Notebook currently fails before data in every mode; this is the superseded requirement.
- Existing real protocol: at most 50,000 rows/category; 32 lexical users with five pre-holdout events; two initial A targets/user; two fresh continuation targets/user; immediately following known-catalogue heldout item; hash-disjoint dev/test users; two four-target blocks/condition; first-best A development NDCG@1 selection across two epochs.
- Parent is acquiring bounded official Electronics and Movies & TV 0-core CSV prefixes at upstream revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`. No local source paths supplied yet; worker has not downloaded data independently.

### 19 October 2026 — data coverage audit and authorized adjustment

- Parent acquired **16 MiB total** with two HTTP 206 ranges, not full categories. Files are ignored `data/amazon2023-prefix/{Electronics,Movies_and_TV}.csv`, with hash/schema/range provenance in matching `.source.json` files. Official upstream: `McAuley-Lab/Amazon-Reviews-2023`, pinned revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`, `benchmark/0core/rating_only/`.
- Electronics: received 8,388,608 bytes, saved 8,388,568 newline-trimmed bytes / 144,630 rows; SHA256 `49240422822a014d8e190f9b9eb8648a512e42e958463da65c5f0b49bc1477fa`.
- Movies & TV: received 8,388,608 bytes, saved 8,388,593 bytes / 144,634 rows; SHA256 `4b884414c4d32107b9873a5497409464f7cb1871913407ca1c03e830858a2cd4`.
- **Original 50,000-row / 32-user audit FAILED**: A/B each had one development target and zero test targets. A excluded 29 unknown-item heldouts + two users without a future event; B excluded 28 + three. No training/metrics were run, no bounds silently changed. Parent independently reproduced failure.
- Requested a scientific decision. Parent explicitly authorized **one** pre-metric coverage adjustment: `max_users=256`, same 50,000-row bound, frozen training-only catalogue/hash splits, two initial epochs and eight continuation targets/condition unchanged. No further alternatives were probed.
- **Authorized 256-user audit PASSED**: 512 initial A targets; A/B each 512 available fresh continuation targets (only eight consumed per nonfrozen branch). A catalogue 1,216, development/test 15/8, excluded 199 unknown labels + 34 no-future. B catalogue 1,230, development/test 2/4, excluded 213 + 37. Selected cross-domain user overlap 66; histories remain separate. A/B timestamp ties dropped: 0/4. Severe exclusion and tiny B development/test cohorts remain limitations, not something to tune away.
- Notebook explicitly declares mode-specific 32 fixture / 256 real-audit cohort, approved change and failed original audit. It hash-verifies acquisition sidecars when present and embeds provenance in the prepared manifest and `acquisition.json`.

### 19 October 2026 — device implementation and first checks

- Added shared `resolve_device` / `device_preflight`, CPU default and explicit fatal CUDA. Initialization preserves process RNG while seeding only its backend. Owned dropout generator stays on device. CPU Adam is non-capturable; CUDA Adam keeps capturable step tensors on GPU. Continuation resets once.
- Format-3 trusted create-only checkpoints preserve full learner state and restore on the same CPU/CUDA backend; incompatible RNG algorithms make cross-backend full-state loading an explicit error. CUDA ordinal remapping remains supported. Formats 1/2 are intentionally rejected.
- `uv lock`: resolved 38 packages with mutually exclusive CPU `neural` / CUDA `neural-cuda` extras and explicit official indexes. `uv sync --locked --extra neural --dry-run` needs only project metadata refresh, no Torch download. Both extras together correctly reject before install (expected exit 2).
- Primary LSP on three changed Python files: zero errors. First `uv run --no-sync pytest -q -rs`: **22 passed, four explicit CUDA skips**, including real CPU lifecycle and fixture notebook execution. First Ruff lint passed; format check found three files needing formatting, now formatted; final checks still due.

### 19 October 2026 — executed real CPU notebook

Exact launch (from repository root):

```sh
CP4285_NEURAL_MODE=real CP4285_DEVICE=cpu CP4285_TORCH_THREADS=1 \
CP4285_AMAZON_A="$PWD/data/amazon2023-prefix/Electronics.csv" \
CP4285_AMAZON_B="$PWD/data/amazon2023-prefix/Movies_and_TV.csv" \
uv run --no-sync python local/run_neural_notebook.py \
  --output runs/neural-real-cpu-executed-20261019.ipynb
```

- **Completed, exit 0.** Private executed notebook: `runs/neural-real-cpu-executed-20261019.ipynb`; exact create-only run: `runs/neural-real-1b6e83b494e1/`. `local/run_neural_notebook.py` sequentially executes shared-namespace code cells and captures outputs/errors with stdlib only. This is headless Python evidence, **not** Jupyter kernel/browser certification.
- Existing project Torch `2.14.1+cpu`, device CPU, one intra-op thread (16 inter-op threads recorded), 42,640 trainable parameters, seed 7. No GPU or new numerical dependency used. Existing Torch missing-NumPy warning is nonfatal; this implementation does not require NumPy.
- 512 A targets per initial epoch, 1,024 supplied during selection. Mean target training CE **9.524186 → 6.911935**, but development-A NDCG/Hit/Recall@1 **0 → 0**. First tied-best epoch **1** selected, whose checkpoint has 512 targets seen. Training CE reduction is not development generalization.
- **All 36 A retention / B development / B adaptation records have NDCG, Hit and Recall@1 = 0**, and change from initial = 0. Frozen remains unchanged. Development-B has only two examples and stays zero, including B-only. This is a real-data pipeline feasibility result; **neither A/B development learning nor meaningful retention/forgetting is established**. No metric, budget, seed, candidate set or protocol retuning followed.
- Nonfrozen branches each consume eight targets total: A-only A8; mixed A4+B4; B-only B8. B-only block CE 8.278268 → 8.410684 is not evidence of learning. No stream recycling; available 512 fresh targets/domain are not silently all consumed.
- Audit/setup/run elapsed **2.497813 s**; runner phase (training/evaluation/checkpoint observer) **1.527850 s**. Notebook final elapsed including disposable reload check **2.511 s**. These clocks begin after import/preflight and exclude source acquisition; not full acquisition/import/end-to-end wall time.
- Cumulative process peak RSS **358,662,144 bytes (~342.05 MiB)** during the run, not incremental model memory; CPU GPU-allocation fields are null. No throughput/VRAM scaling claim.
- Saved 39 progress records, 13 create-only full-state checkpoints, prepared/manifest/results/audit/acquisition/runtime JSON and two SVGs. Source sidecars/hash/size verified. Disposable reload/fork next-update equivalence **PASS**; original caller remains untrained. Experiment budgets/results were not modified by that software check.

### 19 October 2026 — validation after implementation

- `uv sync --locked --extra neural`: passed; refreshed only the editable project package, **no Torch download/global install**.
- `uv run --locked --extra neural python -c "from cp4285.models.sasrec import device_preflight; print(device_preflight('cpu'))"`: actual CPU tensor/backward preflight passed.
- `uv run --locked --extra neural pytest -q -rs`: **22 passed, four explicit CUDA skips**, 6.28 s. Coverage includes real CPU default/auto selection, explicit CUDA fatal-before-data in fixture/audit/real, causal/actual synthetic learning, device-local optimizer/tensors, exact target steps, pure scoring/process RNG, forks, reset-once optimizer and trusted checkpoint next-update equality/backend/format/overwrite rejection, frozen controls, hostile-parent fixture notebook with actual CPU default.
- `uv run --locked --extra neural pytest -q tests/test_documentation.py`: **two passed**, including source/example execution and exact mutually exclusive extra/index configuration.
- `uv run --locked --extra neural ruff check src tests`: passed. `uv run --locked --extra neural ruff format --check src tests`: passed, 16 files formatted. `uv lock --check`: passed, 38 packages resolved.
- `git diff --check` and `git diff --cached --quiet`: passed; nothing staged. Final handoff repeats after document updates.
- Artifact assertions passed: all executed code-cell sources equal the public notebook, public outputs/counts cleared, private cells executed without errors; 36 actual zero-score records; correct cohort counts; full local source hashes/sidecar provenance; 39 progress JSONs and 13 checkpoints. No private IDs printed/published.
- `uv sync --locked --extra neural-cuda --dry-run`: passed, plans replacing CPU Torch with CUDA Torch and CUDA dependencies rather than installing conflicting Torch builds; **dry-run only**, no CUDA downloads/install. Existing environment remains CPU.
- Reference HTML parser check passed: unique/resolving anchors and existing local links. New dated checkpoint/top notice updated; earlier sections/CSS preserved.
- Final post-document `git diff --check`, empty staged diff, Ruff lint/format and `uv lock --check` passed. `git ls-files 'data/*' 'runs/*' 'local/*'` is empty; `git check-ignore` confirms sources/sidecars/executed notebook/local scripts/PR draft are ignored.

### 19 October 2026 — actual Jupyter kernel and parent final verification

- Parent executed the same fixed real-data protocol using an actual local IPython/Jupyter kernel via project-local `uv run --locked --extra neural --with nbclient --with ipykernel python local/execute_jupyter_pilot.py`, with the same real-mode/CPU/thread/source environment as above. Optional kernel packages were isolated by uv; no global install. Private executed notebook: `runs/neural-real-cpu-jupyter-8dc5844cc7ca.ipynb`; run: `runs/neural-real-e97f8b3da603/`. Persisted log: `.pi/tasks/session-4866-4866/b76485bb6.output`.
- All five actual kernel cells executed (counts 1–5), no errors, two SVG display outputs, reload/fork PASS. Same prepared inputs/model/source fingerprints/validation/all 36 result records as the original run; this is a reproducibility check, not a new tuned experiment. Runner phase 1.160 s, audit/setup/run 2.338 s, cumulative peak kernel-process RSS 383,340,544 bytes (~365.58 MiB); final notebook elapsed 2.382 s. Acquisition/import time remains excluded. Browser UI was not exercised.
- Added 11 stable unique public notebook cell IDs to satisfy nbformat 4.5; cell source text and scientific behavior are unchanged. Outputs stay private/ignored. The first kernel launch warned about the pre-existing missing IDs; this source metadata is now corrected.
- Parent independently reran locked CPU tests, Ruff lint/format, lock consistency and whitespace checks: **22 passed, four explicit CUDA skips in 6.70 s**, all checks passed. Persisted transcript: `.pi/tasks/session-4866-4866/b12b04ad1.output`.
- Independent auditor follow-up `2dd94fbd` directly inspected both logs, executed kernel notebook, matching source/prepared fingerprints and fixed-protocol repeat records. **No findings; engineering and CPU pipeline feasibility accepted.** This closes prior real-kernel and separate test-transcript evidence gaps. Browser UI and CUDA hardware remain unvalidated; floor scores still do not establish generalization or forgetting.
- The implementation worker reached its orchestration time limit after code/tests/real execution and initial final-audit documentation were complete. Parent took over final documentation/publication; no work was discarded and the independently audited implementation is unchanged.

## Audit findings / resolution log

Final independent audit and follow-up completed. Acceptance is **engineering implementation and bounded real-CPU feasibility only**, not demonstrated development learning/retention/forgetting. Parent final validation and actual Jupyter-kernel execution also passed.

| Finding | Status | Resolution / evidence |
| --- | --- | --- |
| Initial independent structural review `4fba65e3`: no concrete blockers | Preliminary, superseded by final audit | Initial source/protocol review, not independently sufficient acceptance. |
| Final actual-source/notebook/results audit `936e686a`: **no findings** | Passed for engineering + bounded real-CPU feasibility | Independently matched source/prepared/acquisition hashes, actual executed notebook, 39 progress/36 metric records/13 checkpoints and aggregate holdout isolation. Confirmed losses, zero ranking results and exact branch exposures; reviewed CPU/CUDA/RNG/checkpoint/dependency semantics. |
| Actual Jupyter-kernel/test-transcript follow-up `2dd94fbd`: **no findings** | Passed; previous evidence gaps closed | Independently verified five real kernel cells, two SVG displays, reload PASS, identical fixed-protocol inputs/results, stable cell IDs and parent transcript with 22 passes/four CUDA skips. Browser UI and CUDA remain untested. |

## Blockers and evidence limits

- Real CPU pipeline/private evidence, actual Jupyter-kernel execution, parent final checks and independent follow-up audit completed; **no review findings**. Engineering feasibility is accepted independently; research learning/retention is unestablished. Draft PR #12 is the publication target.
- CUDA hardware execution is unavailable on this VM. CPU evidence cannot establish CUDA correctness; CUDA-specific checks must be explicit skips, with target-machine validation remaining a limit.
- If the supplied prefix cannot support current cohort/holdouts/budgets, stop and request a scientific decision rather than widening the protocol.
