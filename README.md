# CP4285 · Recommender research boilerplate

**Research question:** How does recommendation quality on previously learned tasks change during continued updates, and how does this depend on model capacity and the incoming data distribution?

**16 October 2026 — reset:** the September pipelines were exploratory trials and have been removed from the active implementation. The current project is a small, tested **two-package foundation**, not a completed research experiment. A bounded local Amazon loader and optional-install CPU/CUDA SASRec-style learner are available in a proposed pilot notebook. A bounded real CPU run completed on 19 October: all heldout @1 scores remained zero, so development learning and meaningful retention/forgetting are **not established**. See the [research scratchpad](docs/RESEARCH-SCRATCHPAD.md) for measured evidence and review status. Scientific safeguards carry forward; old architecture and experimental defaults do not.

## Start here

1. **[Develop with the two packages](docs/PACKAGES-AND-BUILDING.md)** — copyable model and evaluator code, an executable contract check, and the notebook recipe.
2. [October meeting synthesis](docs/MEETING-2026-10-15.md) — supported direction versus unresolved choices.
3. [Current context](CONTEXT.md) and [running the boilerplate](docs/IMPLEMENTATION.md).
4. [Dependency-free shared notebook](notebooks/01_shared_experiment.ipynb) and [observable neural pilot](notebooks/02_neural_continual_learning.ipynb) ([CPU/CUDA instructions](docs/IMPLEMENTATION.md#neural-pilot-cpu-or-cuda), [RTX 2070 handoff](docs/RTX2070-HANDOFF.md)).

## The two packages

| Path | Owns |
| --- | --- |
| `src/cp4285/experiment/` | Data/protocol ABC, immutable inputs, model ABC, scheduling, shared selection/evaluation, safe artifacts and plots |
| `src/cp4285/models/` | Model implementations and model-side ablations; counting isolation control and optional-install CPU/CUDA SASRec-style learner |
| `notebooks/` | Reader-facing executable explanation, importing both packages |
| `tests/` | Contract/lifecycle checks; executable documentation and notebook examples |
| `docs/` | Methods, decisions, historical research reasoning and AI-use log |
| `slides/` | Team Google Slides pointer and historical authored figures |
| `reference.html` | Preserved historical walkthrough with dated current-status checkpoints |
| `local/`, `data/`, `runs/`, `reports/` | Ignored private sources, inputs and generated evidence |

One Python distribution, two responsibility packages. No service, plugin framework or core numerical dependency. The core/counting samples need no GPU. **The user supersedes the GPU-only requirement: neural CPU and CUDA are supported; the notebook defaults to CPU.** Explicit `device='cuda'` remains fatal if CUDA is unavailable, never a silent fallback. Models implement the common interface; they do not redefine the benchmark.

## Run the small check

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
```

These checks execute the guide's Python examples and the dependency-free notebook cells in fresh processes. With optional Torch installed, CPU neural lifecycle/notebook checks execute; CUDA cases explicitly skip without hardware. Without Torch, the neural test module skips; loader/core contracts remain tested. They also verify that the documented model/evaluator definitions match maintained source. All data are invented; nothing is downloaded.

```python
from cp4285.experiment import SyntheticSource, run, plot_results
from cp4285.models import CountingRecommender

data = SyntheticSource().load_data()
result = run(CountingRecommender(), data,
             output="runs/my-synthetic-check", initial_epochs=3, k=1)
plot_results("runs/my-synthetic-check/results.json", task="retention-A")
```

Choose a new output directory for each run. The tiny fixture's @1 is not the research metric choice. The counting control keeps A and B independent; its curves are not Amazon results or evidence about neural forgetting.

For the interactive walkthrough (optional launcher installation through uv):

```sh
uv run --with jupyterlab jupyter lab notebooks/01_shared_experiment.ipynb
```

## Neural pilot: CPU or CUDA

Python 3.12 on Linux/Windows; CPU is the notebook/model default:

```sh
uv sync --locked --extra neural
uv run --locked --extra neural python -c "from cp4285.models.sasrec import device_preflight; print(device_preflight('cpu'))"
uv run --locked --extra neural pytest -q -rs
CP4285_DEVICE=cpu CP4285_NEURAL_MODE=fixture uv run --locked --extra neural --with jupyterlab jupyter lab notebooks/02_neural_continual_learning.ipynb
```

`neural` pins official `torch==2.14.1+cpu`. For CUDA, select **instead** `--extra neural-cuda` (official `2.14.1+cu126`), and set `CP4285_DEVICE=cuda`. The extras are mutually exclusive; no global install. Model/API device choices are `cpu`, `cuda`, `auto`; auto selects CUDA if available, CPU otherwise. Explicit CUDA fails before notebook data/artifacts when its tensor/backward preflight fails. [RTX compatibility and acceptance](docs/RTX2070-HANDOFF.md) remain hardware gates, not CPU-tested claims. One optimizer step per target, no batching; full-state checkpoints require the same CPU/CUDA RNG backend.

Default Run All uses **invented fixtures**. The authorized real run used bounded official Electronics/Movies & TV prefixes (16 MiB acquired total, 50k rows/category scanned), 256-user coverage-adjusted cohorts, two initial A epochs and eight continuation targets/branch. It completed in 1.528 s runner time with ~342 MiB peak process RSS. Training CE decreased but A development and every A/B @1 record stayed zero. This is real-data **pipeline feasibility**, not learned generalization or a forgetting result. Source provenance, failed original 32-user audit, approved adjustment, private executed artifact paths and limits are recorded in the [runbook](docs/IMPLEMENTATION.md#actual-evidence-19-october-2026) and scratchpad. CUDA execution remains unvalidated.

## Standing boundary

Keep identities, holdouts, fixed candidates, validation-only selection and independent branch state intact. Separate implemented behavior, research proposals and measured evidence. No real-data acquisition or expensive run follows automatically from a documented idea. Private transcripts, course/team PDFs, datasets, model weights and notebook outputs containing personal identities stay out of Git.

The earlier SASRec/classical code and its runbook remain recoverable from [the pre-reset Git snapshot](https://github.com/Hong-yiii/cp4285_reccomender_systems/tree/8b9edd3). Existing uncommitted trial work was preserved locally before removing its active entry points. Older documents and figures are dated evidence, not current execution instructions. See [CONTEXT.md](CONTEXT.md) for the remaining research decisions and [AI_LOG.md](docs/AI_LOG.md) for assistance disclosure.
