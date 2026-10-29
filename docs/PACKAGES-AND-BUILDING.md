# Two packages, one learner lifecycle

This integration starts from `main`'s real implementations, not from the reset. The small, standard-library-only teaching baseline remains at [tag `boilerplate-v1`](https://github.com/Hong-yiii/cp4285_reccomender_systems/tree/boilerplate-v1) (`c169032`). Its `feat/boiler-plate` branch is retained. This branch preserves SASRec, the classical kernels, the CLI and existing dataset/artifact safeguards.

## Responsibility boundaries

- **`cp4285.experiment`** owns ingestion, source identities, splits, schedules, the model ABC, evaluation/selection, checkpoints and reports.
- **`cp4285.models`** owns model mathematics, update rules, optimizer state, model RNG and model-side ablations.
- **Notebooks** import these packages to expose assumptions and inspect outputs. They do not implement their own training or ranking code.

One distribution and the existing dependencies are enough. No registry, plugin framework, compatibility package or new dependency was added. Import paths changed; CLI commands and config paths did not.

## The ABC teammates implement

See [`experiment/model.py`](../src/cp4285/experiment/model.py). `Recommender[Batch, Queries, Scores]` is a real Python ABC: an incomplete implementation cannot be instantiated.

```text
train(batch) -> float | None             # one native update; optional loss
begin_continuation() -> None             # transition once, retain learned parameters
score(queries, candidates=None) -> Scores # batch × candidate columns; no learning
describe() -> dict                       # JSON-compatible policy/provenance
fork() -> Self                           # default deepcopy of all owned state
```

`train()` operates on the **same learner** before and after the transition. It must not hide epochs, resampling, stream recycling or refits. Experiment code chooses the supplied batch. `begin_continuation()` is called once per independent branch; repeating it is an error, not another optimizer reset. `score()` must not consume training RNG or mutate model/optimizer state. It returns scores, not ranks or metrics.

Explicit candidate indices are **zero-based columns in a fixed catalogue**, in the requested order. They are not raw product IDs; the experiment owns the vocabulary mapping. `None` means the entire catalogue. Empty, duplicate, fractional or out-of-range candidates are rejected.

### A common lifecycle is not a common research protocol

Unlike the tagged boilerplate, the integration does **not** claim every batch is a tuple of fresh single-target `Example`s. Main already has distinct, meaningful learning units:

| Learner/workflow | Initial training | Continued training | Evaluation remains |
| --- | --- | --- | --- |
| `SASRecLearner` / `experiment.pilot` | One Adam step on sampled users; all nonpadding next-item positions | One Adam step on fresh final targets; Adam reset once | Cutoff cohort, familiar initial catalogue, no seen masking, stable item-index ties |
| `IncrementalQRSVD` / `experiment.classical.stream` | Fresh complete user-row chunks | More fresh rows; retain basis, singular values and NumPy RNG | Leave-last-out, seen masking, classical tie policy |
| Classical `sweep` | Independent batch fits at every contamination level | **Not supported** | Existing classical evaluator |

`MostPop`, `QRSVD`, `Markov`, `MarkovQRSVD` and `SLIST` retain their numerical `fit()`/`score()` APIs. They deliberately do **not** inherit the continual learner ABC or pretend that refitting is continuation. In particular, streamed **PureSVD** is not streamed **MarkovQRSVD**; the latter still retains its full sparse transition table.

Both evaluators now belong to `experiment/`, not to individual models. Their policies remain separate: [`neural_evaluation.py`](../src/cp4285/experiment/neural_evaluation.py) and [`classical_evaluation.py`](../src/cp4285/experiment/classical_evaluation.py). Align the research task, split, candidates and metric policy before comparing model scores. Do not concatenate these results into a comparative benchmark merely because their classes share method names.

## Runnable SASRec lifecycle

From the repository root, run these Python blocks in order with the uv environment (or use the notebook). All inputs below are **synthetic software checks**, not Amazon data.

```python
import random
import torch

from cp4285.experiment import Recommender
from cp4285.experiment.pilot import initial_sequences
from cp4285.models.sasrec import SASRec
from cp4285.models.sasrec_learner import SASRecLearner

torch.set_num_threads(2)
torch.manual_seed(4285)
learner = SASRecLearner(
    SASRec(8, 4, hidden=8, layers=1, dropout=0.2),
    learning_rate=0.001,
    continuation_learning_rate=0.001,
    rng=random.Random(4285),
)
assert isinstance(learner, Recommender)
initial = initial_sequences([
    {"user": "u", "history": [1], "target": 2},
    {"user": "u", "history": [1, 2], "target": 3},
])
learner.train(initial)
query = torch.tensor([[0, 1, 2, 3]])  # item IDs 1..8; zero is padding
before = learner.score(query)
branch = learner.fork()
branch.begin_continuation()
branch.train([{"history": [1, 2, 3], "target": 4}])
torch.testing.assert_close(learner.score(query), before, rtol=0, atol=0)
assert branch.score(query, candidates=(7, 0, 2)).shape == (1, 3)
print(branch.describe())
```

The raw `SASRec(nn.Module)` remains unchanged: its `.train()` toggles PyTorch mode, so use **`SASRecLearner`** for the experiment's training interface. The adapter owns Adam plus Python and CPU/device Torch RNG states. It restores the caller's Torch RNG after each update. Its `fork()` copies optimizer parameter bindings as well as weights and RNG. Scoring restores submodule training modes.

The pilot selects weights on validation Hit@10, with NDCG@10 breaking ties, verifies a saved/reloaded initial checkpoint and measures a frozen control. Continuation starts with fresh Adam state and the end-of-initial-training sampler state, not a resumed best-epoch optimizer. Forking no longer constructs a throwaway randomly initialized network for reload, so do not expect bitwise continuation parity with old pilot runs. The objective, sampling policy and evaluation are preserved; record the code revision for reproducibility.

## Runnable incremental QR-SVD lifecycle

```python
import numpy as np
from scipy.sparse import csr_matrix

from cp4285.models.classical import IncrementalQRSVD

rows = csr_matrix(np.random.default_rng(4285).random((24, 8), dtype=np.float32))
qr = IncrementalQRSVD(k=3, residual_rank=2, seed=4285)
qr.train(rows[:12])
query_rows = rows[:2]
before_qr = qr.score(query_rows)
qr_branch = qr.fork()
qr_branch.begin_continuation()
qr_branch.train(rows[12:])  # NEW user rows, not the cumulative matrix
np.testing.assert_array_equal(qr.score(query_rows), before_qr)
np.testing.assert_array_equal(
    qr_branch.score(query_rows, (7, 0, 2)), qr_branch.score(query_rows)[:, (7, 0, 2)]
)
print(qr_branch.describe())
```

This example checks lifecycle and column alignment, not a cross-model experiment. User rows are a different update/exposure unit from SASRec's targets.

## Add a model or an experiment

1. Copy the structure of the appropriate existing learner into a new module under `models/`; subclass `Recommender` with your native batch, query and score types. Implement all four abstract methods. Keep fitting/optimizer state on the instance; no global mutable learner state.
2. Specify what one `train()` call consumes, initial versus continuation supervision, vocabulary/candidate conventions, optimizer reset policy and RNG ownership in `describe()` and documentation. Reject unsupported data before updating. A batch-only model should remain outside this ABC until it has a genuine continuation algorithm.
3. Add a small synthetic lifecycle test using `tests/test_learners.py` as the example: known candidate ordering, expected update, weights/optimizer/RNG unchanged by scoring, parent/sibling isolation after fork, and reset-once behavior. Backend tests must check actual state, not just equal ranking metrics.
4. Wire the model into an **appropriate** experiment-owned runner/evaluator. If the scientific task differs, make that distinction explicit rather than adapting data silently. Only compare models after agreeing a common protocol. No new metric interface is needed: keep the ranking policy in `experiment/`.
5. Use the notebook to explain assumptions and inspect saved evidence; keep computation in the packages. Update `CONTEXT.md` and the AI-use log when behavior or decisions change.

## Verification and limits

```sh
uv sync --locked
uv run cp4285 doctor
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run cp4285 demo --output runs/new-synthetic-check
```

[`notebooks/01_shared_lifecycle.ipynb`](../notebooks/01_shared_lifecycle.ipynb) is a thin offline walkthrough using the actual pilot and incremental model. It needs an existing Jupyter environment pointing at this project's interpreter; Jupyter is not added as a runtime dependency. Its code cells and this guide's Python blocks are executed in fresh Python processes by `tests/test_documentation.py`, so they do not depend on hidden notebook state.

In-memory forks cost a complete model/optimizer copy. RNG isolation is for **sequential** branches, not concurrent threads manipulating Torch globals. Automated backend checks exercise CPU; CUDA/MPS require their own lifecycle validation before research use. Checkpoints retain the existing weight/optimizer/cursor fields and now save the learner's owned RNG state; interrupted-run resume is still not implemented. No real-data experiments, neural mixed-domain learner, multi-seed aggregation or common classical/neural benchmark are added here. PR #9's separate classical study is not incorporated.
