# Develop with the two packages

**Working examples, not a finalized research protocol.** Start here, copy one small implementation, then test it. The [meeting synthesis](MEETING-2026-10-15.md) explains the research direction; [CONTEXT.md](../CONTEXT.md) records outstanding decisions.

> The experiment package owns the benchmark. Models implement its contract. The notebook explains a run. Tests check the promises.

## 1. Where to write

```text
src/cp4285/
  models/
    counting.py       Complete sample model; copy this shape for your model
  experiment/
    model.py          Recommender ABC: the methods a model must implement
    data.py           DataSource ABC, records, validation and sample data
    evaluation.py     Complete shared evaluator; extend metrics here
    runner.py         Initial selection, independent branches and fixed tests
    artifacts.py      Create-only JSON
    plots.py          Figures from saved results
notebooks/            Imports the packages; no private training or metric code
tests/                Small executable checks
```

One distribution, two responsibility packages. No registration system or new framework. Import your new model directly from its module. Add a public `__init__.py` export only if useful.

## 2. Sample model: implement the ABC

This is the actual [CountingRecommender](../src/cp4285/models/counting.py), not pseudocode. Copy its structure into `models/your_model.py`; replace the learning/scoring logic and metadata. It counts supplied targets, ignores history deliberately and keeps domains independent. It is a plumbing sample, **not our classical research model**.

```python
from collections import Counter

from cp4285.experiment import Example, Query, Recommender


class CountingRecommender(Recommender):
    """Count supervised targets independently per domain; ignore history deliberately.

    B cannot change A scores. This demonstrates a meaningful null/control outcome,
    not a mechanism for studying shared-parameter interference or model capacity.
    """

    def __init__(self) -> None:
        self.counts: Counter[tuple[str, str]] = Counter()

    def train(self, examples: tuple[Example, ...]) -> None:
        self.counts.update((e.target.domain, e.target.item) for e in examples)

    def score(self, query: Query, candidates: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(float(self.counts[query.domain, item]) for item in candidates)

    def begin_continuation(self) -> None:
        """No optimizer to reset; keep the learned sufficient statistics."""

    def describe(self) -> dict[str, object]:
        return {
            "family": "independent-domain target counts (plumbing/isolation control)",
            "config": {},
            "seed": None,
            "randomness": "none",
            "update_policy": "increment counts for supplied targets only; no optimizer or refit",
            "parameter_count": None,
            "trainable_parameter_count": None,
            "capacity_note": "dynamic count table, not a neural parameterization",
        }
```

The inherited `fork()` deep-copies this small model. For another backend, include **all** state: weights, optimizer, counters and owned random generators. Override copying only when necessary; test independence rather than assuming it.

| Method | Your implementation must do |
| --- | --- |
| `train(examples)` | Learn from the supplied targets. No hidden extra epochs, dropped examples, replay or full-data refit. Document batching. |
| `score(query, candidates)` | Return finite scores in exactly that candidate order, without mutation. Higher means better. The target label is not passed. |
| `begin_continuation()` | Keep learned state; apply your declared optimizer transition once. Preserve continuation state afterwards. |
| `describe()` | Record real configuration, randomness, update policy and parameter counts; use explicit non-applicability rather than invented values. |
| `fork()` | Produce an independent complete learner; default `deepcopy` suffices for this sample. |

For a neural implementation, wrap the network and optimizer inside this class. Do not confuse this `train(examples)` with PyTorch's `network.train()`, which only switches mode. Restore training mode for updates, use deterministic evaluation/no gradients for scoring, and allocate vocabulary from permitted training data. We deliberately do not ship a pretend neural stub or add Torch before choosing a real model.

## 3. Sample evaluation: one common policy

This is the actual [shared evaluator](../src/cp4285/experiment/evaluation.py). It is an ordinary function, not another ABC. The runner and direct callers use the same implementation.

```python
import math

from cp4285.experiment import Evaluation


def evaluate(model: Recommender, task: Evaluation, *, k: int) -> dict[str, float]:
    """Single-target macro averages; lexical original-item-ID ties, no seen masking.

    Candidate order is an interface detail, not a tie-breaking rule. A repeated
    item remains eligible. Recall@k equals Hit@k for this single-target contract.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if (
        not task.examples
        or not task.candidates
        or len(set(task.candidates)) != len(task.candidates)
    ):
        raise ValueError("Evaluation needs examples and nonempty unique candidates")
    ndcg = hit = 0.0
    for example in task.examples:
        if example.target.domain != task.domain or example.target.item not in task.candidates:
            raise ValueError("Invalid evaluation target/domain")
        scores = tuple(model.score(example.query, task.candidates))
        if len(scores) != len(task.candidates) or any(not math.isfinite(s) for s in scores):
            raise ValueError("score() must return one finite value per candidate")
        target_score = scores[task.candidates.index(example.target.item)]
        rank = 1 + sum(
            score > target_score or (score == target_score and item < example.target.item)
            for item, score in zip(task.candidates, scores, strict=True)
        )
        if rank <= k:
            hit += 1
            ndcg += 1 / math.log2(rank + 1)
    count = len(task.examples)
    return {"ndcg": ndcg / count, "hit": hit / count, "recall": hit / count}
```

**How to add an evaluation:** change `experiment/evaluation.py`, not the model or notebook. For example, a reporting-only MRR@k extension would accumulate `1 / rank` inside `if rank <= k` and return its example average as `mrr`. Test a rank-2 target: MRR@2 must be 0.5. The runner already stores returned metric fields; to graph a new field, deliberately extend the plotter's supported metrics too. Do not collide with reserved result fields such as `task`, `condition` or `step`.

That is an extension recipe, **not an implemented MRR claim**. Initial selection still uses `ndcg`; changing selection is a separate protocol decision. Candidate filtering, target meaning or single-target versus multi-positive evaluation also require a shared methods decision, not merely a new field in the dictionary.

## 4. A small check for both implementations

Run this after the definitions above. It uses a real nonempty history shape but invented events. For a new model, put the equivalent check in `tests/test_your_model.py`, importing your maintained implementation instead of copying it into the test.

```python
from dataclasses import replace

from cp4285.experiment import Event


def test_sample_contract():
    context = Event("A", "user", "context", "context-item", 0)
    training = tuple(
        Example((context,), Event("A", "user", f"train-{i}", item, i))
        for i, item in enumerate(("a", "a", "b"), start=1)
    )
    probe = Example((context,), Event("A", "user", "test", "b", 4))
    task = Evaluation("toy", "A", (probe,), ("a", "b", "c"))
    model = CountingRecommender()
    model.train(training)
    state = model.counts.copy()

    assert model.score(probe.query, task.candidates) == (2.0, 1.0, 0.0)
    reordered = replace(task, candidates=("c", "a", "b"))
    assert model.score(probe.query, reordered.candidates) == (0.0, 2.0, 1.0)
    assert evaluate(model, task, k=2) == evaluate(model, reordered, k=2)
    assert evaluate(model, task, k=1)["hit"] == 0.0
    assert math.isclose(evaluate(model, task, k=2)["ndcg"], 1 / math.log2(3))
    assert evaluate(model, task, k=2)["recall"] == 1.0
    assert model.counts == state  # Evaluation did not mutate this model.

    branch = model.fork()
    branch.begin_continuation()
    branch.train((Example((context,), Event("A", "user", "fresh", "b", 5)),))
    assert branch.score(probe.query, task.candidates) == (2.0, 2.0, 0.0)
    assert model.counts == state  # Updating a copy did not update the original.


test_sample_contract()
```

These checks are **model-specific**. A neural adapter also needs checks for optimizer/RNG isolation and actual target usage. Equal ranking metrics alone do not prove state purity, and passing the counting model's tests says nothing about an untested adapter. The sample carries history but intentionally does not learn from it.

## 5. Combine the packages in a notebook

Development definitions belong in package files. The notebook imports them:

```python
from pathlib import Path
from uuid import uuid4

from cp4285.experiment import SyntheticSource, plot_results, run
from cp4285.models import CountingRecommender


data = SyntheticSource().load_data()
data.validate()
output = Path("runs") / f"sample-{uuid4().hex[:12]}"
result = run(CountingRecommender(), data, output=output, initial_epochs=3, k=1)
assert result["selected_epoch"] == 1
assert (output / "results.json").is_file()
figure = plot_results(output / "results.json", task="retention-A")
figure  # Jupyter displays the SVG; plotting does not train.
```

Use Markdown for the question, input choices and interpretation. Show prepared counts/schedules, validation trace and raw records before presenting graphs. Restart-and-Run-All must work. Clearly distinguish launching `run(...)` from loading existing result files.

For data work, implement `DataSource.load_data()` returning `PreparedExperiment(provenance=..., initial=..., validation=..., tests=..., conditions=...)`. Copy the small [SyntheticSource](../src/cp4285/experiment/data.py) construction, not a historical splitter. Return tuples; frozen dataclasses do not make incorrectly supplied lists immutable. Enforce source-specific cutoffs, timestamp units/ties, training-only preprocessing and exclusion counts in the loader. The shared validator cannot infer these choices.

## 6. Standing rules and limits

- Preserve original event/user/item identities. Do not splice unrelated histories or treat numeric IDs as cross-domain identity.
- Keep test/validation events out of training targets **and prefixes**, and test targets out of validation selection context. Freeze test queries and candidates.
- Select on validation only. Branch from the same selected state. Keep the declared continuation policy and random state.
- Report actual exposure: B/(A+B) supervised targets, not users or padded positions. The sample helper rounds half-up, concatenates A then B inside a block, and fails on exhausted streams. `targets_seen` records supplied continuation targets; adapter tests must check actual use.
- Keep source/configuration/seed and artifact provenance explicit. No overwrites, hidden downloads or expensive runs without approval. Private sources/data/results stay ignored.
- Do not force a different scientific task through this schema. Document the decision before changing the common contract. Improvement, stability and decline are all valid outcomes.

**Current ceiling:** one target per example; in-memory data/copies; one-query scoring; self-reported metadata; no real-data/model adapter, durable checkpoint/resume or multi-seed analysis. Before expensive experiments, verify backend state reload and capture its environment. These examples are software checks, not Amazon results or proof of forgetting.

## 7. Run and contribute

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
uv run --with jupyterlab jupyter lab notebooks/01_shared_experiment.ipynb
```

Jupyter is optional; uv may install the launcher. The normal tests need no dataset or GPU. `tests/test_documentation.py` executes every Python block above in order and checks that the model/evaluator definitions match their maintained source. Keep examples executable, not aspirational.

**Copyable team brief:** Read this guide, CONTEXT and the October meeting synthesis. Choose one question. Change its owning package, add a small executable check and a short Markdown findings note, then import it in the shared notebook. Propose contract changes rather than hiding incompatible behavior. Before comparing runs, agree the data/filter, target horizon, fixed candidates, metrics, selection, exposure budget, model capacity and seeds. Report evidence and limits; do not promise a forgetting curve.
