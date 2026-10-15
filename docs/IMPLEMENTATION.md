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

`run(...)` requires a **new** output directory. It writes `prepared.json` (supplied inputs), `manifest.json` (provenance/settings/source fingerprints) and `results.json` (validation trace and fixed-test records). `plot_results(...)` reads saved results; it never starts training. There is no durable model checkpoint or interrupted-run resume yet.

JSON publication refuses existing files/symlinks and racing writers using a same-filesystem hardlink. Hardlink support is required; atomic visibility is not power-loss durability or a multi-file transaction. A failed run may leave partial evidence: inspect it and use a new path, rather than deleting or overwriting it automatically.

The old trial CLI is retired. Its [historical runbook](https://github.com/Hong-yiii/cp4285_reccomender_systems/blob/8b9edd3/docs/IMPLEMENTATION.md) is not a current API. Before real data or expensive computation, agree the methods and verify the chosen backend's state/reload behavior; see [the guide's rules and limits](PACKAGES-AND-BUILDING.md#6-standing-rules-and-limits).
