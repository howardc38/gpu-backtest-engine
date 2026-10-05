# Contributing

Install `.[dev,viz]`; run `pytest`, `ruff check .`, and `ruff format --check .`.
See [tests/README.md](tests/README.md) for CPU, simulator, and hardware commands.

Keep boundaries clear:

- `src/gpu_backtest/core/` contains generic computation only. It must not import
  examples, test references, benchmark code, CLI, workflows, or deployment tools.
- `workflows/` uses core primitives for splitting/analysis/output charts.
- Example trading rules/data stay under `examples/` and helper code under `tools/`.
- CLI modules are thin adapters; they do not own numerical or cloud logic.

New example strategies need a device-function spec and hand-calculated case.
Use synthetic data. Keep credentials, private plugins/datasets, and research output
out of Git. New tracked files require a reviewed update to the explicit allowlist
in `scripts/check_release.py`. Preserve historical benchmark JSONs; new claims need
new measured evidence. Document intentional numerical/contract changes in the PR.
