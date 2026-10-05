# Contributing

Install `.[dev]`; run `pytest`, `ruff check .`, and `ruff format --check .`.
See [tests/README.md](tests/README.md) for CPU, simulator, and hardware commands.

Keep boundaries clear:

- `src/gpu_backtest/core/` contains generic computation only. It must not import
  examples, test references, benchmark code, CLI, or deployment tools.
- Example trading rules/data stay under `examples/` and helper code under `tools/`.
- CLI modules are thin adapters; they do not own numerical or cloud logic.

Keep output raw: no built-in scoring, ranking, statistical inference, splitting or charts.
Preserve four grouped array values and their parameter alignment.

New example strategies need a device-function spec and hand-calculated case.
Use synthetic data. Keep credentials, private plugins/datasets, and research output
out of Git. New tracked files require a reviewed update to the explicit allowlist
in `scripts/check_release.py`. Preserve historical benchmark JSONs; new claims need
new measured evidence. Document intentional numerical/contract changes in the PR.
