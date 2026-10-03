# Contributing

Install `.[dev,viz]`, then run `pytest`, `ruff check .`, and `ruff format --check .`.
Keep simulator tests in a separate process from real-GPU checks. For numerical
changes, include an explicit expected result or independent reference.

New example strategies need a documented device-function spec and hand-calculated
trade case. Keep demonstrations small and use synthetic data. Do not include
credentials, personal market datasets, private strategy plugins, or research output.

Document numerical and execution-rule changes in the same pull request. New tracked
files must be listed in `scripts/check_release.py` after reviewing their contents.
