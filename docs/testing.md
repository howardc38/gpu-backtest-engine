# Testing and release checks

The default pytest command runs CPU checks and a child process with
`NUMBA_ENABLE_CUDASIM=1`. Simulation is isolated from the parent so that it cannot
silently turn a later GPU run into CPU simulation.

The tests use generated OHLCV data and explicit numeric fixtures. No downloaded
market data, external credentials, or cloud resources are required.

## Proofs used

- RSI hand calculations anchor timing, fee accounting, a losing trade, and
  last-bar mark-to-market behavior independently of implementation agreement.
- A tiny algebraic plugin has a known return matrix. Both GPU reduction passes
  must match exact row/column sums and float32 sums of squares. The plugin is
  created outside the installed engine, exercising the external module path.
- RSI GPU-device results on a synthetic grid are compared with the CPU reference.
- Repeating the same simulated sweep must produce identical arrays and byte-identical
  top CSVs/manifests. This is a regression/determinism check, not an independent
  proof of strategy correctness.
- Group statistics are compared with explicit group-versus-rest calculations.
- Indicators have hand-calculated spot values. Splits must be disjoint and cover
  the requested range; common artifacts preserve decimal precision and deterministic
  tie ordering.
- CLI run/pipeline and optional HTML charts are tested with public sample inputs.
- Plugin validation rejects malformed dimensions, reordered overrides, invalid
  table references, oversized specs, and reserved output names before device work.

## Real GPU

```bash
NUMBA_ENABLE_CUDASIM=0 python -m pytest -m gpu -v
```

GPU tests fail if simulation is enabled, and skip if no NVIDIA GPU is available.
Require actual passes, rather than skips, before claiming hardware validation.
The packaged `gpu-backtest gpu-check` command performs numeric hardware checks
without pytest. It and the public RSI pipeline passed on an RTX 4090; see the
[validation record](runpod-validation.md). This establishes small-grid GPU
compilation/results, not large-grid performance or arbitrary plugin correctness.

RunPod lifecycle tests simulate create/SSH/job/download/cleanup failures, timeouts,
normal cancellation, cancellation during creation, price rejection, ambiguous
allocation responses, metadata-write failure, upload selection, and unsafe tar
entries. They do not rent GPUs during default tests.

The [public performance benchmark](benchmarks.md) separately measures warmed
compiled CPU/GPU reductions and a normal billion-pair engine run. Its CPU baseline
is checked against the Python reference, including float32 sums of squares;
default tests verify the billion grid size without executing that grid.
`--billion-cpu` additionally runs the full billion-pair CPU job through output and
checks all four aggregate arrays against the GPU run; it never substitutes an estimate.

## Distribution contents

`scripts/check_release.py` checks an explicit tracked-file allowlist and common
credential patterns. CI runs it and builds a source distribution and wheel.
The public tree contains the engine, one educational RSI plugin, generated data,
documentation, and generic tests. New files require an explicit allowlist update.

Before publishing a release, inspect both distribution archives, execute the
documented example from a clean installation, and review the tracked diff. The
allowlist complements manual content review; it is not a proof against all secrets.
