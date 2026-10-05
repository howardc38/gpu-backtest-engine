# GPU Backtest Engine

## 10× faster on our public billion-pair benchmark

**CPU: 7 min 44.60 s → RTX 4090: 44.95 s.** The same public RSI grid:
**1,000,000,000 pairs × 1,024 bars**, compared with an **eight-thread compiled
Numba CPU baseline**. Measured speedup **10.34×** on v0.4, saving about seven
minutes per sweep. Cloud setup is additional; [method and raw evidence](docs/benchmarks.md).
v0.6 preserves the GPU reductions and saves raw results instead of ranked analysis.
The historical timing includes v0.4's output processing; it is not a new v0.6 timing.

- **Your algorithm:** load a strategy module/object; private rules can remain private.
- **Large grids:** deterministic GPU reductions without materializing the full return matrix.
- **Your machine or RunPod:** use a local NVIDIA GPU or the separate cloud helper.

## Folder guide

| Folder | What it contains |
|---|---|
| [src/gpu_backtest/core/](src/gpu_backtest/core/) | Runs your strategy on the GPU and saves raw results |
| [src/gpu_backtest/cli/](src/gpu_backtest/cli/) | Reads terminal commands and calls the engine or a helper |
| [examples/](examples/) | A small RSI strategy, runnable config and generated price data |
| [tests/](tests/) | Checks correctness using known answers, CPU simulation and actual GPUs |
| [tools/gpu_backtest_tools/runpod/](tools/gpu_backtest_tools/runpod/) | Rents a GPU, runs your job, downloads results and deletes its pod |
| [tools/gpu_backtest_tools/benchmarks/](tools/gpu_backtest_tools/benchmarks/) | Measures the same workload on CPU and GPU |
| [tools/gpu_backtest_tools/checks/](tools/gpu_backtest_tools/checks/) | Checks GPU availability and a few known numerical answers |
| [benchmarks/results/](benchmarks/results/) | Saved reports supporting the speed and correctness claims |
| [docs/](docs/) | Instructions for writing strategies, using RunPod and measuring speed |
| [scripts/](scripts/) | Checks which files can be included in a public release |
| [.github/workflows/](.github/workflows/) | Runs automated checks on GitHub |

[Every folder and file explained](docs/repository.md).

Start with [core/engine.py](src/gpu_backtest/core/engine.py) to follow a backtest.
[kernels.py](src/gpu_backtest/core/kernels.py) contains the CUDA calculation;
trading rules come from your strategy. Market data and indicator tables are
prepared on the CPU before GPU execution.

## Try the RSI example without a GPU

Python 3.11–3.13, from a checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
NUMBA_ENABLE_CUDASIM=1 gpu-backtest run \
  --config examples/gpu_backtest_examples/rsi/config.json --out-prefix runs/rsi
```

The CUDA simulator is for tiny examples/tests. The [RSI example](examples/gpu_backtest_examples/rsi/README.md)
uses generated data and is packaged separately as `gpu_backtest_examples.rsi.strategy`.

## Use your own strategy

Install your strategy package into the same environment:

```python
from gpu_backtest import run
from my_strategies import example

results = run(example, "market.csv", "runs/custom", buy=0.0015, sell=0.0015)
```

Or use `gpu-backtest run --strategy my_strategies.example --input market.csv
--out-prefix runs/custom`. Plugins implement their complete Numba CUDA trading loop;
see [the contract](docs/strategy.md). Entry/exit parameters form separate Cartesian
axes, up to four dimensions each, with up to four precomputed indicator tables.
Shared-parameter diagonal-only sweeps are not supported.

## Raw output

A run returns four float64 arrays and, when an output prefix is supplied, writes:

- `<prefix>_results.npz`: `entry_sum`, `entry_sumsq`, `exit_sum`, `exit_sumsq`.
- `<prefix>_manifest.json`: ordered parameter dimensions, counts, fees, array schema,
  result-file SHA-256, and reduction consistency checks.

Entry arrays aggregate each entry parameter set across **all exits**; exit arrays
aggregate each exit set across **all entries**. Array indices follow Cartesian
parameter order, with the last dimension varying fastest. Individual returns and
their squares round to float32 before float64 accumulation. No full pair matrix,
trade ledger or equity curve is stored.

```python
import numpy as np

with np.load("runs/rsi_results.npz", allow_pickle=False) as results:
    entry_sum = results["entry_sum"]
    exit_sum = results["exit_sum"]
```

Apply your own analysis downstream. v0.6 removes built-in scoring, ranking,
common-parameter analysis, date-split pipelines and charts. The `top_n` option and
old ranked CSV/schema-1 artifacts are removed; numerical return arrays remain the
same. Old config keys are rejected rather than silently ignored.

[RTX 4090 validation](benchmarks/results/rtx4090_raw_output_parity_20261005.json):
v0.5/v0.6 returned arrays matched byte-for-byte on small, 16,777,216-pair and
1,000,000,000-pair grids. Clean-wheel results matched the default RunPod bundle,
all five physical-GPU tests passed, and owned-pod deletion was confirmed.

## RunPod and validation tools

| Command | Purpose |
|---|---|
| `run` | One GPU backtest with raw results |
| `runpod` | Lease one GPU, upload selected files, run, download, delete |
| `benchmark` | CPU/GPU performance and numeric comparison |
| `gpu-check` | Small numerical checks on actual hardware |

```bash
gpu-backtest runpod --config examples/gpu_backtest_examples/rsi/config.json \
  --ssh-key ~/.ssh/runpod_ed25519 --output-dir runs/runpod-rsi
```

RunPod requires your account/API key and registered SSH key. It defaults to one
full-input run; [setup, manual GPU route, and cleanup](docs/runpod.md).
The public `from gpu_backtest import run` API and old `rsi_meanrev` shorthand
remain usable. Direct internal imports moved under `core/` or
`gpu_backtest_tools` in v0.5.

## Development

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
python scripts/check_release.py
```

[CPU/GPU testing](tests/README.md) · [Plugin contract](docs/strategy.md) ·
[RunPod helper](docs/runpod.md) · [Benchmark method](docs/benchmarks.md)

[MIT](LICENSE). Private strategy plugins retain their own licenses.
