# GPU Backtest Engine

## 10× faster on our public billion-pair benchmark

**CPU: 7 min 44.60 s → RTX 4090: 44.95 s.** Same RSI grid:
**1,000,000,000 pairs × 1,024 bars**, compared with an **eight-thread compiled
Numba CPU baseline**. Measured speedup **10.34×**, saving about seven minutes per
sweep. Cloud setup is additional; [method and raw evidence](docs/benchmarks.md).

- **Your algorithm:** load a strategy module/object; private rules can remain private.
- **Large grids:** deterministic GPU reductions without materializing the full return matrix.
- **Your machine or RunPod:** use a local NVIDIA GPU or the separate cloud helper.

## Where everything lives

```text
src/gpu_backtest/
  core/                         GPU engine, kernels, grids, indicators, statistics, output
  workflows/                    Generic split/common analysis and charts
  cli/                          Thin command-line adapters
examples/gpu_backtest_examples/
  rsi/                          Example strategy + config + generated CSV
  data.py                       Example/benchmark synthetic data generator
tools/gpu_backtest_tools/
  runpod/                       Optional API / SSH / bundle / lifecycle helper
  benchmarks/                   Performance runner and compiled CPU baseline
  checks/                       Hardware smoke checks and CPU test reference
tests/
  cpu/                          CPU contracts, examples, workflows, helper tests
  gpu/                          Isolated CUDA simulation and real-GPU tests
docs/                           Strategy contract, RunPod usage, benchmark method
benchmarks/results/             Historical measurements and validation evidence
```

**Start with `core/engine.py`** for the GPU run. Numerical kernels are in
`core/kernels.py`; trading rules are supplied by a plugin. Core imports no example,
CPU comparison engine, benchmark, or RunPod code. CPU preprocessing of market data
and indicator tables is part of the GPU pipeline; the separate CPU backtest baseline
is only a benchmark/reference tool.

## Try the RSI example without a GPU

Python 3.11–3.13, from a checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz]'
NUMBA_ENABLE_CUDASIM=1 gpu-backtest run \
  --config examples/gpu_backtest_examples/rsi/config.json --out-prefix runs/rsi
gpu-backtest charts --input runs/rsi_top_entry.csv runs/rsi_top_exit.csv \
  --output runs/rsi.html
```

The simulator is for tiny examples/tests. Open `runs/rsi.html` in a browser;
its Vega libraries load from a public CDN. The [RSI example](examples/gpu_backtest_examples/rsi/README.md)
is educational and uses generated data. It is packaged separately as
`gpu_backtest_examples.rsi.strategy`, not as an engine builtin.

## Use your own strategy

Install your strategy package into the same environment:

```python
from gpu_backtest import run
from my_strategies import example

results = run(example, "market.csv", "runs/custom", buy=0.0015, sell=0.0015)
```

Or use `gpu-backtest run --strategy my_strategies.example --input market.csv
--out-prefix runs/custom`. Plugins implement a complete Numba CUDA trading loop;
see [the contract](docs/strategy.md). Entry/exit parameters are separate Cartesian
axes (up to four dimensions each), with up to four precomputed tables. Shared-parameter
diagonal-only sweeps are not supported. The API returns grouped sums/squares and
optional ranked artifacts, not a trade ledger, equity curve, Sharpe, or drawdown series.

## Optional workflows and helpers

| Command | Owner / purpose |
|---|---|
| `pipeline` | `workflows/`: split, run each segment, intersect ranked parameters |
| `split`, `common`, `charts` | `workflows/`: standalone data/result analysis |
| `runpod` | `tools/runpod/`: lease one GPU, upload selected files, download, delete |
| `benchmark` | `tools/benchmarks/`: reproduce the public CPU/GPU measurements |
| `gpu-check` | `tools/checks/`: small numeric checks on actual hardware |

```bash
gpu-backtest runpod --config examples/gpu_backtest_examples/rsi/config.json \
  --ssh-key ~/.ssh/runpod_ed25519 --output-dir runs/runpod-rsi --charts
```

RunPod requires your account/API key and registered SSH key. It is an optional
execution helper; [setup, manual GPU route, and cleanup](docs/runpod.md).
Commands keep their existing names. The public `from gpu_backtest import run` API
and old `rsi_meanrev` shorthand remain usable; direct internal imports moved under
`core/`, `workflows/`, or `gpu_backtest_tools` in v0.5.

## Performance and development

| Same billion-pair RSI job | CPU, eight threads | RTX 4090 | Saved per sweep |
|---|---|---|---|
| 1,000,000,000 pairs × 1,024 bars, through output | 7 min 44.60 s | 44.95 s | 6 min 59.65 s; 10.34× faster |

This is one measured run per side on the published setup, not a universal speed
claim. Small CPU jobs may not justify cloud startup. Statistics describe parameter
combination groups, not independent market samples or a forecast of profits.
See [benchmark details](docs/benchmarks.md) for scope and raw data.

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
