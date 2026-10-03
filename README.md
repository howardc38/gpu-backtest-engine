# GPU Backtest Engine

A Python engine for GPU parameter sweeps, with interchangeable strategy plugins,
deterministic reductions, and separate entry/exit effect-size rankings.

The distribution includes one educational RSI strategy and generated synthetic
OHLCV data. Strategies can live in a separately installed package, including a
private repository; no changes to the engine package are needed.

## Quickstart without a GPU

Python 3.11–3.13 is supported. From a checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz]'
NUMBA_ENABLE_CUDASIM=1 gpu-backtest run \
  --config examples/rsi.json --out-prefix runs/rsi
gpu-backtest charts --input runs/rsi_top_entry.csv runs/rsi_top_exit.csv \
  --output runs/rsi.html
```

Open `runs/rsi.html` in a browser. Its Vega libraries load from a public CDN.
The CUDA simulator is for tiny demonstrations and correctness tests. Large grids
need a real NVIDIA GPU.

The example sweeps four entry combinations against four exit combinations.
It writes two ranked CSVs and an output manifest:

```text
runs/rsi_top_entry.csv
runs/rsi_top_exit.csv
runs/rsi_top_manifest.json
```

`examples/synthetic.csv` contains generated bars, not historical market data.
Regenerate it with `python examples/generate_data.py`.

## RunPod: use a GPU from your laptop

The same engine runs on a rented RunPod GPU. With a funded account, Pod API key,
and registered SSH key:

```bash
gpu-backtest runpod --config examples/rsi.json \
  --ssh-key ~/.ssh/runpod_ed25519 --output-dir runs/runpod-rsi --charts
```

This leases one RTX 4090, uploads selected engine/data/plugin files, installs the
environment, runs hardware numeric checks and the pipeline, downloads output,
and deletes the pod. Add `--dry-run` to inspect the bundle without renting anything;
`--mode run` selects one sweep and `--mode check` runs only hardware checks.
Private modules can be supplied with `--plugin-dir` without adding them to this repo.
See [the RunPod guide](docs/runpod.md) for setup, the manual route, environment
requirements, time/rate limits, custom plugins, and cleanup behavior.

## NVIDIA GPU setup (inside the GPU machine)

Run these commands in the GPU computer's terminal. For RunPod, use the pinned
setup in [the RunPod guide](docs/runpod.md); your laptop only controls the launcher.
For another GPU workstation/server, install a compatible driver and CUDA runtime/toolkit. The separate NVIDIA
`numba-cuda` backend uses the same `from numba import cuda` interface:

```bash
python -m pip install -e '.[cuda]'
# If you also need CUDA 12 Python runtime/toolkit dependencies:
python -m pip install 'numba-cuda[cu12]>=0.30,<0.31'
gpu-backtest run --config examples/rsi.json --out-prefix runs/rsi_gpu
```

See the [NVIDIA installation guide](https://nvidia.github.io/numba-cuda/user/installation.html)
for CUDA 12/13 and driver requirements. Run GPU commands without
`NUMBA_ENABLE_CUDASIM=1`. The project's tested Numba series is 0.65; upgrading the
backend requires rerunning the simulator and real-GPU checks.

## Use your own strategy

Install the package containing your strategy in the same Python environment:

```bash
python -m pip install -e /path/to/your-strategy-package
gpu-backtest run --strategy my_strategies.example \
  --input /path/to/market.csv --out-prefix runs/custom
```

Or pass a module/object through the Python API:

```python
from gpu_backtest import run
from my_strategies import example

results = run(example, "market.csv", "runs/custom", buy=0.0015, sell=0.0015)
```

The engine loads an installed Python module. Plugins are trusted code, and run
locally; they must implement the documented Numba device-function contract.
See [the strategy contract](docs/strategy-contract.md) for parameter specifications,
indicator tables, CPU references, and the RSI execution rules.

## Date splits and common parameters

```bash
NUMBA_ENABLE_CUDASIM=1 gpu-backtest pipeline \
  --config examples/rsi.json --output-dir runs/pipeline
```

With three splits, the pipeline evaluates the full period and its two disjoint
halves. It records split hashes, writes ranked artifacts for every split, and
finds parameter combinations appearing in every top artifact. The output includes
`common_entry.csv` and `common_exit.csv` with per-period, average, and minimum
effect sizes. No common combinations is reported as a failed analysis, not an
empty successful result.

Config `input` paths resolve relative to the JSON file. CLI `--input` and output
paths resolve relative to the current directory. A strategy must always be
specified explicitly. Config keys are `strategy`, `input`, `buy`, `sell`,
`entry_dims`, `exit_dims`, `top_n`, `threads_per_block`, `expected_interval`,
`num_splits`, `start_date`, `end_date`, and `ranges`. Dates use `DD-MM-YYYY`;
custom ranges are arrays such as `[["01-01-2024", "31-01-2024"], ...]`.

`gpu-backtest split`, `common`, and `charts` also work independently; run each
subcommand with `--help` for its arguments.

## Scope and interpretation

- Entry and exit parameters form a Cartesian product, with at most four
  dimensions on each side and four precomputed indicator tables. Shared-parameter
  diagonal-only sweeps are not supported.
- The engine runs two passes over the same return matrix. Returns and squared
  returns are rounded to float32, then accumulated in float64 in a fixed order.
- Ranked rows represent entry/exit parameter groups, not individually selected
  complete strategies. Their observations are parameter combinations, not
  independent market samples. Effect sizes and nominal normal intervals are
  descriptive grid comparisons. `posterior_prob_superior` is a normal-CDF proxy;
  it is not a Bayesian probability or a forecast of profitable trading.
- Strategy plugins own their trading loop, position sizing, and execution rules.
  The included example uses long-only next-open execution, with its boundary and
  fee conventions documented in the strategy contract.
- The current API returns grouped sums and sums of squares, plus optional top
  artifacts. It does not return a trade ledger, equity curve, Sharpe ratio, or
  drawdown series. Example performance is not a trading recommendation.

## Development

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
python scripts/check_release.py
```

Default tests need no GPU. They run simulation in a separate process and verify
hand-calculated results, known reduction matrices, independent CPU references,
output determinism, statistics, indicator values, split coverage, and external
plugin loading. For a real NVIDIA GPU:

```bash
NUMBA_ENABLE_CUDASIM=0 python -m pytest -m gpu -v
```

See [testing details](docs/testing.md) and the [RunPod validation record](docs/runpod-validation.md).
Small numeric checks and the public RSI pipeline have passed on a real RTX 4090.
Large-grid performance, other hardware/backend versions, and custom strategies
still require their own validation.

## License

[MIT](LICENSE). You may use the engine with separately maintained private
strategy plugins, subject to their own licenses.
