# Folder and file guide

Use this page to find what each directory and file does. Start with
[the engine](../src/gpu_backtest/core/engine.py) for GPU execution,
[the RSI example](../examples/gpu_backtest_examples/rsi/README.md) to try a strategy,
or [the RunPod guide](runpod.md) to run on a rented GPU.

GitHub's file list shows **Last commit message** next to each folder and file.
That column describes the latest change touching the path. A commit touching many
paths gives them the same message; the purpose of each path is documented below.

## Repository settings

Files used to install, license and maintain the project.

| File | Purpose |
|---|---|
| [README.md](../README.md) | Project overview, installation and backtest commands. |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to change the code and check your changes. |
| [LICENSE](../LICENSE) | MIT license for the public code. |
| [pyproject.toml](../pyproject.toml) | Package version, dependencies, installation and test settings. |
| [MANIFEST.in](../MANIFEST.in) | Which files go into the downloadable source package. |
| [.gitignore](../.gitignore) | Files Git should ignore, including environments, results and credentials. |

## src/gpu_backtest/ — the backtest package

The Python API and command-line entry points.

| File | Purpose |
|---|---|
| [__init__.py](../src/gpu_backtest/__init__.py) | Provides the public run() function and package version. |
| [__main__.py](../src/gpu_backtest/__main__.py) | Starts the command line when you run python -m gpu_backtest. |

## src/gpu_backtest/core/ — GPU backtesting

Loads a strategy, runs its parameter grid on the GPU and saves raw results.

| File | Purpose |
|---|---|
| [__init__.py](../src/gpu_backtest/core/__init__.py) | Marks this directory as a Python package. |
| [engine.py](../src/gpu_backtest/core/engine.py) | Coordinates one backtest from input CSV to GPU execution and output. |
| [kernels.py](../src/gpu_backtest/core/kernels.py) | CUDA code that sums strategy returns and squared returns for entry and exit parameter sets. |
| [grid.py](../src/gpu_backtest/core/grid.py) | Counts parameter combinations and converts an array index back to parameter values. |
| [data.py](../src/gpu_backtest/core/data.py) | Checks the input timestamps, OHLC prices and volume. |
| [indicators.py](../src/gpu_backtest/core/indicators.py) | Calculates indicator tables such as RSI, EMA and ATR before GPU execution. |
| [strategy.py](../src/gpu_backtest/core/strategy.py) | Loads your strategy and checks its parameter definitions and GPU function. |
| [output.py](../src/gpu_backtest/core/output.py) | Checks the GPU totals and writes raw NPZ arrays with a JSON manifest. |

## src/gpu_backtest/cli/ — terminal commands

Reads command-line arguments and calls the engine or the selected helper.

| File | Purpose |
|---|---|
| [__init__.py](../src/gpu_backtest/cli/__init__.py) | Selects the requested command and reports errors. |
| [backtest.py](../src/gpu_backtest/cli/backtest.py) | Handles the run command. |
| [options.py](../src/gpu_backtest/cli/options.py) | Reads JSON config and command-line settings; rejects unsupported options. |
| [tools.py](../src/gpu_backtest/cli/tools.py) | Handles runpod, benchmark and gpu-check commands. |

## examples/gpu_backtest_examples/ — runnable example

An educational RSI strategy and generated price data.

| File | Purpose |
|---|---|
| [__init__.py](../examples/gpu_backtest_examples/__init__.py) | Marks the examples as a separate Python package. |
| [data.py](../examples/gpu_backtest_examples/data.py) | Generates repeatable OHLCV data for the example and performance tests. |
| [__init__.py](../examples/gpu_backtest_examples/rsi/__init__.py) | Marks the RSI example as a Python package. |
| [README.md](../examples/gpu_backtest_examples/rsi/README.md) | Explains the example trading rules and how to run it. |
| [strategy.py](../examples/gpu_backtest_examples/rsi/strategy.py) | Implements RSI trading rules on the GPU and a CPU version for checking answers. |
| [config.json](../examples/gpu_backtest_examples/rsi/config.json) | Small parameter grid, input path and trading fees for the example. |
| [generate_data.py](../examples/gpu_backtest_examples/rsi/generate_data.py) | Command to regenerate the example price CSV. |
| [synthetic.csv](../examples/gpu_backtest_examples/rsi/synthetic.csv) | 128 generated price bars used as example input. |

## tools/gpu_backtest_tools/ — execution and validation helpers

RunPod execution, hardware checks and performance measurement.

| File | Purpose |
|---|---|
| [__init__.py](../tools/gpu_backtest_tools/__init__.py) | Marks the helpers as a separate Python package. |

## tools/gpu_backtest_tools/runpod/ — run on a rented GPU

Creates a pod, uploads selected code/data, runs the job, downloads results and deletes its pod.

| File | Purpose |
|---|---|
| [__init__.py](../tools/gpu_backtest_tools/runpod/__init__.py) | Provides the RunPod client and launcher functions. |
| [client.py](../tools/gpu_backtest_tools/runpod/client.py) | Reads local credentials and calls the RunPod API. |
| [transport.py](../tools/gpu_backtest_tools/runpod/transport.py) | Connects over SSH, transfers files and safely extracts downloaded results. |
| [bundle.py](../tools/gpu_backtest_tools/runpod/bundle.py) | Selects upload files and creates the remote installation/job script. |
| [launcher.py](../tools/gpu_backtest_tools/runpod/launcher.py) | Manages pod creation, time/rate limits, job stages and cleanup. |
| [requirements.txt](../tools/gpu_backtest_tools/runpod/requirements.txt) | Pinned Python dependencies installed on the remote GPU machine. |

## tools/gpu_backtest_tools/benchmarks/ — CPU/GPU speed measurement

Runs the same public RSI workload on CPU and GPU and records timings.

| File | Purpose |
|---|---|
| [__init__.py](../tools/gpu_backtest_tools/benchmarks/__init__.py) | Marks the performance tools as a Python package. |
| [cpu.py](../tools/gpu_backtest_tools/benchmarks/cpu.py) | Compiled multithreaded CPU implementation used for the speed comparison. |
| [runner.py](../tools/gpu_backtest_tools/benchmarks/runner.py) | Defines test grids, runs CPU/GPU comparisons and writes the measurement report. |

## tools/gpu_backtest_tools/checks/ — check numerical answers

Small checks that compare computed results with known answers.

| File | Purpose |
|---|---|
| [__init__.py](../tools/gpu_backtest_tools/checks/__init__.py) | Marks the numeric checks as a Python package. |
| [gpu.py](../tools/gpu_backtest_tools/checks/gpu.py) | Checks real GPU availability and runs known matrix and hand-calculated RSI cases. |
| [reference.py](../tools/gpu_backtest_tools/checks/reference.py) | Small CPU calculation used to check GPU totals in tests. |

## tests/ — automated checks

Tests correctness, file output, command-line use and RunPod failure handling.

| File | Purpose |
|---|---|
| [README.md](../tests/README.md) | Instructions for CPU, CUDA simulation and actual GPU tests. |

## tests/cpu/ — tests that need no NVIDIA GPU

Checks inputs, indicators, output files and helpers on an ordinary CPU.

| File | Purpose |
|---|---|
| [test_architecture.py](../tests/cpu/test_architecture.py) | Checks that core does not import example/helper code and that RunPod lists its upload files correctly. |
| [test_benchmark.py](../tests/cpu/test_benchmark.py) | Checks the CPU comparison against known answers, grid sizes and saved raw output. |
| [test_cli.py](../tests/cpu/test_cli.py) | Checks run/config behavior and rejection of removed analysis commands. |
| [test_engine_contract.py](../tests/cpu/test_engine_contract.py) | Checks market data, GPU-total validation, exact raw file output and write failures. |
| [test_examples.py](../tests/cpu/test_examples.py) | Checks the RSI CPU implementation against hand-calculated trades. |
| [test_gpu_check.py](../tests/cpu/test_gpu_check.py) | Checks that hardware validation rejects unavailable GPUs and CUDA simulation. |
| [test_runpod.py](../tests/cpu/test_runpod.py) | Tests upload, failures, cancellation and pod cleanup with a fake provider; rents no pods. |
| [test_strategy.py](../tests/cpu/test_strategy.py) | Checks strategy loading, parameter ranges and indicator definitions. |
| [test_tables.py](../tests/cpu/test_tables.py) | Checks calculated indicator tables against known values. |

## tests/gpu/ — tests of the CUDA calculation

Runs CUDA code in a CPU simulator or on an actual NVIDIA GPU.

| File | Purpose |
|---|---|
| [fixtures.py](../tests/gpu/fixtures.py) | Shared generated data, test strategies and expected answers. |
| [simulator_cases.py](../tests/gpu/simulator_cases.py) | Numerical test cases run inside the isolated CUDA simulator process. |
| [test_simulator.py](../tests/gpu/test_simulator.py) | Starts those simulator cases in a separate process. |
| [test_hardware.py](../tests/gpu/test_hardware.py) | Runs known-answer and raw-output checks on an actual NVIDIA GPU. |

## docs/ — usage guides

Instructions for strategies, RunPod and benchmark measurements.

| File | Purpose |
|---|---|
| [repository.md](../docs/repository.md) | This folder and file guide. |
| [strategy.md](../docs/strategy.md) | How to implement a strategy the engine can run. |
| [runpod.md](../docs/runpod.md) | How to configure RunPod, run a job, download results and clean up. |
| [benchmarks.md](../docs/benchmarks.md) | How performance was measured and what the published timings cover. |

## benchmarks/results/ — recorded evidence

Reports from completed hardware runs, kept separately from executable benchmark code.

| File | Purpose |
|---|---|
| [rtx4090_rsi_20261003.json](../benchmarks/results/rtx4090_rsi_20261003.json) | First CPU/GPU timing comparison and GPU-only billion-pair measurement. |
| [rtx4090_rsi_matched_billion_20261003.json](../benchmarks/results/rtx4090_rsi_matched_billion_20261003.json) | Full CPU/GPU billion-pair comparison behind the historical 10.34x claim. |
| [rtx4090_raw_output_parity_20261005.json](../benchmarks/results/rtx4090_raw_output_parity_20261005.json) | Evidence that v0.5/v0.6 raw results match exactly, including the billion-pair grid. |
| [runpod_validation_20261003.md](../benchmarks/results/runpod_validation_20261003.md) | Report of the initial small real-GPU checks and completed pod cleanup. |

## scripts/ — release checks

Checks the public repository before publishing.

| File | Purpose |
|---|---|
| [check_release.py](../scripts/check_release.py) | Checks the approved file list and scans for common credential patterns. |

## .github/workflows/ — GitHub automation

Checks pushed changes and pull requests.

| File | Purpose |
|---|---|
| [ci.yml](../.github/workflows/ci.yml) | Runs tests, lint, formatting, release checks and package builds across Python/backend combinations. |

