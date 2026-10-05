# RunPod GPU helper

Use a local NVIDIA machine or lease one RunPod GPU. RunPod is an optional execution
helper, separate from the numerical engine. It runs a backtest and downloads raw
results; it does not analyze or rank them.

## Prerequisites

- Python 3.11–3.13 locally, an OpenSSH client, and this package installed.
- Your RunPod account with credits and API access.
- A registered SSH public key; keep its private key on your own machine.
- API key supplied through `RUNPOD_API_KEY`, or the local RunPod CLI configuration
  at `~/.runpod/config.toml`. Never put credentials in the repository/config JSON.

The helper reads credentials locally. API credentials are not sent over SSH or
included in upload archives. Your private strategy stays outside this repository;
running it on a rented GPU necessarily uploads the selected strategy source and data.

## Run one backtest

```bash
gpu-backtest runpod --config examples/gpu_backtest_examples/rsi/config.json \
  --ssh-key ~/.ssh/runpod_ed25519 --output-dir runs/runpod-rsi
```

Default mode is `run`, on one RTX 4090 secure-cloud pod. A hardware smoke check
runs first, followed by one backtest over the complete configured input CSV.
The helper then downloads `output/` and attempts deletion on success, errors,
Ctrl-C or termination. Check `runpod_state.json` for `cleanup: deleted`.

The output directory must be new or empty. Downloaded files are under `results/`:

- `result_results.npz`: four raw grouped float64 arrays.
- `result_manifest.json`: parameters, fees, counts, output schema/hash and checks.
- `gpu_check.json`, `gpu.csv`, `requirements-resolved.txt`: environment evidence.

`remote.log` contains installation/job logs. Local `upload.tar.gz` and lifecycle
state are retained for inspection. If deletion fails, the helper reports its owned
pod ID; delete that pod in your RunPod console. `--keep-pod` explicitly retains a
paid pod and must be used deliberately.

## Settings and dry run

```bash
gpu-backtest runpod --config examples/gpu_backtest_examples/rsi/config.json \
  --output-dir runs/runpod-preview --dry-run
```

Dry run validates and builds the selected upload without reading API credentials
or creating a pod. Supported modes are `run`, `check`, and `benchmark`.
`check` needs no config; `benchmark` uses the packaged public RSI/synthetic workload.

The default hourly-rate cap is $1/hour and the startup/job/download deadline is
1,800 seconds. Use `--max-hourly-rate` and `--max-seconds` to change them. The rate
cap is checked after the provider returns the quote; an unacceptable quote causes
owned-pod deletion. This is not a provider-side total-spend cap.

The default image is
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. The remote recipe installs
the bundled `tools/gpu_backtest_tools/runpod/requirements.txt` environment, disables
CUDA simulation, and uses the image's CUDA toolkit. Image overrides must provide
Python 3.11–3.13, OpenSSH, NVIDIA access, and a compatible development toolkit.

## Your strategy

```bash
gpu-backtest runpod --config my-job.json --plugin-dir ./my-plugins \
  --ssh-key ~/.ssh/runpod_ed25519 --output-dir runs/custom-gpu
```

`--plugin-dir` is a Python import root. For `my_strategies.example`, it should
contain `my_strategies/example.py` and package initialization as needed. Only
selected `.py` files are uploaded; Git, environments and caches are excluded.
Plugin symlinks are rejected. The helper creates an isolated virtual environment
and installs only the public package and bundled requirements. Extra plugin dependencies
are not installed automatically; use the manual route and install them into that
virtual environment, or adapt the remote installation recipe.

The config accepts only `strategy`, `input`, `buy`, `sell`, `entry_dims`,
`exit_dims`, `expected_interval`, and `threads_per_block`. Relative input paths
resolve against the config directory. Removed analysis fields, `pipeline` mode
and `--charts` are rejected before allocation.

## Manual GPU route

Create your own GPU pod, register your SSH key, and connect using the provider's
SSH command. Upload only the public package plus the strategy/data you intend to run.
On the remote machine:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install -e . -r tools/gpu_backtest_tools/runpod/requirements.txt
NUMBA_ENABLE_CUDASIM=0 gpu-backtest gpu-check --output runs/gpu_check.json
NUMBA_ENABLE_CUDASIM=0 gpu-backtest run \
  --config examples/gpu_backtest_examples/rsi/config.json --out-prefix runs/rsi
```

Download the NPZ/manifest files and delete your manually created pod through
RunPod. The automatic helper cannot manage a pod created outside its lifecycle.

## Benchmark

```bash
gpu-backtest runpod --mode benchmark --ssh-key ~/.ssh/runpod_ed25519 \
  --output-dir runs/runpod-benchmark --max-seconds 1800
```

This runs the full billion-pair GPU and eight-thread CPU workloads and may take
several minutes. See [measurement scope and historical evidence](benchmarks.md).
