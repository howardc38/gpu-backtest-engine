# Run on a rented RunPod GPU

RunPod supplies the GPU computer; the same engine runs on that computer as on a
local NVIDIA workstation. The launcher runs on your laptop, leases one GPU,
uploads selected files, runs the job, downloads results, and deletes the pod.
You do not need a local NVIDIA GPU.

## Account and SSH setup

You need a funded RunPod account, an API key allowed to create/read/delete Pods,
and an SSH key whose public half is registered under RunPod Credentials.
See [RunPod SSH setup](https://docs.runpod.io/pods/configuration/use-ssh).

If you do not already have a suitable key, create one:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519
cat ~/.ssh/runpod_ed25519.pub
```

Register the public key with RunPod; keep the private key on your laptop. The
launcher uses OpenSSH batch mode, so unlock encrypted keys in your SSH agent first.
New host keys use `accept-new`, with a separate known-hosts file for each run.

Provide the API key through `RUNPOD_API_KEY`, or an `apikey` field in your local
`~/.runpod/config.toml`. The key is excluded from the upload, SSH process environment,
remote job, and state file. Server error bodies/auth headers are not printed.

## Automated example

From an installed checkout:

```bash
gpu-backtest runpod --config examples/gpu_backtest_examples/rsi/config.json \
  --ssh-key ~/.ssh/runpod_ed25519 --output-dir runs/runpod-rsi --charts
```

The default `pipeline` mode evaluates the full period and its two halves and
produces common-parameter results. Add `--mode run` for one full-period sweep.
To run only the built-in numeric hardware checks:

```bash
gpu-backtest runpod --mode check --ssh-key ~/.ssh/runpod_ed25519 \
  --output-dir runs/runpod-check
```

The output directory must be new or empty. A dry run reads no API credentials,
requires no SSH key, and rents no GPU:

```bash
gpu-backtest runpod --config examples/gpu_backtest_examples/rsi/config.json \
  --output-dir runs/runpod-preview --dry-run
```

It writes `upload.tar.gz` and lists the exact files it contains. Inspect this
bundle before using custom plugins. Its contents go to your rented machine;
they are not published to GitHub.

To reproduce the [published CPU/GPU and billion-pair benchmark](benchmarks.md):

```bash
gpu-backtest runpod --mode benchmark --ssh-key ~/.ssh/runpod_ed25519 \
  --output-dir runs/runpod-benchmark --max-seconds 1800
```

This mode needs no config/market file: it generates public synthetic data, requests
at least eight vCPUs, measures an eight-thread compiled CPU baseline and the GPU,
then runs the entire billion-pair RSI grid on **both** CPU and GPU. Allow several
minutes for the CPU measurement. It uses one paid GPU pod and the normal cleanup
behavior. Timing can vary on another host.

## Execution and output

The launcher validates the selected CSV, normalizes config paths, and packages
an explicit engine source-file list plus that CSV and any selected Python plugin
directory. It does not upload an entire repository. It creates one pod, records
its ID, waits for SSH, installs a fresh environment, and forces real CUDA execution.

Each job first checks an explicit reduction matrix, repeat determinism, and
hand-calculated RSI returns on hardware. It then runs the configured job, optionally
creates charts, downloads results, and attempts pod deletion on completion, error,
timeout, or normal cancellation. Only the pod created for this launch is deleted.

`remote.log` records installation/job progress. `runpod_state.json` records the
pod ID, status, and cleanup outcome. Downloaded output is under `results/`, including
`gpu_check.json`, `gpu.csv`, exact resolved Python requirements, and the selected
job's artifacts. Downloaded tar links and path traversal are rejected.

## Environment and limits

Default image:

```text
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

This existing official image supplies Python 3.11, SSH, and CUDA development
libraries. The engine does not use its PyTorch installation. The launcher creates
an isolated venv and installs the engine with the pinned
[validated environment](../tools/gpu_backtest_tools/runpod/requirements.txt), using the
image's CUDA toolkit. Optional charts use Altair 6.3.0. Overrides must supply Python 3.11–3.13,
compatible CUDA development libraries/driver, SSH, tar, and `nvidia-smi`.

The default is one RTX 4090 on Secure Cloud. Override explicitly with `--gpu`,
`--cloud`, or `--image`; there is no silent GPU/cloud fallback. The
[RunPod Pods API](https://docs.runpod.io/api-reference/pods/POST/pods) handles
availability and allocation.

`--max-hourly-rate` defaults to 1.00 USD/hour. The reported rate is checked **after
allocation**; an over-limit or unknown-rate pod is immediately deleted, even with
`--keep-pod`. A brief charge may be incurred before rejection. `--max-seconds`
defaults to 1800 and limits local startup/SSH/job waits after allocation; API
cleanup requests have their own bounded timeouts.

`--keep-pod` deliberately retains a debug machine and its ongoing charges. Normal
interruption triggers cleanup, but a killed launcher, lost machine/network, or
unavailable API can prevent deletion. Check the state file and RunPod Console in
that case. Ambiguous creation is never blindly retried: the launcher reconciles
the unique launch name with your pod list or reports that name for manual inspection.

## Private strategies

Name the dotted module in your JSON config, for example `my_strategies.example`,
and select the directory containing that package:

```bash
gpu-backtest runpod --config /path/to/job.json \
  --plugin-dir /path/to/strategy-project/src \
  --ssh-key ~/.ssh/runpod_ed25519 --output-dir runs/custom
```

Only `.py` files are copied from the selected directory. Git, venv, and cache
directories and non-Python files are excluded; source symlinks are rejected.
Put the strategy package directly beneath the selected directory. Plugins can
use Numba, NumPy, pandas, the engine, and the standard library. Extra dependencies,
non-Python package resources, namespace-only strategy entry packages, and private
package-index authentication are not installed automatically. Use an appropriate
manual route for these cases. The launcher's isolated venv does not inherit global
Python packages preinstalled in a custom image. Plugins are trusted code.

## Manual route

Create a Pod with the image above, exposing TCP port 22, then connect using the
SSH command shown by RunPod. Run these commands **inside the Pod**:

```bash
git clone https://github.com/howardc38/gpu-backtest-engine.git
cd gpu-backtest-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz,cuda]' 'cuda-bindings>=12.9.1,<13'
NUMBA_ENABLE_CUDASIM=0 gpu-backtest gpu-check --output runs/gpu_check.json
NUMBA_ENABLE_CUDASIM=0 gpu-backtest pipeline \
  --config examples/gpu_backtest_examples/rsi/config.json --output-dir runs/example
gpu-backtest charts --input runs/example/*_top_entry.csv runs/example/*_top_exit.csv \
  --output runs/example/charts.html
```

Download `runs/` before deleting the Pod. Manual pods are managed by you; the
launcher deletes only pods it creates. Small hardware checks establish GPU
compilation and explicit numeric cases, not large-grid speed or every strategy's
correctness.

See [the hardware validation record](../benchmarks/results/runpod_validation_20261003.md) for the tested image,
driver, package versions, numeric results, and successful cleanup.
