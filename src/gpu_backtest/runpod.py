"""Run a selected config on one leased RunPod GPU and reclaim the pod afterwards."""

import contextlib
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import __version__

API_URL = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
ENGINE_FILES = (
    "__init__.py",
    "__main__.py",
    "charts.py",
    "cli.py",
    "common.py",
    "engine.py",
    "gpu_check.py",
    "pipeline.py",
    "runpod.py",
    "splits.py",
    "statistics.py",
    "strategy.py",
    "tables.py",
    "strategies/__init__.py",
    "strategies/rsi_meanrev.py",
)
CONFIG_KEYS = {
    "strategy",
    "input",
    "buy",
    "sell",
    "entry_dims",
    "exit_dims",
    "top_n",
    "threads_per_block",
    "expected_interval",
    "num_splits",
    "start_date",
    "end_date",
    "ranges",
}
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


class RunPodError(RuntimeError):
    pass


def load_api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        path = Path.home() / ".runpod/config.toml"
        if path.is_file():
            key = tomllib.loads(path.read_text()).get("apikey")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Set RUNPOD_API_KEY or apikey in ~/.runpod/config.toml")
    return key.strip()


class RunPodClient:
    def __init__(self, key):
        self.key = key

    def request(self, method, path, body=None):
        request = urllib.request.Request(
            API_URL + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"gpu-backtest-engine/{__version__}",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            return json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            if method == "DELETE" and exc.code == 404:
                return None
            # Do not echo server bodies, auth headers, or credentials into logs.
            raise RunPodError(f"RunPod {method} {path}: HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RunPodError(f"RunPod {method} {path}: connection or response failure") from exc

    def create(self, body):
        try:
            result = self.request("POST", "/pods", body)
            if not isinstance(result, dict) or not result.get("id"):
                raise RunPodError("Creation response did not contain a pod ID")
        except RunPodError as exc:
            if "HTTP 4" in str(exc):
                raise
            # POST is never blindly retried: its outcome may already be a paid pod.
            try:
                pods = self.request("GET", "/pods")
                matches = [p for p in pods if p.get("name") == body["name"]]
            except (RunPodError, TypeError):
                matches = []
            if len(matches) == 1 and matches[0].get("id"):
                return matches[0]
            raise RunPodError(
                f"Creation outcome unknown; check RunPod for launch {body['name']}. "
                "No second creation request was sent."
            ) from exc
        return result

    def get(self, pod_id):
        return self.request("GET", "/pods/" + pod_id)

    def delete(self, pod_id):
        for attempt in range(3):
            try:
                self.request("DELETE", "/pods/" + pod_id)
                return
            except RunPodError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)


def endpoint(pod):
    address = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    if not address or not port:
        return None
    try:
        address = str(ipaddress.ip_address(address))
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValueError
    except (ValueError, TypeError):
        raise RunPodError("RunPod returned invalid public SSH connection details") from None
    return address, port


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise RunPodError("RunPod job exceeded its local time limit")
    return seconds


class SSHTransport:
    def __init__(self, address, port, key, output_dir):
        self.base = [
            "ssh",
            "-i",
            str(key),
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={json.dumps(str(Path(output_dir) / 'known_hosts'))}",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            f"root@{address}",
        ]
        self.environment = os.environ.copy()
        self.environment.pop("RUNPOD_API_KEY", None)

    def ready(self):
        result = subprocess.run(
            [*self.base, "true"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=15,
            env=self.environment,
        )
        return result.returncode == 0

    def _run(self, command, *, stdin=None, stdout=None, log=None, timeout):
        with contextlib.ExitStack() as stack:
            input_file = (
                stack.enter_context(Path(stdin).open("rb")) if stdin else subprocess.DEVNULL
            )
            output_file = stack.enter_context(Path(stdout).open("wb")) if stdout else None
            log_file = stack.enter_context(Path(log).open("ab")) if log else None
            process = subprocess.Popen(
                [*self.base, command],
                stdin=input_file,
                stdout=output_file or log_file,
                stderr=log_file,
                env=self.environment,
            )
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                process.kill()
                process.wait()
                raise
        if code:
            raise RunPodError(f"SSH stage failed with exit code {code}; inspect remote.log")

    def upload(self, archive, workspace, output_dir, timeout):
        command = f"mkdir -p {shlex.quote(workspace)} && tar xzf - --no-same-owner -C {shlex.quote(workspace)}"
        self._run(command, stdin=archive, log=Path(output_dir) / "remote.log", timeout=timeout)

    def execute(self, workspace, output_dir, timeout):
        self._run(
            f"cd {shlex.quote(workspace)} && bash job.sh",
            log=Path(output_dir) / "remote.log",
            timeout=timeout,
        )

    def download(self, workspace, output_dir, timeout):
        archive = Path(output_dir) / "results.tar.gz"
        self._run(
            f"cd {shlex.quote(workspace)}/output && tar czf - .",
            stdout=archive,
            log=Path(output_dir) / "remote.log",
            timeout=timeout,
        )
        extract_results(archive, Path(output_dir) / "results")


def extract_results(archive, destination):
    """Accept ordinary result files only; remote tar metadata is untrusted."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise RunPodError("Downloaded results contain an unsafe tar entry")
            if not (destination / path).resolve().is_relative_to(destination.resolve()):
                raise RunPodError("Downloaded results escape their destination")
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as output:
                    shutil.copyfileobj(handle.extractfile(member), output)


def _write_state(path, **fields):
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(fields)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _record_cleanup_state(path, **fields):
    """Failure to write metadata must not prevent or misreport API cleanup."""
    try:
        _write_state(path, **fields)
    except (OSError, ValueError) as exc:
        print(f"Could not update local state {path}: {type(exc).__name__}", flush=True)


def build_bundle(archive, *, mode, config_path=None, plugin_dir=None, charts=False):
    if mode not in ("run", "pipeline", "check"):
        raise ValueError("RunPod mode must be run, pipeline, or check")
    config = {}
    if mode != "check":
        if not config_path:
            raise ValueError("RunPod run/pipeline requires --config")
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
            raise ValueError("RunPod config must contain only documented engine config keys")
        if not config.get("strategy") or not config.get("input"):
            raise ValueError("RunPod config must specify strategy and input")
        if not isinstance(config["strategy"], str) or not all(
            p.isidentifier() for p in config["strategy"].split(".")
        ):
            raise ValueError("RunPod strategy must be a builtin name or dotted module name")
        source = (config_path.parent / config["input"]).resolve()
        from .engine import validate_market_data

        validate_market_data(source, expected_interval=config.get("expected_interval"))
    package = Path(__file__).parent
    distribution = importlib.metadata.distribution("gpu-backtest-engine")
    base_requires = [r for r in distribution.requires or [] if "extra ==" not in r]
    with tempfile.TemporaryDirectory(prefix="runpod-bundle-") as directory:
        staging = Path(directory)
        engine = staging / "engine"
        for relative in ENGINE_FILES:
            path = package / relative
            if path.is_symlink():
                raise ValueError("Engine bundle cannot include symlinked source files")
            output = engine / "gpu_backtest" / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
        shutil.copyfile(
            package / "runpod_requirements.txt", engine / "gpu_backtest/runpod_requirements.txt"
        )
        project = (
            '[build-system]\nrequires = ["setuptools>=77", "wheel"]\nbuild-backend = "setuptools.build_meta"\n'
            f'[project]\nname = "gpu-backtest-engine"\nversion = {json.dumps(__version__)}\n'
            'requires-python = ">=3.11,<3.14"\n'
            f"dependencies = {json.dumps(base_requires)}\n"
            '[project.scripts]\ngpu-backtest = "gpu_backtest.cli:main"\n'
            '[tool.setuptools.packages.find]\nwhere = ["."]\n'
            '[tool.setuptools.package-data]\ngpu_backtest = ["runpod_requirements.txt"]\n'
        )
        (engine / "pyproject.toml").write_text(project)
        for entry in distribution.files or []:
            if str(entry).endswith("/licenses/LICENSE"):
                shutil.copyfile(distribution.locate_file(entry), engine / "LICENSE")
                break
        if mode != "check":
            (staging / "input").mkdir()
            shutil.copyfile(source, staging / "input/market.csv")
            config["input"] = "input/market.csv"
            (staging / "job.json").write_text(json.dumps(config, indent=2) + "\n")
        if plugin_dir:
            plugin_dir = Path(plugin_dir).resolve()
            if not plugin_dir.is_dir():
                raise ValueError("Plugin directory does not exist")
            copied = 0
            for path in sorted(plugin_dir.rglob("*.py")):
                relative = path.relative_to(plugin_dir)
                if set(relative.parts) & IGNORED_PARTS:
                    continue
                if path.is_symlink():
                    raise ValueError("Plugin source symlinks are not supported")
                output = staging / "plugins" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
                copied += 1
            if not copied:
                raise ValueError("Plugin directory contains no Python files")
        if mode != "check" and config["strategy"] not in (
            "rsi_meanrev",
            "gpu_backtest.strategies.rsi_meanrev",
        ):
            relative = Path(*config["strategy"].split("."))
            if (
                not (staging / "plugins" / relative.with_suffix(".py")).is_file()
                and not (staging / "plugins" / relative / "__init__.py").is_file()
            ):
                raise ValueError("External strategy must be present in --plugin-dir")
        (staging / "job.sh").write_text(remote_script(mode, charts=charts))
        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "file": path.relative_to(staging).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        (staging / "bundle_manifest.json").write_text(json.dumps(files, indent=2) + "\n")
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    handle.add(path, arcname=path.relative_to(staging).as_posix(), recursive=False)
    return files


def remote_script(mode, *, charts=False):
    script = """#!/bin/bash
set -euo pipefail
export NUMBA_ENABLE_CUDASIM=0
export PYTHONPATH="$PWD/plugins${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p output
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Python 3.11-3.13 required"'
python3 -m venv env
env/bin/python -m pip install --disable-pip-version-check ./engine -r engine/gpu_backtest/runpod_requirements.txt
nvidia-smi --query-gpu=name,driver_version --format=csv > output/gpu.csv
env/bin/python -m gpu_backtest gpu-check --output output/gpu_check.json
"""
    if mode == "run":
        script += (
            "env/bin/python -m gpu_backtest run --config job.json --out-prefix output/result\n"
        )
    elif mode == "pipeline":
        script += "env/bin/python -m gpu_backtest pipeline --config job.json --output-dir output/pipeline\n"
    if charts and mode != "check":
        folder = "output/pipeline" if mode == "pipeline" else "output"
        script += "env/bin/python -m pip install 'altair==6.3.0'\n"
        script += f"env/bin/python -m gpu_backtest charts --input {folder}/*_top_entry.csv {folder}/*_top_exit.csv --output output/charts.html\n"
    script += "env/bin/python -m pip freeze > output/requirements-resolved.txt\n"
    return script


@contextlib.contextmanager
def defer_creation_signals():
    """Record Ctrl-C/termination until ownership of a creation reply is recorded."""
    interrupted = [False]
    if threading.current_thread() is not threading.main_thread():
        yield interrupted
        return
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def defer(signum, frame):
        interrupted[0] = True

    for number in previous:
        signal.signal(number, defer)
    try:
        yield interrupted
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextlib.contextmanager
def terminate_on_signal():
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("RunPod launcher interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def launch(
    *,
    config_path=None,
    output_dir,
    ssh_key=None,
    mode="pipeline",
    plugin_dir=None,
    image=DEFAULT_IMAGE,
    gpu="NVIDIA GeForce RTX 4090",
    cloud="SECURE",
    max_seconds=1800,
    max_hourly_rate=1.0,
    keep_pod=False,
    charts=False,
    dry_run=False,
    client=None,
    transport_factory=SSHTransport,
):
    if not isinstance(max_seconds, int) or max_seconds < 1:
        raise ValueError("max_seconds must be a positive integer")
    if not math.isfinite(max_hourly_rate) or max_hourly_rate <= 0:
        raise ValueError("max_hourly_rate must be finite and positive")
    if cloud not in ("SECURE", "COMMUNITY"):
        raise ValueError("cloud must be SECURE or COMMUNITY")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("RunPod output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = build_bundle(
        output_dir / "upload.tar.gz",
        mode=mode,
        config_path=config_path,
        plugin_dir=plugin_dir,
        charts=charts,
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "image": image,
                    "gpu": gpu,
                    "cloud": cloud,
                    "files": [f["file"] for f in files],
                    "archive": str(output_dir / "upload.tar.gz"),
                    "pod_created": False,
                },
                indent=2,
            )
        )
        return output_dir
    if not ssh_key or not Path(ssh_key).expanduser().is_file():
        raise ValueError("Specify an existing --ssh-key; add its public key to RunPod Credentials")
    key = Path(ssh_key).expanduser().resolve()
    if not shutil.which("ssh"):
        raise ValueError("An OpenSSH client is required")
    client = client or RunPodClient(load_api_key())
    launch_name = "gpu-backtest-" + uuid.uuid4().hex[:12]
    body = {
        "name": launch_name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "cloudType": cloud,
        "computeType": "GPU",
        "containerDiskInGb": 30,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
    }
    public_key = Path(str(key) + ".pub")
    if public_key.is_file():
        body["env"] = {"PUBLIC_KEY": " ".join(public_key.read_text().split()[:2])}
    state_path = output_dir / "runpod_state.json"
    _write_state(state_path, name=launch_name, image=image, gpu=gpu, status="creating")
    pod_id = None
    failure = None
    keep_allowed = False
    print(f"Creating one {gpu} pod on {cloud}...", flush=True)
    try:
        with defer_creation_signals() as cancelled:
            pod = client.create(body)
            candidate_id = pod["id"]
            if not isinstance(candidate_id, str) or not candidate_id.isalnum():
                raise RunPodError("RunPod returned an invalid pod ID")
            pod_id = candidate_id
            _write_state(state_path, pod_id=pod_id, status="starting")
        if cancelled[0]:
            raise KeyboardInterrupt("RunPod creation was cancelled")
        deadline = time.monotonic() + max_seconds
        quoted_rate = pod.get("adjustedCostPerHr")
        if quoted_rate is None:
            quoted_rate = pod.get("costPerHr")
        rate = float(quoted_rate) if quoted_rate is not None else math.inf
        if not math.isfinite(rate) or rate < 0 or rate > max_hourly_rate:
            raise RunPodError(f"Pod rate exceeds the requested {max_hourly_rate:g}/hour limit")
        keep_allowed = True
        _write_state(state_path, hourly_rate=rate)
        print(f"Pod {pod_id}: ${rate:g}/hour. Waiting for SSH...", flush=True)
        while True:
            _remaining(deadline)
            details = client.get(pod_id)
            connection = endpoint(details)
            if connection:
                transport = transport_factory(*connection, key, output_dir)
                try:
                    if transport.ready():
                        break
                except subprocess.TimeoutExpired:
                    pass
            time.sleep(min(5, _remaining(deadline)))
        workspace = "/workspace/" + launch_name
        _write_state(state_path, status="uploading")
        print("Uploading selected engine/data/plugin files...", flush=True)
        transport.upload(output_dir / "upload.tar.gz", workspace, output_dir, _remaining(deadline))
        _write_state(state_path, status="running")
        print(
            "Installing environment and running GPU checks/job; progress is in remote.log...",
            flush=True,
        )
        transport.execute(workspace, output_dir, _remaining(deadline))
        _write_state(state_path, status="downloading")
        print("Downloading results...", flush=True)
        transport.download(workspace, output_dir, _remaining(deadline))
        _write_state(state_path, status="completed")
    except BaseException as exc:
        failure = exc
        _record_cleanup_state(state_path, status="failed", error=type(exc).__name__)
        raise
    finally:
        if pod_id:
            if keep_pod and keep_allowed:
                _record_cleanup_state(state_path, cleanup="kept")
                print(
                    f"Pod {pod_id} retained by --keep-pod; it continues to incur charges.",
                    flush=True,
                )
            else:
                try:
                    client.delete(pod_id)
                except BaseException as exc:
                    _record_cleanup_state(state_path, cleanup="failed")
                    message = f"Could not delete pod {pod_id}; delete it in RunPod Console. State: {state_path}"
                    print(message, flush=True)
                    if failure is None:
                        raise RunPodError(message) from exc
                else:
                    _record_cleanup_state(state_path, cleanup="deleted")
                    print(f"Pod {pod_id} deleted.", flush=True)
    return output_dir
