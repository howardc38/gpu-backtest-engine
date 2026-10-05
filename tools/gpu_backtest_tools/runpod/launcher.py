"""Own the paid-pod lifecycle, deadlines, cancellation, and cleanup state."""

import contextlib
import json
import math
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .bundle import build_bundle
from .client import RunPodClient, RunPodError, load_api_key
from .transport import SSHTransport, endpoint

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise RunPodError("RunPod job exceeded its local time limit")
    return seconds


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
    mode="run",
    plugin_dir=None,
    image=DEFAULT_IMAGE,
    gpu="NVIDIA GeForce RTX 4090",
    cloud="SECURE",
    max_seconds=1800,
    max_hourly_rate=1.0,
    keep_pod=False,
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
    if mode == "benchmark":
        body["minVCPUPerGPU"] = 8
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
