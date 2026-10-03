import io
import json
import os
import signal
import tarfile
import urllib.error
from pathlib import Path

import pytest

from gpu_backtest import runpod

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, rate=0.5, cleanup_error=False):
        self.rate = rate
        self.cleanup_error = cleanup_error
        self.created = []
        self.deleted = []

    def create(self, body):
        self.created.append(body)
        return {"id": "ownedpod", "costPerHr": self.rate}

    def get(self, pod_id):
        assert pod_id == "ownedpod"
        return {"publicIp": "203.0.113.1", "portMappings": {"22": 2200}}

    def delete(self, pod_id):
        self.deleted.append(pod_id)
        if self.cleanup_error:
            raise runpod.RunPodError("Deletion unavailable")


class Transport:
    failure = None

    def __init__(self, *args):
        self.stages = []

    def ready(self):
        return True

    def stage(self, name):
        self.stages.append(name)
        if name == self.failure:
            raise runpod.RunPodError("Injected stage failure")
        if self.failure == "interrupt" and name == "execute":
            raise KeyboardInterrupt

    def upload(self, *args):
        self.stage("upload")

    def execute(self, *args):
        self.stage("execute")

    def download(self, workspace, output, timeout):
        self.stage("download")
        (Path(output) / "results").mkdir()


def launch_options(tmp_path, client):
    key = tmp_path / "ssh_key"
    key.write_text("test-key")
    return dict(
        config_path=ROOT / "examples/rsi.json",
        output_dir=tmp_path / "output",
        ssh_key=key,
        client=client,
        transport_factory=Transport,
    )


def state(tmp_path):
    return json.loads((tmp_path / "output/runpod_state.json").read_text())


def test_success_deletes_only_the_newly_created_pod(tmp_path):
    client = Client()
    runpod.launch(**launch_options(tmp_path, client))
    assert len(client.created) == 1
    assert client.deleted == ["ownedpod"]
    assert state(tmp_path)["status"] == "completed"
    assert state(tmp_path)["cleanup"] == "deleted"


@pytest.mark.parametrize("stage", ["upload", "execute", "download", "interrupt"])
def test_every_failure_and_cancellation_cleans_up(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(Transport, "failure", stage)
    client = Client()
    with pytest.raises((runpod.RunPodError, KeyboardInterrupt)):
        runpod.launch(**launch_options(tmp_path, client))
    assert client.deleted == ["ownedpod"]
    assert state(tmp_path)["status"] == "failed"
    assert state(tmp_path)["cleanup"] == "deleted"


def test_cleanup_failure_is_reported_even_after_success(tmp_path):
    client = Client(cleanup_error=True)
    with pytest.raises(runpod.RunPodError, match="delete it in RunPod Console"):
        runpod.launch(**launch_options(tmp_path, client))
    assert state(tmp_path)["cleanup"] == "failed"
    assert state(tmp_path)["pod_id"] == "ownedpod"


def test_keep_pod_is_explicit(tmp_path):
    client = Client()
    runpod.launch(**launch_options(tmp_path, client), keep_pod=True)
    assert client.deleted == [] and state(tmp_path)["cleanup"] == "kept"


def test_local_state_failure_does_not_prevent_pod_deletion(tmp_path, monkeypatch):
    original = runpod._write_state

    def fail_after_creation(path, **fields):
        if fields.get("status") in ("completed", "failed") or "cleanup" in fields:
            raise OSError("Disk unavailable")
        original(path, **fields)

    monkeypatch.setattr(runpod, "_write_state", fail_after_creation)
    client = Client()
    with pytest.raises(OSError, match="Disk unavailable"):
        runpod.launch(**launch_options(tmp_path, client))
    assert client.deleted == ["ownedpod"]


def test_startup_api_failure_deletes_allocated_pod(tmp_path, monkeypatch):
    client = Client()

    def fail(pod_id):
        raise runpod.RunPodError("Startup unavailable")

    monkeypatch.setattr(client, "get", fail)
    with pytest.raises(runpod.RunPodError, match="Startup unavailable"):
        runpod.launch(**launch_options(tmp_path, client))
    assert client.deleted == ["ownedpod"]


def test_local_timeout_deletes_allocated_pod(tmp_path, monkeypatch):
    ticks = iter([100.0, 200.0])
    monkeypatch.setattr(runpod.time, "monotonic", lambda: next(ticks))
    client = Client()
    with pytest.raises(runpod.RunPodError, match="time limit"):
        runpod.launch(**launch_options(tmp_path, client), max_seconds=1)
    assert client.deleted == ["ownedpod"]


@pytest.mark.parametrize("rate", [2.0, None, float("nan")])
def test_rate_guard_deletes_even_with_keep_pod(tmp_path, rate):
    client = Client(rate=rate)
    with pytest.raises(runpod.RunPodError, match="rate exceeds"):
        runpod.launch(**launch_options(tmp_path, client), keep_pod=True)
    assert client.deleted == ["ownedpod"]


def test_cancel_during_creation_records_ownership_before_cleanup(tmp_path):
    class CancelClient(Client):
        def create(self, body):
            os.kill(os.getpid(), signal.SIGINT)
            return super().create(body)

    client = CancelClient()
    with pytest.raises(KeyboardInterrupt):
        runpod.launch(**launch_options(tmp_path, client))
    assert client.deleted == ["ownedpod"]
    assert state(tmp_path)["cleanup"] == "deleted"


def test_dry_run_never_reads_credentials_or_creates_a_pod(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("Credentials must not be read")

    monkeypatch.setattr(runpod, "load_api_key", forbidden)
    client = Client()
    runpod.launch(
        config_path=ROOT / "examples/rsi.json",
        output_dir=tmp_path / "output",
        dry_run=True,
        client=client,
    )
    assert client.created == []
    assert not (tmp_path / "output/runpod_state.json").exists()


def test_bundle_contains_only_selected_python_plugin_and_data(tmp_path):
    plugins = tmp_path / "plugins"
    (plugins / "custom").mkdir(parents=True)
    (plugins / "custom/__init__.py").write_text("")
    (plugins / "custom/strategy.py").write_text('NAME="custom"\n')
    (plugins / ".git").mkdir()
    (plugins / ".git/unrelated.py").write_text("do not upload")
    (plugins / ".env").write_text("do not upload")
    (plugins / "ssh_key").write_text("do not upload")
    config = json.loads((ROOT / "examples/rsi.json").read_text())
    config.update(strategy="custom.strategy", input=str(ROOT / "examples/synthetic.csv"))
    path = tmp_path / "job.json"
    path.write_text(json.dumps(config))
    archive = tmp_path / "bundle.tar.gz"
    files = runpod.build_bundle(archive, mode="run", config_path=path, plugin_dir=plugins)
    names = {f["file"] for f in files}
    assert {"plugins/custom/strategy.py", "plugins/custom/__init__.py", "engine/LICENSE"}.issubset(
        names
    )
    assert not any(".git" in n or ".env" in n or "ssh_key" in n for n in names)
    with tarfile.open(archive) as handle:
        normalized = json.load(handle.extractfile("job.json"))
    assert normalized["input"] == "input/market.csv"
    assert str(tmp_path) not in json.dumps(normalized)


def test_missing_plugin_and_unknown_config_fields_fail_before_creation(tmp_path):
    path = tmp_path / "job.json"
    config = json.loads((ROOT / "examples/rsi.json").read_text())
    config.update(strategy="missing.strategy", input=str(ROOT / "examples/synthetic.csv"))
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="--plugin-dir"):
        runpod.build_bundle(tmp_path / "upload.tar.gz", mode="run", config_path=path)
    config["unexpected"] = "not-an-engine-option"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="documented engine config"):
        runpod.build_bundle(tmp_path / "upload.tar.gz", mode="run", config_path=path)


def test_plugin_symlinks_are_rejected(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (tmp_path / "outside.py").write_text("outside")
    (plugins / "linked.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="symlinks"):
        runpod.build_bundle(tmp_path / "upload.tar.gz", mode="check", plugin_dir=plugins)


@pytest.mark.parametrize(
    "name,kind", [("../escape", "file"), ("/absolute", "file"), ("link", "symlink")]
)
def test_download_rejects_path_traversal_and_links(tmp_path, name, kind):
    archive = tmp_path / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        entry = tarfile.TarInfo(name)
        if kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../escape"
            handle.addfile(entry)
        else:
            entry.size = 1
            handle.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(runpod.RunPodError, match="unsafe tar"):
        runpod.extract_results(archive, tmp_path / "results")
    assert not (tmp_path / "escape").exists()


def test_auth_headers_and_server_bodies_are_not_exposed(monkeypatch):
    key = "test-token"

    def fail(request, timeout):
        assert request.get_header("Authorization") == f"Bearer {key}"
        assert request.get_header("User-agent").startswith("gpu-backtest-engine/")
        raise urllib.error.HTTPError(
            request.full_url, 403, "Forbidden", {}, io.BytesIO(key.encode())
        )

    monkeypatch.setattr(runpod.urllib.request, "urlopen", fail)
    with pytest.raises(runpod.RunPodError) as error:
        runpod.RunPodClient(key).request("GET", "/pods")
    assert key not in str(error.value)
    assert "HTTP 403" in str(error.value)


@pytest.mark.parametrize("ambiguous", ["timeout", "missing_id", "invalid_id"])
def test_ambiguous_creation_recovers_by_unique_name_without_reposting(monkeypatch, ambiguous):
    calls = []
    client = runpod.RunPodClient("test-token")

    def request(method, path, body=None):
        calls.append(method)
        if method == "POST":
            if ambiguous == "timeout":
                raise runpod.RunPodError("Connection failed")
            if ambiguous == "invalid_id":
                return {"id": "../invalid"}
            return {}
        return [{"id": "unrelated", "name": "other"}, {"id": "ownedpod", "name": "unique"}]

    monkeypatch.setattr(client, "request", request)
    assert client.create({"name": "unique"})["id"] == "ownedpod"
    assert calls == ["POST", "GET"]


def test_uncertain_creation_does_not_blindly_retry(monkeypatch):
    calls = []
    client = runpod.RunPodClient("test-token")

    def request(method, path, body=None):
        calls.append(method)
        if method == "POST":
            raise runpod.RunPodError("Connection failed")
        return []

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(runpod.RunPodError, match="outcome unknown"):
        client.create({"name": "unique"})
    assert calls == ["POST", "GET"]


def test_api_key_is_removed_from_ssh_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-token")
    transport = runpod.SSHTransport("203.0.113.1", 2200, tmp_path / "key", tmp_path)
    assert "RUNPOD_API_KEY" not in transport.environment
    assert "StrictHostKeyChecking=accept-new" in transport.base


@pytest.mark.parametrize(
    "pod",
    [
        {"publicIp": "-oProxyCommand=bad", "portMappings": {"22": 22}},
        {"publicIp": "203.0.113.1", "portMappings": {"22": 70000}},
    ],
)
def test_invalid_server_ssh_metadata_is_rejected(pod):
    with pytest.raises(runpod.RunPodError, match="invalid public SSH"):
        runpod.endpoint(pod)
