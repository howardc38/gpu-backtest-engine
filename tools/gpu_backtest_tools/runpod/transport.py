"""SSH transport and safe result extraction; no allocation logic."""

import contextlib
import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from .client import RunPodError


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
    return (address, port)


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
            if (
                path.is_absolute()
                or ".." in path.parts
                or (not (member.isfile() or member.isdir()))
            ):
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
