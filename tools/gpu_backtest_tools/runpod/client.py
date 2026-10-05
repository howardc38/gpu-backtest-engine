"""RunPod authentication and idempotent API operations."""

import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from gpu_backtest import __version__

API_URL = "https://rest.runpod.io/v1"


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
            raise RunPodError(f"RunPod {method} {path}: HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RunPodError(f"RunPod {method} {path}: connection or response failure") from exc

    def create(self, body):
        try:
            result = self.request("POST", "/pods", body)
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("id"), str)
                or (not result["id"].isascii())
                or (not result["id"].isalnum())
            ):
                raise RunPodError("Creation response did not contain a valid pod ID")
        except RunPodError as exc:
            if "HTTP 4" in str(exc):
                raise
            try:
                pods = self.request("GET", "/pods")
                matches = [p for p in pods if p.get("name") == body["name"]]
            except (RunPodError, TypeError):
                matches = []
            if (
                len(matches) == 1
                and isinstance(matches[0].get("id"), str)
                and matches[0]["id"].isascii()
                and matches[0]["id"].isalnum()
            ):
                return matches[0]
            raise RunPodError(
                f"Creation outcome unknown; check RunPod for launch {body['name']}. No second creation request was sent."
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
