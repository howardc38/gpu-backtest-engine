import os
import subprocess
import sys
from pathlib import Path


def test_isolated_simulator_suite():
    original = os.environ.get("NUMBA_ENABLE_CUDASIM")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).with_name("simulator_cases.py")),
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        env={**os.environ, "NUMBA_ENABLE_CUDASIM": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.environ.get("NUMBA_ENABLE_CUDASIM") == original
