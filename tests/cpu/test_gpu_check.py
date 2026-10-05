from types import SimpleNamespace

import pytest
from gpu_backtest_tools.checks import gpu as gpu_check


def test_gpu_check_refuses_simulation(monkeypatch):
    monkeypatch.setattr(gpu_check, "config", SimpleNamespace(ENABLE_CUDASIM=True))
    with pytest.raises(ValueError, match="simulation is not GPU validation"):
        gpu_check.check_gpu()


def test_gpu_check_refuses_missing_hardware(monkeypatch):
    monkeypatch.setattr(gpu_check, "config", SimpleNamespace(ENABLE_CUDASIM=False))
    monkeypatch.setattr(gpu_check, "cuda", SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ValueError, match="No usable NVIDIA"):
        gpu_check.check_gpu()
