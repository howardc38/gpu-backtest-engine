import pytest
from helpers import check_matrix, check_rsi_reference
from numba import config, cuda

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_real_gpu():
    if config.ENABLE_CUDASIM:
        pytest.fail("GPU tests require NUMBA_ENABLE_CUDASIM=0")
    if not cuda.is_available():
        pytest.skip("No NVIDIA GPU is available")


def test_real_gpu_known_matrix(tmp_path, monkeypatch):
    check_matrix(tmp_path, monkeypatch)


def test_real_gpu_rsi_reference(tmp_path):
    check_rsi_reference(tmp_path)
