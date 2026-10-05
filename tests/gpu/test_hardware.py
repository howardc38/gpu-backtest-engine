import pytest
from fixtures import check_matrix, check_rsi_reference, check_single_output
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


@pytest.mark.parametrize(
    "exit_open,fee,total,square",
    [
        (120, 0.0, 50.0, 2500.0),
        (120, 0.0015, 49.625, 2462.640625),
        (60, 0.0, -25.0, 625.0),
    ],
)
def test_single_combination_raw_output(tmp_path, exit_open, fee, total, square):
    check_single_output(tmp_path, exit_open, fee, total, square)
