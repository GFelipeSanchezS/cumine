import numpy as np
import pytest

from cumine import MINE, cstats, pstats
from cumine._backend import cupy_available, native_cuda_available


def test_cpu_device():
    x = np.linspace(0, 1, 300)
    y = x
    mine = MINE(device="cpu")
    mine.compute_score(x, y)
    assert mine.resolved_device_ == "cpu"
    assert abs(mine.mic() - 1.0) < 0.02


def test_default_estimator_is_high_fidelity():
    mine = MINE()
    assert mine.est == "high_fidelity"


def test_auto_device_small_pair():
    x = np.linspace(0, 1, 300)
    y = x ** 2
    mine = MINE(device="auto")
    mine.compute_score(x, y)
    assert mine.resolved_device_ in {"cpu", "cupy", "cuda"}
    assert np.isfinite(mine.mic())


def test_cupy_device_if_available():
    if not cupy_available():
        pytest.skip("CuPy is not available")
    x = np.linspace(0, 1, 300)
    y = x
    mine = MINE(device="cupy")
    mine.compute_score(x, y)
    assert mine.resolved_device_ == "cupy"
    assert abs(mine.mic() - 1.0) < 0.02


def test_native_cuda_device_if_available():
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")
    x = np.linspace(0, 1, 300)
    y = x
    mine = MINE(device="cuda")
    mine.compute_score(x, y)
    assert mine.resolved_device_ == "cuda"
    assert abs(mine.mic() - 1.0) < 0.02


def test_invalid_device():
    with pytest.raises(ValueError):
        MINE(device="banana").compute_score(np.arange(10), np.arange(10))


def test_invalid_estimator():
    with pytest.raises(ValueError):
        MINE(est="banana").compute_score(np.arange(10), np.arange(10))


def test_pstats_auto_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 200))
    mic, tic = pstats(X, device="auto")
    assert mic.shape == (4, 4)
    assert tic.shape == (4, 4)
    assert np.allclose(mic, mic.T)


def test_cstats_auto_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 200))
    Y = rng.standard_normal((2, 200))
    mic, tic = cstats(X, Y, device="auto")
    assert mic.shape == (3, 2)
    assert tic.shape == (3, 2)
