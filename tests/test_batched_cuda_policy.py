import numpy as np
import pytest

from cumine import MINE, cstats, pstats
from cumine._backend import native_cuda_available


def test_pstats_cpu_basic_high_fidelity():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 200))
    mic, tic = pstats(X, device="cpu")
    assert mic.shape == (4, 4)
    assert tic.shape == (4, 4)
    assert np.allclose(mic, mic.T)
    assert np.allclose(tic, tic.T)
    assert np.allclose(np.diag(mic), 1.0)
    assert np.allclose(np.diag(tic), 1.0)


def test_cstats_cpu_basic_high_fidelity():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 200))
    Y = rng.standard_normal((2, 200))
    mic, tic = cstats(X, Y, device="cpu")
    assert mic.shape == (3, 2)
    assert tic.shape == (3, 2)
    assert np.all(np.isfinite(mic))
    assert np.all(np.isfinite(tic))


def test_pstats_cuda_matches_cpu_fast_if_available():
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    rng = np.random.default_rng(2)
    X = rng.standard_normal((5, 300))
    X[1] = X[0]

    mic_cpu, tic_cpu = pstats(X, alpha=0.6, est="fast", device="cpu")
    mic_cuda, tic_cuda = pstats(X, alpha=0.6, est="fast", device="cuda")

    assert mic_cuda.shape == mic_cpu.shape
    assert tic_cuda.shape == tic_cpu.shape
    assert np.allclose(mic_cuda, mic_cpu, atol=1e-10)
    assert np.allclose(tic_cuda, tic_cpu, atol=1e-10)


def test_cstats_cuda_matches_cpu_fast_if_available():
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    rng = np.random.default_rng(3)
    X = rng.standard_normal((4, 250))
    Y = rng.standard_normal((3, 250))
    Y[0] = X[2]

    mic_cpu, tic_cpu = cstats(X, Y, alpha=0.6, est="fast", device="cpu")
    mic_cuda, tic_cuda = cstats(X, Y, alpha=0.6, est="fast", device="cuda")

    assert mic_cuda.shape == mic_cpu.shape
    assert tic_cuda.shape == tic_cpu.shape
    assert np.allclose(mic_cuda, mic_cpu, atol=1e-10)
    assert np.allclose(tic_cuda, tic_cpu, atol=1e-10)


def test_single_pair_high_fidelity_cuda_matches_cpu_if_available():
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    x = np.linspace(0, 1, 500)
    y = np.sin(10 * np.pi * x) + x

    m_cpu = MINE(device="cpu", est="high_fidelity")
    m_cuda = MINE(device="cuda", est="high_fidelity")
    m_cpu.compute_score(x, y)
    m_cuda.compute_score(x, y)

    assert m_cuda.resolved_device_ == "cuda"
    assert np.allclose(m_cuda.mic(), m_cpu.mic(), atol=1e-10)
    assert np.allclose(m_cuda.tic(norm=True), m_cpu.tic(norm=True), atol=1e-10)


def test_single_pair_fast_cuda_matches_cpu_if_available():
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    x = np.linspace(0, 1, 500)
    y = x ** 2
    m_cpu = MINE(device="cpu", est="fast")
    m_cuda = MINE(device="cuda", est="fast")
    m_cpu.compute_score(x, y)
    m_cuda.compute_score(x, y)
    assert np.allclose(m_cuda.mic(), m_cpu.mic(), atol=1e-10)


def test_pstats_constant_rows_are_zero_off_diagonal_cpu():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((4, 150))
    X[0] = 1.0
    mic, tic = pstats(X, alpha=0.6, est="fast", device="cpu")
    assert np.allclose(np.diag(mic), 1.0)
    assert np.allclose(mic[0, 1:], 0.0)
    assert np.allclose(mic[1:, 0], 0.0)


def test_cstats_constant_rows_are_zero_cpu():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((3, 150))
    Y = rng.standard_normal((2, 150))
    X[1] = 2.0
    Y[0] = -1.0
    mic, tic = cstats(X, Y, alpha=0.6, est="fast", device="cpu")
    assert np.allclose(mic[1, :], 0.0)
    assert np.allclose(mic[:, 0], 0.0)
