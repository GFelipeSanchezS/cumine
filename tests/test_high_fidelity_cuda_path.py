import numpy as np
import pytest

from cumine import MINE, pstats, cstats
from cumine._backend import native_cuda_available


def test_single_pair_high_fidelity_cuda_calls_native_path(monkeypatch):
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    import cumine._cuda_native as cuda_native

    def sentinel(*args, **kwargs):
        raise RuntimeError("sentinel: high_fidelity_scores was called")

    monkeypatch.setattr(cuda_native, "high_fidelity_scores", sentinel)

    x = np.linspace(0, 1, 300)
    y = np.sin(10 * np.pi * x) + x

    mine = MINE(est="high_fidelity", device="cuda")

    with pytest.raises(RuntimeError, match="sentinel: high_fidelity_scores was called"):
        mine.compute_score(x, y)


def test_pstats_high_fidelity_cuda_calls_batched_native_path(monkeypatch):
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    import cumine._cuda_native as cuda_native

    def sentinel(*args, **kwargs):
        raise RuntimeError("sentinel: batch_high_fidelity_mic_tic was called")

    monkeypatch.setattr(cuda_native, "batch_high_fidelity_mic_tic", sentinel)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 300))

    with pytest.raises(RuntimeError, match="sentinel: batch_high_fidelity_mic_tic was called"):
        pstats(X, est="high_fidelity", device="cuda")


def test_cstats_high_fidelity_cuda_calls_batched_native_path(monkeypatch):
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    import cumine._cuda_native as cuda_native

    def sentinel(*args, **kwargs):
        raise RuntimeError("sentinel: batch_high_fidelity_mic_tic was called")

    monkeypatch.setattr(cuda_native, "batch_high_fidelity_mic_tic", sentinel)

    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 300))
    Y = rng.standard_normal((3, 300))

    with pytest.raises(RuntimeError, match="sentinel: batch_high_fidelity_mic_tic was called"):
        cstats(X, Y, est="high_fidelity", device="cuda")
