import numpy as np
import pytest

from cumine import MINE, pstats, cstats
from cumine._backend import cupy_available, native_cuda_available, backend_name


ESTIMATORS = ["high_fidelity", "fast"]


def available_devices():
    devices = ["cpu"]
    if cupy_available():
        devices.append("cupy")
    if native_cuda_available():
        devices.append("cuda")
    return devices


def make_cases(n=700):
    rng = np.random.default_rng(123)
    x = np.linspace(0, 1, n)

    return {
        "linear": (x, x),
        "quadratic": (x, (x - 0.5) ** 2),
        "sine": (x, np.sin(10 * np.pi * x)),
        "sine_plus_trend": (x, np.sin(10 * np.pi * x) + x),
        "noisy_sine": (x, np.sin(8 * np.pi * x) + rng.normal(0, 0.25, n)),
        "random_independent": (rng.uniform(0, 1, n), rng.uniform(0, 1, n)),
        "constant_y": (x, np.ones_like(x)),
    }


def assert_close(name, a, b, atol=1e-6, rtol=1e-6):
    assert np.allclose(a, b, atol=atol, rtol=rtol), (
        f"{name} mismatch: max abs diff={np.max(np.abs(np.asarray(a) - np.asarray(b)))}"
    )


@pytest.mark.parametrize("est", ESTIMATORS)
def test_single_pair_devices_match_cpu(est):
    cases = make_cases()

    for case_name, (x, y) in cases.items():
        ref = MINE(est=est, device="cpu")
        ref.compute_score(x, y)

        ref_mic = ref.mic()
        ref_tic = ref.tic(norm=True)

        assert np.isfinite(ref_mic)
        assert np.isfinite(ref_tic)
        assert 0.0 <= ref_mic <= 1.0000001

        if case_name == "linear":
            assert ref_mic > 0.98

        if case_name == "constant_y":
            assert ref_mic < 1e-10

        for device in available_devices():
            mine = MINE(est=est, device=device)
            mine.compute_score(x, y)

            assert mine.resolved_device_ == device
            assert_close(f"{est} {case_name} MIC cpu vs {device}", ref_mic, mine.mic())
            assert_close(f"{est} {case_name} TIC cpu vs {device}", ref_tic, mine.tic(norm=True))


@pytest.mark.parametrize("est", ESTIMATORS)
def test_pstats_devices_match_cpu(est):
    rng = np.random.default_rng(456)
    X = rng.standard_normal((7, 500))

    mic_cpu, tic_cpu = pstats(X, est=est, device="cpu")

    assert mic_cpu.shape == (7, 7)
    assert tic_cpu.shape == (7, 7)
    assert_close(f"{est} pstats MIC symmetry", mic_cpu, mic_cpu.T)
    assert_close(f"{est} pstats TIC symmetry", tic_cpu, tic_cpu.T)
    assert_close(f"{est} pstats MIC diagonal", np.diag(mic_cpu), np.ones(7))

    for device in available_devices():
        mic_dev, tic_dev = pstats(X, est=est, device=device)
        assert_close(f"{est} pstats MIC cpu vs {device}", mic_cpu, mic_dev)
        assert_close(f"{est} pstats TIC cpu vs {device}", tic_cpu, tic_dev)


@pytest.mark.parametrize("est", ESTIMATORS)
def test_cstats_devices_match_cpu(est):
    rng = np.random.default_rng(789)
    X = rng.standard_normal((6, 450))
    Y = rng.standard_normal((4, 450))

    mic_cpu, tic_cpu = cstats(X, Y, est=est, device="cpu")

    assert mic_cpu.shape == (6, 4)
    assert tic_cpu.shape == (6, 4)

    for device in available_devices():
        mic_dev, tic_dev = cstats(X, Y, est=est, device=device)
        assert_close(f"{est} cstats MIC cpu vs {device}", mic_cpu, mic_dev)
        assert_close(f"{est} cstats TIC cpu vs {device}", tic_cpu, tic_dev)


def test_default_estimator_is_high_fidelity():
    mine = MINE()
    assert mine.est == "high_fidelity"


def test_fast_pstats_uses_native_cuda_when_requested():
    if not native_cuda_available():
        pytest.skip("native CUDA extension is not built")

    rng = np.random.default_rng(999)
    X = rng.standard_normal((8, 600))

    pstats(X, est="fast", device="cuda")
    assert backend_name() == "cuda"


def test_invalid_estimator_rejected():
    with pytest.raises(ValueError):
        MINE(est="banana")
