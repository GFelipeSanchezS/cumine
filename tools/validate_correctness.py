import numpy as np
from cumine import MINE, pstats, cstats
from cumine._backend import native_cuda_available

ATOL = 1e-6
RTOL = 1e-6

def assert_close(name, a, b, atol=ATOL, rtol=RTOL):
    if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False):
        diff = np.max(np.abs(np.asarray(a) - np.asarray(b)))
        raise AssertionError(f"{name} mismatch: max abs diff={diff}")
    print(f"{name:35s} ✓")

def check_single_pair():
    rng = np.random.default_rng(123)

    cases = {
        "linear": (
            np.linspace(0, 1, 2000),
            np.linspace(0, 1, 2000),
        ),
        "quadratic": (
            np.linspace(-1, 1, 2000),
            np.linspace(-1, 1, 2000) ** 2,
        ),
        "sine": (
            np.linspace(0, 1, 2000),
            np.sin(10 * np.pi * np.linspace(0, 1, 2000)),
        ),
        "random_independent": (
            rng.uniform(0, 1, 2000),
            rng.uniform(0, 1, 2000),
        ),
        "noisy_function": (
            np.linspace(0, 1, 2000),
            np.sin(8 * np.pi * np.linspace(0, 1, 2000)) + rng.normal(0, 0.3, 2000),
        ),
    }

    for name, (x, y) in cases.items():
        m_cpu = MINE(device="cpu", est="mic_approx")
        m_cpu.compute_score(x, y)

        m_cuda = MINE(device="cuda", est="mic_approx")
        m_cuda.compute_score(x, y)

        assert_close(f"single MIC {name}", m_cpu.mic(), m_cuda.mic())
        assert_close(f"single TIC {name}", m_cpu.tic(norm=True), m_cuda.tic(norm=True))

def check_pstats():
    rng = np.random.default_rng(456)
    X = rng.standard_normal((12, 1500))

    mic_cpu, tic_cpu = pstats(X, device="cpu", est="mic_approx")
    mic_cuda, tic_cuda = pstats(X, device="cuda", est="mic_approx")

    assert_close("pstats MIC matrix", mic_cpu, mic_cuda)
    assert_close("pstats TIC matrix", tic_cpu, tic_cuda)

    assert_close("pstats MIC symmetry", mic_cuda, mic_cuda.T)
    assert_close("pstats TIC symmetry", tic_cuda, tic_cuda.T)
    assert_close("pstats MIC diagonal", np.diag(mic_cuda), np.ones(mic_cuda.shape[0]))

def check_cstats():
    rng = np.random.default_rng(789)
    X = rng.standard_normal((8, 1200))
    Y = rng.standard_normal((6, 1200))

    mic_cpu, tic_cpu = cstats(X, Y, device="cpu", est="mic_approx")
    mic_cuda, tic_cuda = cstats(X, Y, device="cuda", est="mic_approx")

    assert_close("cstats MIC matrix", mic_cpu, mic_cuda)
    assert_close("cstats TIC matrix", tic_cpu, tic_cuda)

if __name__ == "__main__":
    if not native_cuda_available():
        raise SystemExit("Native CUDA extension is not available. Build with CUMINE_BUILD_CUDA=1 pip install -e .")

    check_single_pair()
    check_pstats()
    check_cstats()
    print("\nAll CPU-vs-CUDA correctness checks passed.")
