import numpy as np

from cumine import MINE


def score(x, y, est):
    mine = MINE(est=est, device="cpu")
    mine.compute_score(x, y)
    return mine.mic()


def test_random_independent_is_not_perfect():
    rng = np.random.default_rng(123)
    x = rng.uniform(0, 1, 1000)
    y = rng.uniform(0, 1, 1000)

    for est in ["fast", "high_fidelity"]:
        mic = score(x, y, est)
        assert mic < 0.35, f"{est} gives suspiciously high MIC for random independent data: {mic}"


def test_noise_reduces_sine_score():
    rng = np.random.default_rng(456)
    x = np.linspace(0, 1, 1000)
    y_clean = np.sin(10 * np.pi * x)
    y_noisy = y_clean + rng.normal(0, 0.6, len(x))

    for est in ["fast", "high_fidelity"]:
        clean = score(x, y_clean, est)
        noisy = score(x, y_noisy, est)
        assert noisy < clean, f"{est} did not reduce MIC under noise: clean={clean}, noisy={noisy}"


def test_constant_variable_is_zero():
    x = np.linspace(0, 1, 1000)
    y = np.ones_like(x)

    for est in ["fast", "high_fidelity"]:
        mic = score(x, y, est)
        assert mic < 1e-10, f"{est} constant variable should have MIC near zero, got {mic}"
