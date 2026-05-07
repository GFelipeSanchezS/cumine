import numpy as np

from cumine import MINE, pstats


def test_high_fidelity_and_fast_are_valid_estimators():
    x = np.linspace(0, 1, 400)
    y = np.sin(10 * np.pi * x) + x

    hf = MINE(est="high_fidelity", device="cpu")
    fast = MINE(est="fast", device="cpu")
    hf.compute_score(x, y)
    fast.compute_score(x, y)

    assert np.isfinite(hf.mic())
    assert np.isfinite(fast.mic())


def test_high_fidelity_sine_not_lower_than_fast_on_structured_case():
    x = np.linspace(0, 1, 500)
    y = np.sin(10 * np.pi * x) + x

    hf = MINE(est="high_fidelity", device="cpu")
    fast = MINE(est="fast", device="cpu")
    hf.compute_score(x, y)
    fast.compute_score(x, y)

    assert hf.mic() >= fast.mic()


def test_estimator_aliases_map_internally():
    assert MINE(est="mic_sc").est == "high_fidelity"
    assert MINE(est="mic_approx").est == "fast"


def test_pstats_fast_shape():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((5, 200))
    mic, tic = pstats(X, est="fast", device="cpu")
    assert mic.shape == (5, 5)
    assert tic.shape == (5, 5)
