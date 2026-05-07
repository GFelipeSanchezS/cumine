import numpy as np
from cumine import MINE


def mic(x, y, est, device="cpu"):
    m = MINE(est=est, device=device)
    m.compute_score(x, y)
    return m.mic()


def run(n=800):
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, n)

    cases = {
        "linear": x,
        "quadratic": (x - 0.5) ** 2,
        "sine": np.sin(10 * np.pi * x),
        "sine_plus_trend": np.sin(10 * np.pi * x) + x,
        "high_freq_sine": np.sin(30 * np.pi * x),
        "checker_like": np.sin(14 * np.pi * x) + 0.15 * np.sin(80 * np.pi * x),
        "step": (x > 0.5).astype(float),
        "noisy_sine": np.sin(10 * np.pi * x) + rng.normal(0, 0.4, n),
        "random_independent": rng.uniform(0, 1, n),
    }

    print(f"n={n}")
    print(f"{'case':20s} {'high_fidelity':>15s} {'fast':>15s} {'gap':>15s}")
    print("-" * 70)

    for name, y in cases.items():
        hf = mic(x, y, "high_fidelity", device="cpu")
        fast = mic(x, y, "fast", device="cpu")
        print(f"{name:20s} {hf:15.6f} {fast:15.6f} {hf - fast:15.6f}")


if __name__ == "__main__":
    for n in [300, 800, 2000]:
        run(n)
        print()
