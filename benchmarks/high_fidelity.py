import os
import time
import numpy as np

from cumine import MINE, pstats, cstats
from cumine._backend import backend_name


DEVICE = os.environ.get("CUMINE_DEVICE", "auto").lower()


def timeit(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def bench_single_pair():
    print(f"single-pair MINE  est=high_fidelity  requested={DEVICE}")
    print(f"{'n':>8} {'time':>12} {'backend':>10} {'MIC':>10}")
    print("-" * 48)

    for n in [300, 800, 2000, 5000]:
        x = np.linspace(0, 1, n)
        y = np.sin(10 * np.pi * x) + x

        mine = MINE(est="high_fidelity", device=DEVICE)

        dt, _ = timeit(lambda: mine.compute_score(x, y))

        print(
            f"{n:8d} {dt:10.3f}s "
            f"{mine.resolved_device_:>10s} {mine.mic():10.6f}"
        )


def bench_pstats():
    print()
    print(f"pstats  est=high_fidelity  requested={DEVICE}")
    print(f"{'vars':>8} {'samples':>8} {'pairs':>8} {'time':>12} {'backend':>10}")
    print("-" * 60)

    rng = np.random.default_rng(0)

    # Keep these modest for CPU runs; CUDA can handle larger cases via benchmarks/pairwise.py.
    for vars_, samples in [(6, 300), (8, 500), (10, 800)]:
        X = rng.standard_normal((vars_, samples))
        pairs = vars_ * (vars_ - 1) // 2

        dt, _ = timeit(lambda: pstats(X, est="high_fidelity", device=DEVICE))

        print(
            f"{vars_:8d} {samples:8d} {pairs:8d} "
            f"{dt:10.3f}s {backend_name():>10s}"
        )


def bench_cstats():
    print()
    print(f"cstats  est=high_fidelity  requested={DEVICE}")
    print(f"{'xvars':>8} {'yvars':>8} {'samples':>8} {'pairs':>8} {'time':>12} {'backend':>10}")
    print("-" * 72)

    rng = np.random.default_rng(1)

    # Keep these modest for CPU runs; CUDA can handle larger cases via benchmarks/pairwise.py.
    for xvars, yvars, samples in [(5, 4, 300), (8, 6, 500), (10, 8, 800)]:
        X = rng.standard_normal((xvars, samples))
        Y = rng.standard_normal((yvars, samples))
        pairs = xvars * yvars

        dt, _ = timeit(lambda: cstats(X, Y, est="high_fidelity", device=DEVICE))

        print(
            f"{xvars:8d} {yvars:8d} {samples:8d} {pairs:8d} "
            f"{dt:10.3f}s {backend_name():>10s}"
        )


if __name__ == "__main__":
    bench_single_pair()
    bench_pstats()
    bench_cstats()
