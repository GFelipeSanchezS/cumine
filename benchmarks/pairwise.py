import argparse
import os
import time

import numpy as np

from cumine import pstats, cstats
from cumine._backend import backend_name


def timeit(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def pstats_cases(est, scale):
    if est == "fast":
        return [
            (10, 1000),
            (25, 2000),
            (50, 5000),
            (100, 5000),
        ]

    if scale == "large":
        return [
            (6, 300),
            (8, 500),
            (10, 800),
            (16, 1000),
            (25, 1500),
        ]

    return [
        (6, 300),
        (8, 500),
        (10, 800),
    ]


def cstats_cases(est, scale):
    if est == "fast":
        return [
            (10, 8, 1000),
            (25, 20, 2000),
            (50, 40, 5000),
        ]

    if scale == "large":
        return [
            (5, 4, 300),
            (8, 6, 500),
            (10, 8, 800),
            (16, 12, 1000),
            (25, 20, 1500),
        ]

    return [
        (5, 4, 300),
        (8, 6, 500),
        (10, 8, 800),
    ]


def bench_pstats(est, device, scale):
    print(f"pstats  est={est}  requested={device}  scale={scale}")
    print(f"{'vars':>8} {'samples':>8} {'pairs':>8} {'time':>12} {'backend':>10}")
    print("-" * 62)

    rng = np.random.default_rng(0)

    for vars_, samples in pstats_cases(est, scale):
        X = rng.standard_normal((vars_, samples))
        pairs = vars_ * (vars_ - 1) // 2

        dt, _ = timeit(lambda: pstats(X, est=est, device=device))

        print(
            f"{vars_:8d} {samples:8d} {pairs:8d} "
            f"{dt:10.3f}s {backend_name():>10s}"
        )


def bench_cstats(est, device, scale):
    print()
    print(f"cstats  est={est}  requested={device}  scale={scale}")
    print(f"{'xvars':>8} {'yvars':>8} {'samples':>8} {'pairs':>8} {'time':>12} {'backend':>10}")
    print("-" * 74)

    rng = np.random.default_rng(1)

    for xvars, yvars, samples in cstats_cases(est, scale):
        X = rng.standard_normal((xvars, samples))
        Y = rng.standard_normal((yvars, samples))
        pairs = xvars * yvars

        dt, _ = timeit(lambda: cstats(X, Y, est=est, device=device))

        print(
            f"{xvars:8d} {yvars:8d} {samples:8d} {pairs:8d} "
            f"{dt:10.3f}s {backend_name():>10s}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--est",
        choices=["fast", "high_fidelity"],
        default=os.environ.get("CUMINE_EST", "fast"),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("CUMINE_DEVICE", "auto"),
    )
    parser.add_argument(
        "--scale",
        choices=["standard", "large"],
        default="standard",
    )
    args = parser.parse_args()

    bench_pstats(args.est, args.device, args.scale)
    bench_cstats(args.est, args.device, args.scale)


if __name__ == "__main__":
    main()
