# cumine

`cumine` is a Python implementation of MINE-style dependency statistics with CPU, CuPy, and optional native CUDA backends.

Current version: `0.1.0`.

## Documentation map

Start here if you are handing the project to another developer or AI agent:

- [ARCHITECTURE.md](ARCHITECTURE.md) — internal design, file responsibilities, estimator/backend flow, and CUDA integration boundaries.
- [API_REFERENCE.md](API_REFERENCE.md) — public Python API for `MINE`, `pstats`, `cstats`, backend helpers, parameters, return shapes, and examples.
- [DEVELOPMENT.md](DEVELOPMENT.md) — development workflow, build/test commands, CUDA debugging notes, benchmark workflow, and recommended repository cleanup.
- [LIMITATIONS.md](LIMITATIONS.md) — tested platforms, unsupported/partially supported cases, numerical caveats, and release constraints.
- [BENCHMARKS.md](BENCHMARKS.md) — current CPU vs native CUDA benchmark checkpoint.
- [INSTALL_UBUNTU_22.md](INSTALL_UBUNTU_22.md) — tested Ubuntu 22.04 installation path.
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) — validation gates before tagging or publishing a release.

## What is included

Backends:

- `cpu`: NumPy CPU backend.
- `cupy`: CuPy backend, CUDA-backed through CuPy where implemented.
- `cuda`: cumine's native compiled CUDA extension, built from this repository's `.cu` and `.cpp` sources.
- `gpu`: prefer native CUDA if available, otherwise CuPy.
- `auto`: choose a backend based on workload size and available backends.

Estimators:

- `est="high_fidelity"` — default estimator. Uses a global equal-frequency y partition and adaptive x-partition dynamic programming. The native CUDA backend accelerates the expensive DP scoring core.
- `est="fast"` — global equal-frequency estimator optimized for large batched exploratory scans.

`cumine` does not depend on any external MINE package at runtime.

## Backend wording

CuPy already uses CUDA underneath, so `device="cupy"` means CuPy CUDA-backed execution where the estimator path uses CuPy kernels.

`device="cuda"` means cumine's own compiled CUDA extension. This is the main optimized GPU path for both `fast` and `high_fidelity`.

For `high_fidelity`, the native CUDA path accelerates adaptive DP scoring. Rank/order and prefix preparation are still performed on the host.

## Install

CPU-only editable install:

```bash
pip install -e "[dev]"
pytest -qv
```

CuPy backend for CUDA 12.x:

```bash
pip install -e "[gpu,dev]"
```

Native CUDA extension backend:

```bash
# Only needed once per shell/session if these are not already configured.
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

rm -rf build cumine.egg-info
rm -f cumine/_cuda_ext*.so
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

CUMINE_BUILD_CUDA=1 pip install -e "[dev]" --no-cache-dir
pytest -qv
CUMINE_DEVICE=cuda pytest -qv
```

Native CUDA requires:

- NVIDIA driver.
- CUDA toolkit with `nvcc`.
- `CUDA_HOME` set correctly, or `nvcc` available on `PATH`.

Check:

```bash
nvidia-smi
nvcc --version
python - <<'PY'
from cumine import available_backends, native_cuda_available, cupy_available
print("available_backends:", available_backends())
print("native_cuda_available:", native_cuda_available())
print("cupy_available:", cupy_available())
PY
```

Optional architecture pinning:

```bash
CUMINE_BUILD_CUDA=1 CUMINE_CUDA_ARCH=sm_75 pip install -e "[dev]" --no-cache-dir
```

For RTX 2080 Ti, `sm_75` is appropriate.

## Usage

```python
import numpy as np
from cumine import MINE

x = np.linspace(0, 1, 1000)
y = np.sin(10 * np.pi * x) + x

mine = MINE(alpha=0.6, c=15, est="high_fidelity", device="auto")
mine.compute_score(x, y)

print(mine.mic())
print(mine.tic())
print(mine.resolved_device_)
```

Explicit backend choice:

```python
MINE(device="cpu")    # NumPy CPU
MINE(device="cupy")   # CuPy backend where implemented
MINE(device="cuda")   # native compiled CUDA extension
MINE(device="gpu")    # native CUDA if available, otherwise CuPy
MINE(device="auto")   # recommended default
```

Estimator choice:

```python
MINE(est="high_fidelity")  # default, adaptive DP estimator
MINE(est="fast")           # global equal-frequency estimator
```

Pairwise statistics:

```python
import numpy as np
from cumine import pstats, cstats

X = np.random.randn(10, 500)
mic, tic = pstats(X, est="high_fidelity", device="cuda")

Y = np.random.randn(3, 500)
mic_xy, tic_xy = cstats(X, Y, est="high_fidelity", device="cuda")
```

Large exploratory scans:

```python
mic, tic = pstats(X, est="fast", device="cuda")
mic_xy, tic_xy = cstats(X, Y, est="fast", device="cuda")
```

## Estimator behavior

### `est="high_fidelity"`

Default estimator. It uses global equal-frequency y bins and optimizes x partitions with dynamic programming over candidate cuts.

Backend support:

| Call | CPU | CuPy | Native CUDA |
|---|---:|---:|---:|
| `MINE.compute_score()` | yes | runs through common path | yes, CUDA DP core |
| `pstats()` | yes | runs through common path | yes, batched CUDA DP core |
| `cstats()` | yes | runs through common path | yes, batched CUDA DP core |

The native CUDA implementation accelerates the DP scoring core. Host-side rank/order and prefix preparation remain CPU-side.

### `est="fast"`

Fast estimator using global equal-frequency binning for both variables. It is designed for large exploratory dependency screens.

Backend support:

| Call | CPU | CuPy | Native CUDA |
|---|---:|---:|---:|
| `MINE.compute_score()` | yes | yes | yes |
| `pstats()` | yes | looped path | yes, batched CUDA |
| `cstats()` | yes | looped path | yes, batched CUDA |

## Backend selection

| Device | Meaning |
|---|---|
| `cpu` | NumPy CPU backend |
| `cupy` | CuPy backend where implemented |
| `cuda` | Native compiled CUDA extension backend |
| `gpu` | Prefer native CUDA if available, otherwise CuPy |
| `auto` | Choose a backend based on workload size and estimator |

Explicit Python API choices win over environment variables. The `CUMINE_DEVICE` environment variable is intended mainly for scripts and benchmarks.

## Tests

```bash
pytest -qv
CUMINE_DEVICE=cuda pytest -qv
```

Current green checkpoint:

```text
36 passed
36 passed with CUMINE_DEVICE=cuda
```

The suite includes a regression test that verifies high-fidelity CUDA calls the native CUDA path instead of silently falling back to CPU.

## Benchmarks

Single-pair high-fidelity benchmark:

```bash
CUMINE_DEVICE=cpu python benchmark_high_fidelity.py
CUMINE_DEVICE=cuda python benchmark_high_fidelity.py
```

Pairwise/cross-pair benchmark:

```bash
python benchmark_pairwise.py --est fast --device cpu
python benchmark_pairwise.py --est fast --device cuda

python benchmark_pairwise.py --est high_fidelity --device cpu
python benchmark_pairwise.py --est high_fidelity --device cuda
python benchmark_pairwise.py --est high_fidelity --device cuda --scale large
```

Representative RTX 2080 Ti checkpoint numbers:

| Workload | CPU | Native CUDA | Approx. speedup |
|---|---:|---:|---:|
| `fast` `pstats`, 100 vars × 5000 samples, 4950 pairs | 373.055 s | 2.289 s | 163× |
| `fast` `cstats`, 50×40 vars × 5000 samples, 2000 pairs | 147.843 s | 0.951 s | 155× |
| `high_fidelity` `MINE`, n=5000 | 24.162 s | 0.990 s | 24× |
| `high_fidelity` `pstats`, 10 vars × 800 samples, 45 pairs | 39.739 s | 0.491 s | 81× |
| `high_fidelity` `cstats`, 10×8 vars × 800 samples, 80 pairs | 70.909 s | 0.853 s | 83× |

Benchmark results vary by GPU, CPU, CUDA toolkit, driver, and selected `CUMINE_CUDA_ARCH`.

## Characterization

Run:

```bash
python tools/characterize_estimators.py
```

Sanity expectations:

- Linear/quadratic/step relationships should score high.
- Random independent data should stay low and generally decrease as sample size grows.
- Noisy relationships should score below cleaner structured relationships.
- `high_fidelity` and `fast` may disagree on some structures because they use different approximations.

## Current v0.1 release scope

Included:

- CPU backend.
- CuPy backend where implemented.
- Native CUDA extension backend.
- `high_fidelity` default estimator.
- `fast` estimator.
- Native CUDA single-pair and batched paths for `high_fidelity`.
- Native batched CUDA paths for `fast` `pstats/cstats`.
- Backend consistency and statistical sanity tests.
- Regression tests to ensure high-fidelity CUDA calls native CUDA.

Before release:

- Run the release checklist in `RELEASE_CHECKLIST.md`.
- Confirm source distributions include CUDA sources and exclude compiled artifacts.
- Decide license and package publishing target.
