# cuMINE architecture

This document explains how `cuMINE` is organized internally so a developer or AI agent can understand the project without reading every source file first.

## Project goal

`cuMINE` computes MINE-style dependency statistics for one pair of variables and for many pairwise/cross-pair variable combinations.

The package exposes two estimator modes:

- `est="high_fidelity"`: the default correctness-oriented estimator. It uses a global equal-frequency y partition and adaptive x-partition dynamic programming. Native CUDA accelerates the expensive dynamic-programming scoring core.
- `est="fast"`: a global equal-frequency estimator optimized for large exploratory pairwise scans. Native CUDA includes batched kernels for `pstats` and `cstats`.

The package is standalone. It does not depend on any external MINE package at runtime.

## Current backend status

| Backend | `fast` | `high_fidelity` | Notes |
|---|---|---|---|
| CPU / NumPy | yes | yes | Reference path and fallback path. |
| CuPy | yes where implemented | runs through common path | CuPy is available as a backend, but native CUDA is the main optimized GPU path. |
| Native CUDA | yes | yes | Compiled extension built from this repository's `.cu` and `.cpp` files. |

Native CUDA currently accelerates:

- `fast` single-pair scoring.
- `fast` batched `pstats` and `cstats`.
- `high_fidelity` single-pair adaptive-DP scoring core.
- `high_fidelity` batched `pstats` and `cstats` adaptive-DP scoring core.

For `high_fidelity`, host-side rank/order and prefix preparation remain CPU-side. The CUDA port accelerates the expensive grid scoring and dynamic-programming portion.

## Top-level files

Recommended release-era layout is described in `DEVELOPMENT.md`. The current important files are:

| File | Purpose |
|---|---|
| `README.md` | Main user-facing overview, install, usage, current benchmark summary, and documentation map. |
| `INSTALL_UBUNTU_22.md` | Tested Ubuntu 22.04 installation instructions. |
| `BENCHMARKS.md` | Current benchmark checkpoint and commands. |
| `RELEASE_CHECKLIST.md` | Release gate checklist. |
| `ARCHITECTURE.md` | This file. Internal design map. |
| `API_REFERENCE.md` | Public API reference. |
| `DEVELOPMENT.md` | Developer workflow and cleanup recommendations. |
| `LIMITATIONS.md` | Platform, numerical, and maintenance limitations. |
| `setup.py` | Editable install, optional native CUDA extension build, CUDA toolkit discovery. |
| `pyproject.toml` | Build-system metadata. |
| `MANIFEST.in` | Source distribution inclusion/exclusion rules. |
| `requirements.txt` | Minimal base dependency list. |

## Python package layout

```text
cumine/
├── __init__.py
├── mine.py
├── _backend.py
├── _kernels.py
├── _cuda_native.py
├── cuda_ext.cpp
└── cuda_kernels.cu
```

### `cumine/__init__.py`

Public import surface. It exports:

- `MINE`
- `pstats`
- `cstats`
- `available_backends`
- `native_cuda_available`
- `cupy_available`

It also defines `__version__`.

### `cumine/mine.py`

Main public API implementation.

Responsibilities:

- Normalize estimator names and aliases.
- Define the `MINE` class.
- Implement `MINE.compute_score()` and statistics accessors: `mic()`, `tic()`, `mas()`, `mev()`, `mcn()`, `mcn_general()`, `gmic()`.
- Implement `pstats()` for all-pairs statistics inside one matrix.
- Implement `cstats()` for cross-pair statistics between two matrices.
- Route `fast` and `high_fidelity` calls to CPU, CuPy/common, or native CUDA paths.
- Handle constant-variable masks so constant rows produce zero off-diagonal associations.

Important routing behavior:

- `MINE.compute_score()` resolves the requested device and calls `compute_characteristic_matrix()` from `_kernels.py`.
- `pstats(..., est="high_fidelity", device="cuda")` calls `batch_high_fidelity_cuda()` from `_kernels.py`.
- `cstats(..., est="high_fidelity", device="cuda")` calls `batch_high_fidelity_cuda()` from `_kernels.py`.
- `pstats(..., est="fast", device="cuda")` calls `cuda_native.batch_mic_tic_fast()`.
- `cstats(..., est="fast", device="cuda")` calls `cuda_native.batch_mic_tic_fast()`.

### `cumine/_backend.py`

Backend/device policy and availability detection.

Responsibilities:

- Detect whether CuPy is importable.
- Detect whether the native CUDA extension `cuMINE._cuda_ext` is importable.
- Implement `resolve_device()` for `cpu`, `cupy`, `cuda`, `gpu`, and `auto`.
- Track the last used backend via `set_last_backend()` and `backend_name()`.
- Provide convenience helpers: `available_backends()`, `native_cuda_available()`, `cupy_available()`, `cuda_native_module()`, `cupy_module()`, `is_gpu()`.

Important policy:

- Explicit API choices should win over environment variables.
- `device="cuda"` means the native compiled extension.
- `device="cupy"` means the CuPy backend.
- `device="gpu"` prefers native CUDA if available, otherwise CuPy.
- `device="auto"` chooses CPU for small workloads and native CUDA for larger pairwise workloads when available.

### `cumine/_kernels.py`

Estimator implementation and Python/CUDA bridge.

Responsibilities:

- Implement stable ranking and equal-frequency binning helpers.
- Implement the CPU/common `fast` characteristic matrix path.
- Implement the CPU/common `high_fidelity` adaptive x-partition path.
- Implement normalized mutual information calculation helpers.
- Call native CUDA functions through `_cuda_native.py` when `device="cuda"`.
- Prepare host-side inputs for high-fidelity CUDA, including rank/order/candidate-cut preparation and batch packing.

High-fidelity estimator concept:

1. Rank/order input samples.
2. Build global equal-frequency y partitions.
3. Generate candidate x cuts.
4. Score intervals.
5. Use dynamic programming to choose an x partition that maximizes normalized mutual information for each grid shape.
6. Return a characteristic matrix `M`.

CUDA acceleration boundary:

- Python/NumPy still prepares ranks/order/candidate cuts and batch inputs.
- Native CUDA computes the expensive adaptive-DP scoring core.

### `cumine/_cuda_native.py`

Thin Python wrapper around the compiled extension.

Responsibilities:

- Import `cuMINE._cuda_ext`.
- Expose native CUDA functions in a stable Python-facing form.
- Provide wrappers such as `norm_mi_cuda`, `batch_mic_tic_fast`, `high_fidelity_scores`, and `batch_high_fidelity_mic_tic`.

This file should stay thin. Algorithmic logic should live in `_kernels.py` or in the native CUDA implementation.

### `cumine/cuda_ext.cpp`

CPython/NumPy extension glue for native CUDA.

Responsibilities:

- Validate Python arguments.
- Convert NumPy arrays to raw pointers and dimensions.
- Allocate output NumPy arrays.
- Call CUDA launcher functions implemented in `cuda_kernels.cu`.
- Expose compiled functions in the `cuMINE._cuda_ext` module.

This file is not the place for estimator design. Keep it focused on argument conversion, memory layout, launch calls, and return values.

### `cumine/cuda_kernels.cu`

Native CUDA kernels and launcher implementations.

Responsibilities:

- Implement fast estimator CUDA kernels.
- Implement batched fast `pstats`/`cstats` kernels.
- Implement high-fidelity adaptive-DP CUDA scoring kernels.
- Implement batched high-fidelity CUDA scoring kernels.
- Provide C++ launcher functions called by `cuda_ext.cpp`.

When editing this file, always rebuild with:

```bash
rm -rf build cuMINE.egg-info
rm -f cumine/_cuda_ext*.so
CUMINE_BUILD_CUDA=1 pip install -e "[dev]" --no-cache-dir
```

## Runtime flow: single-pair `MINE`

```text
User code
  ↓
MINE(...).compute_score(x, y)
  ↓
mine.py resolves estimator and backend
  ↓
_backend.resolve_device(...)
  ↓
_kernels.compute_characteristic_matrix(...)
  ↓
CPU/CuPy/common path OR native CUDA wrapper
  ↓
MINE stores characteristic matrix `_M`
  ↓
mic(), tic(), mas(), mev(), mcn(), etc. read `_M`
```

For `est="high_fidelity", device="cuda"`:

```text
MINE.compute_score
  ↓
_kernels.compute_characteristic_matrix
  ↓
prepare high-fidelity ranked/cut data on host
  ↓
_cuda_native.high_fidelity_scores
  ↓
_cuda_ext.cpp argument conversion
  ↓
cuda_kernels.cu high-fidelity DP kernel
  ↓
return characteristic matrix values to Python
```

## Runtime flow: `pstats`

`pstats(X)` expects `X` shaped as:

```text
(n_variables, n_samples)
```

The output matrices are shaped:

```text
(n_variables, n_variables)
```

For `est="high_fidelity", device="cuda"`:

```text
pstats(X, est="high_fidelity", device="cuda")
  ↓
resolve native CUDA backend
  ↓
_kernels.batch_high_fidelity_cuda(X, X, symmetric=True)
  ↓
prepare batch input on host
  ↓
_cuda_native.batch_high_fidelity_mic_tic
  ↓
_cuda_ext.cpp
  ↓
cuda_kernels.cu batched high-fidelity DP path
  ↓
apply constant-row mask
  ↓
return mic, tic matrices
```

For `est="fast", device="cuda"`:

```text
pstats(X, est="fast", device="cuda")
  ↓
rank rows on host
  ↓
_cuda_native.batch_mic_tic_fast(..., symmetric=True)
  ↓
return mic, tic matrices
```

## Runtime flow: `cstats`

`cstats(X, Y)` expects:

```text
X.shape == (n_x_variables, n_samples)
Y.shape == (n_y_variables, n_samples)
```

The output matrices are shaped:

```text
(n_x_variables, n_y_variables)
```

For `est="high_fidelity", device="cuda"`:

```text
cstats(X, Y, est="high_fidelity", device="cuda")
  ↓
resolve native CUDA backend
  ↓
_kernels.batch_high_fidelity_cuda(X, Y, symmetric=False)
  ↓
prepare batch input on host
  ↓
_cuda_native.batch_high_fidelity_mic_tic
  ↓
_cuda_ext.cpp
  ↓
cuda_kernels.cu batched high-fidelity DP path
  ↓
apply constant-row masks
  ↓
return mic, tic matrices
```

## Characteristic matrix convention

`MINE.get_score()` returns a ragged list of NumPy arrays representing the characteristic matrix search over grid shapes.

Each row corresponds to an x-grid size beginning at 2. Values inside each row correspond to y-grid sizes beginning at 2.

Statistics are derived from this matrix:

- `mic()` returns the maximum matrix value.
- `tic(norm=False)` sums all matrix values.
- `tic(norm=True)` averages matrix values.
- `mas()` compares asymmetry across transposed grid shapes where available.
- `mev()` returns the maximum edge value from the first row/column-like entries.
- `mcn(eps)` returns the minimal grid complexity achieving near-MIC.

## Tests that protect architecture assumptions

Key test files:

| Test file | Purpose |
|---|---|
| `tests/test_device_policy.py` | Device resolution and explicit device behavior. |
| `tests/test_batched_cuda_policy.py` | CUDA pairwise behavior and constant-variable handling. |
| `tests/test_estimator_backend_consistency.py` | CPU/GPU consistency across estimators. |
| `tests/test_estimator_behavior.py` | Basic estimator behavior checks. |
| `tests/test_estimator_statistical_sanity.py` | Prevents high-fidelity degeneracy on random/noisy/constant data. |
| `tests/test_high_fidelity_cuda_path.py` | Regression test proving high-fidelity CUDA calls the native CUDA path. |

## What not to break

Do not regress these guarantees:

- Random independent data must not score as perfect.
- Constant variables must score near zero off-diagonal.
- Explicit `device="cpu"`, `device="cupy"`, and `device="cuda"` must be respected.
- `fast` CUDA batched `pstats`/`cstats` must remain fast.
- `high_fidelity` CUDA must call the native CUDA path, not silently fall back to CPU.
- Source distributions must include `.cu` and `.cpp` files but exclude compiled `.so` artifacts.
