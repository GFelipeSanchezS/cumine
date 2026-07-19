# Development Guide

This guide explains how to work on the `cumine` source tree, run tests, build the optional native CUDA extension, and run the project benchmarks.

## Repository layout

```text
cumine/
  cumine/
    __init__.py
    mine.py
    _backend.py
    _kernels.py
    _cuda_native.py
    cuda_ext.cpp
    cuda_kernels.cu

  tests/
    test_batched_cuda_policy.py
    test_device_policy.py
    test_estimator_backend_consistency.py
    test_estimator_behavior.py
    test_estimator_statistical_sanity.py
    test_high_fidelity_cuda_path.py

  benchmarks/
    high_fidelity.py
    pairwise.py

  tools/
    characterize_estimators.py
    validate_correctness.py

  README.md
  API_REFERENCE.md
  ARCHITECTURE.md
  BENCHMARKS.md
  INSTALL_UBUNTU.md
  LIMITATIONS.md
  RELEASE_CHECKLIST.md
  MANIFEST.in
  pyproject.toml
  setup.py
  requirements.txt
```

## Core source files

| File | Purpose |
|---|---|
| `cumine/mine.py` | Public API implementation for `MINE`, `pstats`, and `cstats`. |
| `cumine/_backend.py` | Backend detection, device resolution, and backend status helpers. |
| `cumine/_kernels.py` | CPU/CuPy estimator logic and native CUDA dispatch. |
| `cumine/_cuda_native.py` | Python wrapper around the compiled native CUDA extension. |
| `cumine/cuda_ext.cpp` | CPython/NumPy extension boundary for native CUDA functions. |
| `cumine/cuda_kernels.cu` | Native CUDA kernels for accelerated scoring. |

## Estimators

`cumine` provides two estimators.

| Estimator | Purpose |
|---|---|
| `fast` | Fast equal-frequency grid estimator for large pairwise scans. |
| `high_fidelity` | Adaptive x-partition estimator for stronger structured-dependence characterization. |

## Backends

| Backend | Description |
|---|---|
| `cpu` | NumPy implementation. |
| `cupy` | CuPy-backed path where available. |
| `cuda` | Native compiled CUDA extension. |
| `gpu` | Native CUDA when available, otherwise CuPy. |
| `auto` | Automatic backend selection. |

## Development install

CPU-only editable install:

```bash
pip install -e ".[dev]"
```

Native CUDA editable install:

```bash
CUMINE_BUILD_CUDA=1 pip install -e ".[dev]" --no-cache-dir
```

Rebuild the native CUDA extension after changing any of these files:

```text
cumine/cuda_ext.cpp
cumine/cuda_kernels.cu
setup.py
```

## CUDA environment

In a new shell, set CUDA paths once if needed:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Check the active environment:

```bash
env | grep -iE "CUDA|PATH|LD"
```

Avoid repeatedly prepending the same CUDA paths in one terminal session.

## Test commands

Run the full suite:

```bash
pytest -qv
```

Run the full suite with CUDA requested:

```bash
CUMINE_DEVICE=cuda pytest -qv
```

Run CUDA-path regression tests:

```bash
pytest -qv tests/test_high_fidelity_cuda_path.py
```

Run estimator statistical sanity tests:

```bash
pytest -qv tests/test_estimator_statistical_sanity.py
```

## Benchmark commands

Fast estimator with native CUDA:

```bash
python benchmarks/pairwise.py --est fast --device cuda
```

High-fidelity estimator with native CUDA:

```bash
python benchmarks/pairwise.py --est high_fidelity --device cuda
```

High-fidelity CPU/CUDA comparison:

```bash
CUMINE_DEVICE=cpu python benchmarks/high_fidelity.py
CUMINE_DEVICE=cuda python benchmarks/high_fidelity.py
```

Larger high-fidelity CUDA workload:

```bash
python benchmarks/pairwise.py --est high_fidelity --device cuda --scale large
```

Estimator characterization:

```bash
python tools/characterize_estimators.py
```

Manual correctness validation:

```bash
python tools/validate_correctness.py
```

## Cleaning generated files

For a clean source tree:

```bash
rm -rf build dist cumine.egg-info
rm -f cumine/_cuda_ext*.so
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
```

The native `.so` file is build output and should not be committed or included in a source release.

## Packaging contents

A source package should include:

```text
Python source files
CUDA/C++ source files
benchmarks/
tools/
tests/
Markdown documentation
pyproject.toml
setup.py
MANIFEST.in
requirements.txt
```

A source package should exclude:

```text
compiled shared objects
__pycache__/
*.pyc
build/
dist/
*.egg-info/
.pytest_cache/
```

## Release validation

Run this validation set for a CUDA-capable release environment:

```bash
pytest -qv
CUMINE_DEVICE=cuda pytest -qv
python tools/characterize_estimators.py
python benchmarks/pairwise.py --est fast --device cuda
python benchmarks/pairwise.py --est high_fidelity --device cuda
CUMINE_DEVICE=cpu python benchmarks/high_fidelity.py
CUMINE_DEVICE=cuda python benchmarks/high_fidelity.py
```

Release gate:

```text
All tests pass.
CUDA-specific tests pass.
Statistical sanity checks pass.
Native CUDA acceleration is confirmed by benchmarks.
The source tree contains no compiled artifacts.
Documentation describes the current package state.
```

