# cumine limitations and support status

This document states what is tested, what is expected to work, and what is not officially supported yet.

## Tested checkpoint

Current confirmed development checkpoint:

```text
Ubuntu 22.04
Python 3.10
CUDA 12.x environment
NVIDIA RTX 2080 Ti class GPU
Native CUDA editable build with CUMINE_BUILD_CUDA=1
```

Current validation checkpoint:

```text
pytest -qv                 -> 36 passed
CUMINE_DEVICE=cuda pytest  -> 36 passed
```

Current performance checkpoint includes real native CUDA acceleration for both `fast` and `high_fidelity` estimators.

## Platform support

| Platform | CPU mode | CuPy mode | Native CUDA extension | Status |
|---|---:|---:|---:|---|
| Ubuntu 22.04 | yes | likely/yes if CuPy installed | yes, tested | Primary tested platform. |
| Ubuntu 20.04 | likely | likely if CUDA/CuPy compatible | expected, not confirmed | Needs clean build validation. |
| Ubuntu 24.04 | likely | likely if CUDA/CuPy compatible | expected, not confirmed | Should be tested next. |
| Other Linux | likely | depends on CuPy/CUDA | possible, not confirmed | Requires supported NVIDIA driver/toolchain. |
| WSL2 Ubuntu on Windows 11 | likely | likely if NVIDIA WSL CUDA works | expected, not confirmed | Preferred Windows CUDA route. |
| Native Windows 11 | likely | likely if CuPy installed | not officially supported | Current build system is Linux-oriented. |
| macOS | likely CPU-only | no NVIDIA CUDA | no | CPU-only expected. |

## Windows status

Native Windows 11 CUDA builds are not officially supported yet.

CPU mode should be possible on Windows if Python and NumPy install correctly.

CuPy mode may work if the user installs a compatible CuPy package and has a supported NVIDIA driver stack.

Native `device="cuda"` currently assumes a Linux-style extension build, including assumptions such as:

- `nvcc` command-line flow compatible with the current `setup.py`.
- Linux-style shared library output.
- `-fPIC` compiler option.
- CUDA library paths like `lib64`.

For Windows users who need native CUDA acceleration, use this recommendation for now:

```text
Windows 11 + WSL2 Ubuntu + NVIDIA CUDA support
```

## Python version support

The package currently declares:

```text
python_requires >= 3.10
```

Tested version in the current checkpoint:

```text
Python 3.10
```

Other Python versions should not be claimed as tested until validated.

## CUDA version support

The current development environment used CUDA 12.x.

Native CUDA builds require:

- NVIDIA driver.
- CUDA toolkit with `nvcc`.
- Compatible host compiler.
- Correct include/library paths.

The project supports optional architecture pinning:

```bash
CUMINE_BUILD_CUDA=1 CUMINE_CUDA_ARCH=sm_75 pip install -e "[dev]" --no-cache-dir
```

For RTX 2080 Ti, `sm_75` is appropriate.

Do not claim broad CUDA Toolkit support until tested on each target toolkit/version.

## CuPy limitations

CuPy is available as a backend where implemented, but the main optimized GPU backend is native CUDA via `device="cuda"`.

Current wording should be:

```text
CuPy backend exists and may run selected paths, but native CUDA is the optimized GPU path for both fast and high_fidelity.
```

Do not imply that CuPy is equally optimized for high-fidelity unless benchmarks prove it.

## High-fidelity implementation boundary

`high_fidelity` native CUDA accelerates the adaptive dynamic-programming scoring core.

The following preparation remains host-side:

- Input conversion to NumPy arrays.
- Stable ranking/order preparation.
- Candidate cut preparation.
- Some batch packing and constant-variable masking.

This means:

- CUDA speedup is real for moderate and large workloads.
- Very small workloads can be slower on CUDA because launch/setup overhead dominates.
- Further optimization is possible by moving more preparation onto the GPU, but that is not required for the current checkpoint.

## Small workload behavior

CUDA may be slower than CPU for small single-pair workloads.

Observed checkpoint example:

```text
high_fidelity single-pair n=300:
  CPU   0.195 s
  CUDA  0.401 s
```

This is expected. CUDA launch/setup overhead dominates tiny jobs.

For larger jobs, CUDA wins substantially.

## Numerical and statistical limitations

`cumine` computes MINE-style estimates. Scores depend on estimator choice, grid budget, sample size, and data distribution.

Important limitations:

- The package should not be represented as a formal drop-in replacement for any specific external MINE implementation unless dedicated compatibility testing is performed.
- `high_fidelity` and `fast` may rank some patterns differently.
- A higher score from `fast` than `high_fidelity` on a specific synthetic pattern is not automatically a bug.
- The current sanity goal is robust behavior, not identical results to another library.

Protected sanity expectations:

- Random independent data should not score near 1.0.
- Constant-variable relationships should score near zero off-diagonal.
- Noise should reduce score relative to a cleaner structured relationship.
- CPU and CUDA results should match within expected tolerance for tested cases.

## API limitations

Current public API:

- `MINE`
- `pstats`
- `cstats`
- `available_backends`
- `native_cuda_available`
- `cupy_available`

Potentially unstable/internal details:

- Functions inside `_backend.py`, `_kernels.py`, `_cuda_native.py` other than the exported availability helpers.
- Exact structure of the ragged characteristic matrix returned by `get_score()`.
- Estimator aliases beyond `high_fidelity` and `fast`.

Recommended stable user-facing estimator names:

```python
est="high_fidelity"
est="fast"
```

## Packaging limitations

Source distributions must include:

```text
cumine/cuda_ext.cpp
cumine/cuda_kernels.cu
```

Source distributions must exclude:

```text
__pycache__/
*.pyc
*.so
build/
dist/
cumine.egg-info/
.pytest_cache/
```

The compiled extension is machine-specific and should not be distributed as part of a source checkpoint.

## Benchmark limitations

Benchmark numbers in `BENCHMARKS.md` are representative development checkpoint numbers, not universal performance guarantees.

They vary with:

- GPU model.
- CPU model.
- CUDA Toolkit version.
- NVIDIA driver version.
- `CUMINE_CUDA_ARCH`.
- Python/NumPy version.
- Thermal/power limits.
- Background system load.

Benchmark tables should be presented as checkpoint evidence, not guaranteed results.

## Release limitations

Before a public release, at minimum:

```bash
pytest -qv
CUMINE_DEVICE=cuda pytest -qv
python tools/characterize_estimators.py
python benchmark_pairwise.py --est fast --device cuda
python benchmark_pairwise.py --est high_fidelity --device cuda
CUMINE_DEVICE=cpu python benchmark_high_fidelity.py
CUMINE_DEVICE=cuda python benchmark_high_fidelity.py
```

Recommended before broad release:

- Clean build from a fresh source checkout.
- Test on Ubuntu 22.04 from scratch.
- Test on Ubuntu 24.04.
- Test WSL2 Ubuntu if Windows users are a target.
- Confirm source distribution includes `.cu`/`.cpp` and excludes compiled artifacts.
- Confirm README commands match actual file locations.

## Current non-goals

Current non-goals unless explicitly prioritized:

- Native Windows CUDA build support.
- Full GPU-side ranking/sorting for high-fidelity.
- Matching another MINE package exactly.
- Supporting Python versions older than 3.10.
- Publishing binary wheels for every CUDA/Python/platform combination.
