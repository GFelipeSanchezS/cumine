# cuMINE API reference

This document describes the public Python API of `cumine`.

## Imports

```python
from cumine import MINE, pstats, cstats
from cumine import available_backends, native_cuda_available, cupy_available
```

## Version

```python
import cumine
print(cumine.__version__)
```

Current version in the checkpoint: `0.1.0`.

## Estimators

Accepted estimator names:

| Estimator | Meaning |
|---|---|
| `high_fidelity` | Default adaptive-DP estimator. Correctness-oriented and native-CUDA accelerated. |
| `fast` | Global equal-frequency estimator optimized for large exploratory scans. |

Aliases currently accepted by the implementation:

| Alias | Normalized estimator |
|---|---|
| `None` | `high_fidelity` |
| `mic` | `high_fidelity` |
| `default` | `high_fidelity` |
| `hf` | `high_fidelity` |
| `sc` | `high_fidelity` |
| `superclumps` | `high_fidelity` |
| `mic_sc` | `high_fidelity` |
| `approx` | `fast` |
| `equifreq` | `fast` |
| `mic_approx` | `fast` |

Recommended public names are only:

```python
est="high_fidelity"
est="fast"
```

The aliases exist for compatibility and development convenience. New docs and examples should prefer the recommended names.

## Devices and backends

Accepted device names:

| Device | Meaning |
|---|---|
| `cpu` | NumPy CPU backend. |
| `cupy` | CuPy backend where implemented. |
| `cuda` | Native compiled CUDA extension from this repository. |
| `gpu` | Prefer native CUDA if available, otherwise CuPy. |
| `auto` | Choose a backend based on workload and availability. |
| `None` | Treated like `auto`. |

Important distinction:

- `device="cupy"` means the CuPy backend.
- `device="cuda"` means the native compiled extension built from `cuda_ext.cpp` and `cuda_kernels.cu`.

## `MINE`

### Constructor

```python
MINE(alpha=0.6, c=15, est="high_fidelity", device="auto")
```

Parameters:

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `alpha` | float | `0.6` | Controls the grid budget as roughly `B = n ** alpha`. Must be in `(0, 1]`. |
| `c` | int | `15` | Compatibility/configuration parameter. Currently retained in the API. |
| `est` | str | `"high_fidelity"` | Estimator name or accepted alias. |
| `device` | str or None | `"auto"` | Backend selector. |

Raises:

- `ValueError` if `alpha` is not in `(0, 1]`.
- `ValueError` if `est` is unknown.

### `compute_score(x, y)`

```python
mine = MINE(est="high_fidelity", device="cuda")
mine.compute_score(x, y)
```

Computes and stores the characteristic matrix for one pair of variables.

Input requirements:

- `x` and `y` must be array-like.
- After flattening, `x` and `y` must have the same length.
- Values should be numeric and convertible to NumPy arrays.

Return value:

- Returns `None`.
- Stores the computed characteristic matrix internally.

Raises:

- `ValueError` if `x` and `y` have different lengths.
- `RuntimeError` if a requested backend is unavailable, for example native CUDA was requested but the extension was not built.

### `resolved_device_`

After `compute_score()`, this attribute records the actual backend selected:

```python
mine.compute_score(x, y)
print(mine.resolved_device_)
```

Examples:

```text
cpu
cupy
cuda
```

### `is_computed()`

```python
mine.is_computed()
```

Returns `True` if `compute_score()` has been called successfully.

### `get_score()`

```python
M = mine.get_score()
```

Returns the characteristic matrix as a ragged list of NumPy arrays.

Raises:

- `RuntimeError` if called before `compute_score()`.

### `mic()`

```python
value = mine.mic()
```

Returns the maximum information coefficient estimate from the characteristic matrix.

Raises:

- `RuntimeError` if called before `compute_score()`.

### `tic(norm=False)`

```python
total = mine.tic()
mean_value = mine.tic(norm=True)
```

Returns the total information coefficient-style aggregate over the characteristic matrix.

Parameters:

| Parameter | Default | Description |
|---|---:|---|
| `norm` | `False` | If `False`, return the sum. If `True`, return the average over matrix entries. |

Raises:

- `RuntimeError` if called before `compute_score()`.

### `mas()`

```python
value = mine.mas()
```

Returns a maximum asymmetry-style statistic derived from available transposed grid-shape entries.

Raises:

- `RuntimeError` if called before `compute_score()`.

### `mev()`

```python
value = mine.mev()
```

Returns a maximum edge value-style statistic.

Raises:

- `RuntimeError` if called before `compute_score()`.

### `mcn(eps=0)`

```python
value = mine.mcn(eps=0)
```

Returns a minimal cell number-style statistic. It searches for the smallest grid complexity that reaches at least `(1 - eps) * MIC`.

Parameters:

| Parameter | Default | Description |
|---|---:|---|
| `eps` | `0` | Allowed tolerance below MIC. |

Raises:

- `RuntimeError` if called before `compute_score()`.

### `mcn_general()`

```python
value = mine.mcn_general()
```

Compatibility wrapper for `mcn(0)`.

### `gmic(p=-1)`

```python
value = mine.gmic()
```

Currently returns `mic()`. The `p` parameter is retained for API compatibility.

### `repr(mine)`

The representation includes alpha, c, estimator, requested device, resolved/backend display, and whether the object has been computed.

Example:

```python
MINE(alpha=0.6, c=15, est='high_fidelity', device='cuda', backend='cuda', computed)
```

## `pstats`

```python
mic, tic = pstats(X, alpha=0.6, c=15, est="high_fidelity", device="auto")
```

Computes all pairwise dependency statistics between rows of `X`.

Input shape:

```text
X.shape == (n_variables, n_samples)
```

Return values:

```text
mic.shape == (n_variables, n_variables)
tic.shape == (n_variables, n_variables)
```

Output conventions:

- `mic` is symmetric.
- `tic` is symmetric.
- Diagonal values are set to `1.0`.
- Off-diagonal entries involving constant variables are set to `0.0`.

Raises:

- `ValueError` if `X` is not 2D.
- `RuntimeError` if a requested backend is unavailable.

Example:

```python
import numpy as np
from cumine import pstats

X = np.random.randn(10, 500)
mic, tic = pstats(X, est="high_fidelity", device="cuda")
```

Backend notes:

- `pstats(..., est="fast", device="cuda")` uses a native batched CUDA path.
- `pstats(..., est="high_fidelity", device="cuda")` uses a native batched high-fidelity CUDA path.

## `cstats`

```python
mic, tic = cstats(X, Y, alpha=0.6, c=15, est="high_fidelity", device="auto")
```

Computes cross-pair dependency statistics between rows of `X` and rows of `Y`.

Input shapes:

```text
X.shape == (n_x_variables, n_samples)
Y.shape == (n_y_variables, n_samples)
```

Return values:

```text
mic.shape == (n_x_variables, n_y_variables)
tic.shape == (n_x_variables, n_y_variables)
```

Output conventions:

- Entries involving constant rows in either `X` or `Y` are set to `0.0`.
- No diagonal convention is applied because `X` and `Y` are separate matrices.

Raises:

- `ValueError` if `X` or `Y` is not 2D.
- `ValueError` if `X` and `Y` have different sample counts.
- `RuntimeError` if a requested backend is unavailable.

Example:

```python
import numpy as np
from cumine import cstats

X = np.random.randn(10, 500)
Y = np.random.randn(3, 500)
mic, tic = cstats(X, Y, est="high_fidelity", device="cuda")
```

Backend notes:

- `cstats(..., est="fast", device="cuda")` uses a native batched CUDA path.
- `cstats(..., est="high_fidelity", device="cuda")` uses a native batched high-fidelity CUDA path.

## Backend helper functions

### `available_backends()`

```python
from cumine import available_backends
print(available_backends())
```

Returns a list of available backend names. CPU is always included. CuPy and native CUDA appear only if import/build detection succeeds.

Typical examples:

```python
['cpu']
['cpu', 'cupy']
['cpu', 'cuda']
['cpu', 'cupy', 'cuda']
```

### `native_cuda_available()`

```python
from cumine import native_cuda_available
print(native_cuda_available())
```

Returns `True` if the native CUDA extension can be imported.

This requires building with:

```bash
CUMINE_BUILD_CUDA=1 pip install -e ".[dev]" --no-cache-dir
```

### `cupy_available()`

```python
from cumine import cupy_available
print(cupy_available())
```

Returns `True` if CuPy can be imported.

For CUDA 12.x, install with:

```bash
pip install -e ".[gpu,dev]"
```

## Environment variables

| Variable | Purpose |
|---|---|
| `CUMINE_DEVICE` | Default device for scripts/benchmarks when the script reads it. Public API explicit `device=` should be preferred in code. |
| `CUMINE_EST` | Default estimator for scripts/benchmarks when the script reads it. |
| `CUMINE_BUILD_CUDA` | If set to `1`, build the native CUDA extension during install. |
| `CUMINE_CUDA_ARCH` | Optional CUDA architecture pin, for example `sm_75` for RTX 2080 Ti. |
| `CUDA_HOME` | CUDA toolkit root, commonly `/usr/local/cuda`. |
| `NVCC` | Optional explicit path to `nvcc`. |

## Minimal examples

### Single-pair high-fidelity CPU

```python
import numpy as np
from cumine import MINE

x = np.linspace(0, 1, 1000)
y = np.sin(10 * np.pi * x) + x

mine = MINE(est="high_fidelity", device="cpu")
mine.compute_score(x, y)
print(mine.mic())
```

### Single-pair high-fidelity native CUDA

```python
import numpy as np
from cumine import MINE

x = np.linspace(0, 1, 5000)
y = np.sin(10 * np.pi * x) + x

mine = MINE(est="high_fidelity", device="cuda")
mine.compute_score(x, y)
print(mine.mic(), mine.resolved_device_)
```

### Pairwise fast CUDA scan

```python
import numpy as np
from cumine import pstats

X = np.random.randn(100, 5000)
mic, tic = pstats(X, est="fast", device="cuda")
```

### Cross-pair high-fidelity CUDA scan

```python
import numpy as np
from cumine import cstats

X = np.random.randn(10, 800)
Y = np.random.randn(8, 800)
mic, tic = cstats(X, Y, est="high_fidelity", device="cuda")
```
