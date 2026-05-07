# Install cuMINE on Ubuntu 22

These instructions assume Ubuntu 22, Python 3.10, and an NVIDIA GPU if you want native CUDA acceleration.

## 1. Create and activate an environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

If you already have an activated virtualenv, keep using it.

## 2. CPU-only install

```bash
pip install -e "[dev]"
pytest -qv
```

## 3. Optional CuPy backend

For CUDA 12.x:

```bash
pip install -e "[gpu,dev]"
CUMINE_DEVICE=cupy python - <<'PY2'
import numpy as np
from cuMINE import MINE
x = np.linspace(0, 1, 300)
y = x ** 2
m = MINE(est='fast', device='cupy')
m.compute_score(x, y)
print(m.resolved_device_, m.mic())
PY2
```

CuPy is CUDA-backed, but the main optimized GPU path in this project is the native `device="cuda"` extension.

## 4. Native CUDA backend

Check that the driver and CUDA compiler are visible:

```bash
nvidia-smi
nvcc --version
```

Set CUDA environment variables if they are not already set:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Do not keep re-exporting these in the same terminal. Check the current values with:

```bash
env | grep -iE "CUDA|PATH|LD"
```

Clean stale build products:

```bash
rm -rf build cuMINE.egg-info
rm -f cuMINE/_cuda_ext*.so
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
```

Build native CUDA:

```bash
CUMINE_BUILD_CUDA=1 pip install -e "[dev]" --no-cache-dir
```

For RTX 2080 Ti, you can pin architecture:

```bash
CUMINE_BUILD_CUDA=1 CUMINE_CUDA_ARCH=sm_75 pip install -e "[dev]" --no-cache-dir
```

## 5. Verify installation

```bash
python - <<'PY'
from cuMINE import available_backends, native_cuda_available, cupy_available
print("available_backends:", available_backends())
print("native_cuda_available:", native_cuda_available())
print("cupy_available:", cupy_available())
PY
```

Expected if native CUDA built correctly:

```text
native_cuda_available: True
```

Run tests:

```bash
pytest -qv
CUMINE_DEVICE=cuda pytest -qv
```

Current green checkpoint:

```text
36 passed
36 passed with CUMINE_DEVICE=cuda
```

## 6. Run benchmarks

High-fidelity single-pair and small pairwise benchmark:

```bash
CUMINE_DEVICE=cpu python benchmarks/high_fidelity.py
CUMINE_DEVICE=cuda python benchmarks/high_fidelity.py
```

Pairwise/cross-pair benchmark:

```bash
python benchmarks/pairwise.py --est fast --device cpu
python benchmarks/pairwise.py --est fast --device cuda

python benchmarks/pairwise.py --est high_fidelity --device cpu
python benchmarks/pairwise.py --est high_fidelity --device cuda
python benchmarks/pairwise.py --est high_fidelity --device cuda --scale large
```

Characterization:

```bash
python tools/characterize_estimators.py
```

## 7. Troubleshooting

### `device='cuda' requested, but the native CUDA extension is not built`

Rebuild with:

```bash
rm -rf build cuMINE.egg-info
rm -f cuMINE/_cuda_ext*.so
CUMINE_BUILD_CUDA=1 pip install -e "[dev]" --no-cache-dir
```

### `nvcc` not found

Check:

```bash
which nvcc
nvcc --version
echo "$CUDA_HOME"
```

Then set:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

### Repeated CUDA paths in `PATH` or `LD_LIBRARY_PATH`

This usually happens after exporting repeatedly in the same terminal. It is usually harmless. Open a fresh shell or clean the variable manually if it bothers you.
