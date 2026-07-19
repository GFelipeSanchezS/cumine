# Install cuMINE on Ubuntu

These instructions assume Ubuntu, Python 3.10+, and an NVIDIA GPU if you want native CUDA acceleration. See "Known environment issues" near the end for fixes needed on newer Ubuntu releases (e.g. WSL2 + Ubuntu 26.04).

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
pip install -e ".[dev]"
pytest -qv
```

## 3. Optional CuPy backend

For CUDA 12.x:

```bash
pip install -e ".[gpu,dev]"
CUMINE_DEVICE=cupy python - <<'PY2'
import numpy as np
from cumine import MINE
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
rm -rf build cumine.egg-info
rm -f cumine/_cuda_ext*.so
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
```

Build native CUDA:

```bash
CUMINE_BUILD_CUDA=1 pip install -e ".[dev]" --no-cache-dir
```

For RTX 2080 Ti, you can pin architecture:

```bash
CUMINE_BUILD_CUDA=1 CUMINE_CUDA_ARCH=sm_75 pip install -e ".[dev]" --no-cache-dir
```

## 5. Verify installation

```bash
python - <<'PY'
from cumine import available_backends, native_cuda_available, cupy_available
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
rm -rf build cumine.egg-info
rm -f cumine/_cuda_ext*.so
CUMINE_BUILD_CUDA=1 pip install -e ".[dev]" --no-cache-dir
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

## 8. Known environment issues

### WSL2 + Ubuntu 26.04 (glibc 2.43): CUDA 12.6 fails to compile

Symptom, compiling `cuda_kernels.cu`:

```text
error: exception specification is incompatible with that of previous function "cospi"
error: exception specification is incompatible with that of previous function "sinpi"
error: exception specification is incompatible with that of previous function "rsqrt"
```

Cause: CUDA 12.6's bundled `crt/math_functions.h` (Oct 2024) predates this glibc's own `noexcept`-qualified declarations of `cospi`/`sinpi`/`rsqrt` (GNU extensions in `bits/mathcalls.h`). This is a toolkit-vs-glibc version mismatch, not a GPU/architecture limitation — RTX 30-series (`sm_86`) is fully supported by both 12.6 and 13.x.

Fix: install a newer CUDA toolkit (13.3 confirmed working) from NVIDIA's `wsl-ubuntu` apt repo, which can coexist with an existing 12.6 install (separate `/usr/local/cuda-13.3` directory):

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-3
```

### nvcc rejects gcc-15 (Ubuntu 26.04's default) as a host compiler

Symptom:

```text
error: #error -- unsupported GNU version! gcc versions later than 13 are not supported!
```

Fix: install `g++-13` alongside the system default, and pin nvcc to it with `-ccbin` — either via a wrapper script set as the `NVCC` env var (`setup.py` reads `NVCC` to override the nvcc binary it invokes), or directly if calling nvcc yourself:

```bash
sudo apt-get install -y g++-13
```

```bash
cat > /tmp/nvcc_g13.sh <<'EOF'
#!/bin/bash
exec /usr/local/cuda-13.3/bin/nvcc -ccbin=/usr/bin/g++-13 "$@"
EOF
chmod +x /tmp/nvcc_g13.sh

export CUDA_HOME=/usr/local/cuda-13.3
export NVCC=/tmp/nvcc_g13.sh
CUMINE_BUILD_CUDA=1 CUMINE_CUDA_ARCH=sm_86 pip install -e ".[dev]" --no-cache-dir
```

No `LD_LIBRARY_PATH` changes needed at runtime — the CUDA 13.3 `.deb` registers its lib path with `ldconfig` automatically.

### Building for a new Python version (e.g. Python 3.14)

No source changes are needed — `cuda_ext.cpp` only uses stable, long-standing CPython/NumPy C API calls. `setuptools` picks up the active interpreter's headers automatically, so simply rebuilding inside the target Python's virtualenv (with `setuptools`/`wheel` installed) produces a correctly tagged `_cuda_ext.cpython-<version>-*.so` alongside any older one — no need to delete the old build first.
