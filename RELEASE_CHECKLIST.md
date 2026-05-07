# cumine v0.1 release checklist

Use this before tagging or publishing a release.

## Clean tree

Do not include generated artifacts in source releases:

```bash
rm -rf build dist cumine.egg-info .pytest_cache
rm -f cumine/_cuda_ext*.so
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
```

## CPU validation

```bash
pip install -e ".[dev]" --no-cache-dir
pytest -qv
python tools/characterize_estimators.py
python benchmarks/pairwise.py --est fast --device cpu
python benchmarks/pairwise.py --est high_fidelity --device cpu
```

Expected:

- Test suite passes.
- Random independent characterization stays low.
- Constant-variable tests pass.
- Noisy relationships score below cleaner structured relationships.

## Native CUDA validation

```bash
rm -rf build cumine.egg-info
rm -f cumine/_cuda_ext*.so
CUMINE_BUILD_CUDA=1 pip install -e ".[dev]" --no-cache-dir

pytest -qv
CUMINE_DEVICE=cuda pytest -qv

CUMINE_DEVICE=cpu python benchmarks/high_fidelity.py
CUMINE_DEVICE=cuda python benchmarks/high_fidelity.py

python benchmarks/pairwise.py --est fast --device cuda
python benchmarks/pairwise.py --est high_fidelity --device cuda
python benchmarks/pairwise.py --est high_fidelity --device cuda --scale large
```

Expected:

- Full test suite passes under default device.
- Full test suite passes under `CUMINE_DEVICE=cuda`.
- `tests/test_high_fidelity_cuda_path.py` confirms the high-fidelity CUDA path calls native CUDA.
- `fast` CUDA pairwise benchmarks remain much faster than CPU for large workloads.
- `high_fidelity` CUDA benchmarks show meaningful speedup over CPU for non-trivial workloads.

## Packaging checks

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Confirm source distribution includes:

- `cumine/cuda_ext.cpp`
- `cumine/cuda_kernels.cu`
- `cumine/_cuda_native.py`
- `cumine/_kernels.py`
- tests
- benchmark scripts
- docs

Confirm source distribution excludes:

- compiled `.so` files
- `__pycache__`
- `.pyc`
- `.pytest_cache`
- `build/`
- `cumine.egg-info/`

## Metadata decisions before public release

- Decide and add a license.
- Decide package name availability and publishing target.
- Add project URLs if publishing to PyPI.
- Confirm minimum Python version.
- Confirm CUDA toolkit support statement.
- Confirm whether CuPy is optional documentation-only or part of supported release surface.

## Release gate

Do not tag release if any of these are false:

- CPU tests pass.
- CUDA tests pass on a CUDA machine.
- High-fidelity statistical sanity passes.
- High-fidelity CUDA path regression tests pass.
- Benchmarks show real native CUDA acceleration for high-fidelity.
- Native CUDA extension builds from a clean tree.
