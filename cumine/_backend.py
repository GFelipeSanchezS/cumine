"""
Backend/device policy for cumine.

Public device names:
  auto  - choose a recommended backend for the workload
  cpu   - NumPy CPU backend
  cupy  - CuPy CUDA-backed GPU backend
  cuda  - native compiled CUDA extension backend
  gpu   - prefer native CUDA if available, otherwise CuPy

Important policy:
  Explicit Python API choices always win.

Examples:
  MINE(device="cpu")   must use CPU
  MINE(device="cupy")  must use CuPy
  MINE(device="cuda")  must use native CUDA

The CUMINE_DEVICE environment variable is intended for command-line scripts
such as benchmark scripts. Library calls should pass device=... explicitly.
"""

from __future__ import annotations

from typing import Literal

Device = Literal["auto", "cpu", "cupy", "cuda", "gpu"]

_LAST_BACKEND = None

try:
    import cupy as _cupy
    _CUPY_AVAILABLE = True
except Exception:
    _cupy = None
    _CUPY_AVAILABLE = False

try:
    from . import _cuda_native as _cuda_native
    _NATIVE_CUDA_AVAILABLE = _cuda_native.is_available()
except Exception:
    _cuda_native = None
    _NATIVE_CUDA_AVAILABLE = False


def cupy_available() -> bool:
    return bool(_CUPY_AVAILABLE)


def native_cuda_available() -> bool:
    return bool(_NATIVE_CUDA_AVAILABLE)


def cupy_module():
    if not _CUPY_AVAILABLE:
        raise RuntimeError("CuPy backend requested, but CuPy is not available")
    return _cupy


def cuda_native_module():
    if not _NATIVE_CUDA_AVAILABLE or _cuda_native is None:
        raise RuntimeError(
            "Native CUDA backend requested, but cumine._cuda_ext is not built. "
            "Build it with: CUMINE_BUILD_CUDA=1 pip install -e ."
        )
    return _cuda_native


def available_backends() -> list[str]:
    backends = ["cpu"]
    if _CUPY_AVAILABLE:
        backends.append("cupy")
    if _NATIVE_CUDA_AVAILABLE:
        backends.append("cuda")
    return backends


def _normalize_device(device: str | None) -> str:
    if device is None:
        device = "auto"

    d = str(device).strip().lower()

    aliases = {
        "numpy": "cpu",
        "np": "cpu",
        "cuda-native": "cuda",
        "native-cuda": "cuda",
    }
    d = aliases.get(d, d)

    if d not in {"auto", "cpu", "cupy", "cuda", "gpu"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cupy', 'cuda', 'gpu'")

    return d


def resolve_device(
    requested: str | None,
    *,
    n_samples: int | None = None,
    workload: str = "single",
    n_pairs: int | None = None,
) -> str:
    """
    Resolve a user-requested device into a concrete backend.

    v0.1 policy:
      - explicit cpu/cupy/cuda/gpu always wins
      - auto uses native CUDA for large single-pair jobs if available
      - auto uses native CUDA for large all-pairs jobs if available
      - auto uses CPU for small jobs
      - gpu means native CUDA if available, otherwise CuPy
    """

    d = _normalize_device(requested)

    if d == "cpu":
        resolved = "cpu"

    elif d == "cupy":
        if not _CUPY_AVAILABLE:
            raise RuntimeError("device='cupy' requested, but CuPy is not available")
        resolved = "cupy"

    elif d == "cuda":
        if not _NATIVE_CUDA_AVAILABLE:
            raise RuntimeError(
                "device='cuda' requested, but the native CUDA extension is not built. "
                "Run: CUMINE_BUILD_CUDA=1 pip install -e ."
            )
        resolved = "cuda"

    elif d == "gpu":
        if _NATIVE_CUDA_AVAILABLE:
            resolved = "cuda"
        elif _CUPY_AVAILABLE:
            resolved = "cupy"
        else:
            raise RuntimeError("device='gpu' requested, but no GPU backend is available")

    else:  # auto
        if workload in {"pstats", "cstats", "pairwise", "cross"}:
            # Pairwise GPU launch overhead is worth it only when there are enough
            # pairs or enough samples. Explicit device='cuda' can still force it.
            if _NATIVE_CUDA_AVAILABLE and (
                (n_pairs is not None and n_pairs >= 128)
                or (n_samples is not None and n_samples >= 10_000)
            ):
                resolved = "cuda"
            else:
                resolved = "cpu"
        elif n_samples is not None and n_samples >= 10_000 and _NATIVE_CUDA_AVAILABLE:
            resolved = "cuda"
        elif n_samples is not None and n_samples >= 50_000 and _CUPY_AVAILABLE:
            resolved = "cupy"
        else:
            resolved = "cpu"

    set_last_backend(resolved)
    return resolved


def set_last_backend(name: str) -> None:
    global _LAST_BACKEND
    _LAST_BACKEND = name


def backend_name() -> str:
    """Return the most recently used backend."""
    if _LAST_BACKEND:
        return _LAST_BACKEND
    return "cpu"


def is_gpu() -> bool:
    return backend_name() in {"cupy", "cuda"}
