"""
cumine - MINE statistics with CPU, CuPy, and optional native CUDA backends.
"""

from .mine import MINE, pstats, cstats
from ._backend import available_backends, native_cuda_available, cupy_available

__version__ = "0.1.0"
__all__ = [
    "MINE",
    "pstats",
    "cstats",
    "available_backends",
    "native_cuda_available",
    "cupy_available",
]
