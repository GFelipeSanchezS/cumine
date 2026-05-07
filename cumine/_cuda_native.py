"""
Thin Python wrapper around the optional native CUDA extension.

The compiled module is cumine._cuda_ext. It is only present when installed with:
  CUMINE_BUILD_CUDA=1 pip install -e .
"""

from __future__ import annotations

import numpy as np

try:
    from . import _cuda_ext
    _AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on local CUDA build
    _cuda_ext = None
    _IMPORT_ERROR = exc
    _AVAILABLE = False


def is_available() -> bool:
    return bool(_AVAILABLE)


def unavailable_reason() -> str:
    if _AVAILABLE:
        return "available"
    return repr(_IMPORT_ERROR)


def norm_mi_cuda(xb, yb, rx: int, ry: int, n: int) -> float:
    if not _AVAILABLE or _cuda_ext is None:
        raise RuntimeError(
            "Native CUDA extension is not available. "
            "Build with: CUMINE_BUILD_CUDA=1 pip install -e ."
        )

    xb = np.asarray(xb, dtype=np.int32, order="C")
    yb = np.asarray(yb, dtype=np.int32, order="C")
    return float(_cuda_ext.norm_mi(xb, yb, int(rx), int(ry), int(n)))


def batch_mic_tic_fast(ranks_x, ranks_y, alpha: float = 0.6, symmetric: bool = False):
    """
    Native CUDA batched fast-estimator path.

    Parameters
    ----------
    ranks_x, ranks_y:
        int32 rank matrices shaped (n_variables, n_samples). Each row must
        contain ranks 0..n-1 for one variable.
    alpha:
        MINE grid budget exponent.
    symmetric:
        True for pstats(X) where ranks_x and ranks_y refer to the same matrix.
        False for cstats(X, Y).

    Returns
    -------
    mic, tic_norm:
        float64 matrices. TIC is normalized to match pstats/cstats behavior in
        cumine's Python API for normalized TIC values.
    """
    if not _AVAILABLE or _cuda_ext is None:
        raise RuntimeError(
            "Native CUDA extension is not available. "
            "Build with: CUMINE_BUILD_CUDA=1 pip install -e ."
        )

    ranks_x = np.asarray(ranks_x, dtype=np.int32, order="C")
    ranks_y = np.asarray(ranks_y, dtype=np.int32, order="C")

    if ranks_x.ndim != 2 or ranks_y.ndim != 2:
        raise ValueError("rank matrices must be 2D arrays shaped (n_variables, n_samples)")
    if ranks_x.shape[1] != ranks_y.shape[1]:
        raise ValueError("rank matrices must have the same number of samples")

    mic, tic = _cuda_ext.batch_mic_tic_equifreq(
        ranks_x,
        ranks_y,
        float(alpha),
        int(bool(symmetric)),
    )
    return np.asarray(mic, dtype=np.float64), np.asarray(tic, dtype=np.float64)


# Backward-compatible internal alias. Not part of the public API.
def batch_mic_tic_equifreq(ranks_x, ranks_y, alpha: float = 0.6, symmetric: bool = False):
    return batch_mic_tic_fast(ranks_x, ranks_y, alpha=alpha, symmetric=symmetric)


def high_fidelity_scores(prefix_flat, prefix_offsets, rx_list, ry_list, ry_index, cuts, n: int):
    """Native CUDA DP scores for one high_fidelity characteristic matrix."""
    if not _AVAILABLE or _cuda_ext is None:
        raise RuntimeError(
            "Native CUDA extension is not available. "
            "Build with: CUMINE_BUILD_CUDA=1 pip install -e ."
        )

    prefix_flat = np.asarray(prefix_flat, dtype=np.int32, order="C")
    prefix_offsets = np.asarray(prefix_offsets, dtype=np.int32, order="C")
    rx_list = np.asarray(rx_list, dtype=np.int32, order="C")
    ry_list = np.asarray(ry_list, dtype=np.int32, order="C")
    ry_index = np.asarray(ry_index, dtype=np.int32, order="C")
    cuts = np.asarray(cuts, dtype=np.int32, order="C")

    scores = _cuda_ext.high_fidelity_scores(
        prefix_flat,
        prefix_offsets,
        rx_list,
        ry_list,
        ry_index,
        cuts,
        int(n),
    )
    return np.asarray(scores, dtype=np.float64)


def batch_high_fidelity_mic_tic(prefix_flat, prefix_offsets, rx_list, ry_list, ry_index, cuts, n_pairs: int, n_rys: int, n: int):
    """Native CUDA batched high_fidelity MIC/TIC for a chunk of pairs."""
    if not _AVAILABLE or _cuda_ext is None:
        raise RuntimeError(
            "Native CUDA extension is not available. "
            "Build with: CUMINE_BUILD_CUDA=1 pip install -e ."
        )

    prefix_flat = np.asarray(prefix_flat, dtype=np.int32, order="C")
    prefix_offsets = np.asarray(prefix_offsets, dtype=np.int32, order="C")
    rx_list = np.asarray(rx_list, dtype=np.int32, order="C")
    ry_list = np.asarray(ry_list, dtype=np.int32, order="C")
    ry_index = np.asarray(ry_index, dtype=np.int32, order="C")
    cuts = np.asarray(cuts, dtype=np.int32, order="C")

    mic, tic = _cuda_ext.batch_high_fidelity_mic_tic(
        prefix_flat,
        prefix_offsets,
        rx_list,
        ry_list,
        ry_index,
        cuts,
        int(n_pairs),
        int(n_rys),
        int(n),
    )
    return np.asarray(mic, dtype=np.float64), np.asarray(tic, dtype=np.float64)
