"""
Core MINE computation kernels.

Estimators:
  high_fidelity - adaptive x-axis optimization over global equal-frequency
                  y partitions. This is the correctness-oriented estimator.
  fast          - global equal-frequency bins for both variables. This is the
                  fastest estimator and supports native batched CUDA.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._backend import cupy_module, cuda_native_module


def _equifreq_bins(rank, n, k):
    """Map rank array (int64) to k equal-frequency bins (int32)."""
    return np.minimum((rank * k) // n, k - 1).astype(np.int32)


def _rank_stable(a):
    return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.int64)


# ---------------------------------------------------------------------------
# Fast estimator: global equal-frequency bins for both axes
# ---------------------------------------------------------------------------

def _char_matrix_equifreq(x, y, alpha, device):
    n = len(x)
    B = max(int(n ** alpha), 4)

    x_rank = _rank_stable(x)
    y_rank = _rank_stable(y)

    xp = cupy_module() if device == "cupy" else None
    native_cuda = cuda_native_module() if device == "cuda" else None

    max_rx = min(int(np.floor(B / 2)) + 1, n // 2 + 1)
    max_rx = max(max_rx, 2)

    x_cache = {}
    y_cache = {}
    M = []

    for rx in range(2, max_rx + 1):
        max_ry = min(int(np.floor(B / rx)) + 1, n // 2 + 1)
        max_ry = max(max_ry, 2)

        if rx not in x_cache:
            xb = _equifreq_bins(x_rank, n, rx)
            x_cache[rx] = (xb, xp.asarray(xb)) if device == "cupy" else (xb, None)

        row = np.zeros(max_ry - 1, dtype=np.float64)

        for ry in range(2, max_ry + 1):
            if ry not in y_cache:
                yb = _equifreq_bins(y_rank, n, ry)
                y_cache[ry] = (yb, xp.asarray(yb)) if device == "cupy" else (yb, None)

            xb_np, xb_g = x_cache[rx]
            yb_np, yb_g = y_cache[ry]

            if device == "cupy":
                row[ry - 2] = _norm_mi_cupy(xp, xb_g, yb_g, rx, ry, n)
            elif device == "cuda":
                row[ry - 2] = native_cuda.norm_mi_cuda(xb_np, yb_np, rx, ry, n)
            else:
                row[ry - 2] = _norm_mi_np(xb_np, yb_np, rx, ry, n)

        M.append(row)

    return M


# ---------------------------------------------------------------------------
# High-fidelity estimator: adaptive x-axis optimization
# ---------------------------------------------------------------------------

def _candidate_cuts(n, max_candidates):
    """Candidate split positions in x-sorted order, always including 0 and n."""
    max_candidates = int(max(2, min(n, max_candidates)))
    cuts = np.unique(np.round(np.linspace(0, n, max_candidates + 1)).astype(np.int32))
    if cuts[0] != 0:
        cuts = np.r_[0, cuts]
    if cuts[-1] != n:
        cuts = np.r_[cuts, n]
    return cuts.astype(np.int32)


def _interval_score_matrix(yb_ordered, ry, n, cuts):
    """
    Score every candidate x interval [cuts[a], cuts[b]) for a fixed y partition.

    The y-axis partition is global. The x-axis is optimized by dynamic
    programming over contiguous x-sorted intervals. This avoids the degenerate
    behavior of assigning y bins conditionally inside each x column, which can
    manufacture perfect MIC on independent random data.
    """
    K = len(cuts)

    prefix = np.zeros((n + 1, ry), dtype=np.int32)
    prefix[np.arange(1, n + 1), yb_ordered] = 1
    prefix = np.cumsum(prefix, axis=0)
    pc = prefix[cuts]

    py = np.bincount(yb_ordered, minlength=ry).astype(np.float64) / float(n)
    score = np.full((K, K), -np.inf, dtype=np.float64)

    for a in range(K - 1):
        counts = pc[a + 1 :] - pc[a]
        lengths = (cuts[a + 1 :] - cuts[a]).astype(np.float64)

        joint = counts.astype(np.float64) / float(n)
        p_col = lengths[:, None] / float(n)
        expected = p_col * py[None, :]
        nz = counts > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(nz, joint / np.where(expected > 0, expected, 1.0), 1.0)
            vals = np.where(nz, joint * np.log(ratio), 0.0).sum(axis=1)

        score[a, a + 1 :] = vals

    return score


def _max_mi_over_x_partitions(yb_ordered, rx, ry, n, cuts):
    """Dynamic program for best contiguous x partition for a fixed global y partition."""
    if len(cuts) - 1 < rx:
        cuts = np.arange(n + 1, dtype=np.int32)

    K = len(cuts)
    interval_score = _interval_score_matrix(yb_ordered, ry, n, cuts)

    dp = np.full((rx + 1, K), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for k in range(1, rx + 1):
        for b in range(k, K):
            vals = dp[k - 1, :b] + interval_score[:b, b]
            dp[k, b] = np.max(vals)

    mi = float(dp[rx, K - 1])
    denom = np.log(min(rx, ry))
    if denom <= 0 or not np.isfinite(mi):
        return 0.0

    return float(np.clip(mi / denom, 0.0, 1.0))




def _high_fidelity_grid_spec(n, alpha):
    B = max(int(n ** alpha), 4)
    max_rx = min(int(np.floor(B / 2)) + 1, n // 2 + 1)
    max_rx = max(max_rx, 2)

    rx_list = []
    ry_list = []
    for rx in range(2, max_rx + 1):
        max_ry = min(int(np.floor(B / rx)) + 1, n // 2 + 1)
        max_ry = max(max_ry, 2)
        for ry in range(2, max_ry + 1):
            rx_list.append(rx)
            ry_list.append(ry)

    ry_values = np.array(sorted(set(ry_list)), dtype=np.int32)
    ry_to_index = {int(ry): idx for idx, ry in enumerate(ry_values)}
    ry_index = np.array([ry_to_index[int(ry)] for ry in ry_list], dtype=np.int32)

    max_candidates = min(n, max(64, int(2 * B)))
    cuts = _candidate_cuts(n, max_candidates)

    return (
        np.asarray(rx_list, dtype=np.int32),
        np.asarray(ry_list, dtype=np.int32),
        ry_values,
        ry_index,
        cuts,
    )


def _cut_prefix_counts(yb_ordered, ry, cuts):
    n = len(yb_ordered)
    prefix = np.zeros((n + 1, ry), dtype=np.int32)
    prefix[np.arange(1, n + 1), yb_ordered] = 1
    prefix = np.cumsum(prefix, axis=0)
    return np.asarray(prefix[cuts], dtype=np.int32, order="C")


def _high_fidelity_prefix_single(y_rank, order_x, ry_values, cuts):
    n = len(order_x)
    chunks = []
    offsets = []
    offset = 0
    for ry in ry_values:
        ry_int = int(ry)
        yb = _equifreq_bins(y_rank, n, ry_int)
        yb_ordered = yb[order_x].astype(np.int32)
        pc = _cut_prefix_counts(yb_ordered, ry_int, cuts)
        offsets.append(offset)
        flat = pc.ravel()
        chunks.append(flat)
        offset += int(flat.size)

    prefix_flat = np.concatenate(chunks).astype(np.int32, copy=False) if chunks else np.empty(0, dtype=np.int32)
    prefix_offsets = np.asarray(offsets, dtype=np.int32)
    return prefix_flat, prefix_offsets


def _scores_to_matrix(scores, rx_list, ry_list):
    M = []
    idx = 0
    unique_rx = []
    for rx in rx_list:
        rx_i = int(rx)
        if not unique_rx or unique_rx[-1] != rx_i:
            unique_rx.append(rx_i)

    for rx in unique_rx:
        count = int(np.sum(rx_list == rx))
        row = np.asarray(scores[idx: idx + count], dtype=np.float64)
        M.append(row)
        idx += count
    return M


def _char_matrix_high_fidelity_cuda(x, y, alpha, c):
    n = len(x)
    order_x = np.argsort(x, kind="stable")
    y_rank = _rank_stable(y)

    rx_list, ry_list, ry_values, ry_index, cuts = _high_fidelity_grid_spec(n, alpha)
    prefix_flat, prefix_offsets = _high_fidelity_prefix_single(y_rank, order_x, ry_values, cuts)

    native_cuda = cuda_native_module()
    scores = native_cuda.high_fidelity_scores(
        prefix_flat,
        prefix_offsets,
        rx_list,
        ry_list,
        ry_index,
        cuts,
        n,
    )
    return _scores_to_matrix(scores, rx_list, ry_list)

def _char_matrix_high_fidelity(x, y, alpha, c, device):
    """
    Characteristic matrix using a global y partition plus optimized x partitions.

    For device="cuda", the dynamic-programming grid scoring is executed by
    the native CUDA extension. Rank/order preparation remains on the host.
    """
    if device == "cuda":
        return _char_matrix_high_fidelity_cuda(x, y, alpha, c)

    n = len(x)
    B = max(int(n ** alpha), 4)

    order_x = np.argsort(x, kind="stable")
    y_rank = _rank_stable(y)

    # Conservative candidate budget: enough to adapt x cuts, not so many that
    # pure Python becomes unusable. Native CUDA batching remains available for
    # est="fast".
    max_candidates = min(n, max(64, int(2 * B)))
    cuts = _candidate_cuts(n, max_candidates)

    max_rx = min(int(np.floor(B / 2)) + 1, n // 2 + 1)
    max_rx = max(max_rx, 2)

    yb_cache = {}
    M = []

    for rx in range(2, max_rx + 1):
        max_ry = min(int(np.floor(B / rx)) + 1, n // 2 + 1)
        max_ry = max(max_ry, 2)
        row = np.zeros(max_ry - 1, dtype=np.float64)

        for ry in range(2, max_ry + 1):
            if ry not in yb_cache:
                yb = _equifreq_bins(y_rank, n, ry)
                yb_cache[ry] = yb[order_x].astype(np.int32)

            row[ry - 2] = _max_mi_over_x_partitions(yb_cache[ry], rx, ry, n, cuts)

        M.append(row)

    return M


# ---------------------------------------------------------------------------
# Normalized mutual information
# ---------------------------------------------------------------------------

def _norm_mi_np(xb, yb, rx, ry, n):
    joint = np.zeros((rx, ry), dtype=np.float64)
    np.add.at(joint, (xb, yb), 1.0 / n)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px * py
    nz = joint > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_outer = np.where(outer > 0, outer, 1.0)
        log_r = np.where(nz, np.log(np.where(nz, joint / safe_outer, 1.0)), 0.0)
        mi = float(np.sum(np.where(nz, joint * log_r, 0.0)))
    denom = np.log(min(rx, ry))
    return float(np.clip(mi / denom, 0.0, 1.0)) if denom > 0 else 0.0


def _norm_mi_cupy(xp, xb_g, yb_g, rx, ry, n):
    joint_flat = xp.zeros(rx * ry, dtype=xp.float64)
    idx_g = xb_g.astype(xp.int64) * ry + yb_g.astype(xp.int64)
    xp.add.at(joint_flat, idx_g, 1.0 / n)
    joint = joint_flat.reshape(rx, ry)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px * py
    nz = joint > 0
    safe_outer = xp.where(outer > 0, outer, 1.0)
    log_r = xp.where(nz, xp.log(xp.where(nz, joint / safe_outer, 1.0)), 0.0)
    mi = float(xp.sum(xp.where(nz, joint * log_r, 0.0)))
    denom = np.log(min(rx, ry))
    return float(np.clip(mi / denom, 0.0, 1.0)) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def compute_characteristic_matrix(x, y, alpha, c, est, device="cpu"):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length ({len(x)} vs {len(y)})")

    n = len(x)

    if n < 2:
        raise ValueError("x and y must contain at least two samples")

    if np.std(x) == 0 or np.std(y) == 0:
        B = max(int(n ** alpha), 4)
        max_rx = min(int(np.floor(B / 2)) + 1, n // 2 + 1)
        max_rx = max(max_rx, 2)

        M = []
        for rx in range(2, max_rx + 1):
            max_ry = min(int(np.floor(B / rx)) + 1, n // 2 + 1)
            max_ry = max(max_ry, 2)
            M.append(np.zeros(max_ry - 1, dtype=np.float64))

        return M

    if est == "high_fidelity":
        return _char_matrix_high_fidelity(x, y, alpha, c, device)

    if est == "fast":
        return _char_matrix_equifreq(x, y, alpha, device)

    raise ValueError("est must be one of: 'high_fidelity', 'fast'")


def _prepare_chunk_prefix(chunk, y_rank_cache, order_x_cache, ry_values, cuts, n_rys):
    """Host-side rank/cut-prefix construction for one chunk of (i, j) pairs.

    Pure CPU work, no CUDA calls — designed to run in a background thread
    while a previous chunk's GPU kernel is in flight (see batch_high_fidelity_cuda).
    """
    prefix_chunks = []
    prefix_offsets = np.empty((len(chunk), n_rys), dtype=np.int32)
    offset = 0

    for pidx, (i, j) in enumerate(chunk):
        flat, offsets = _high_fidelity_prefix_single(
            y_rank_cache[j],
            order_x_cache[i],
            ry_values,
            cuts,
        )
        prefix_offsets[pidx, :] = offsets + offset
        prefix_chunks.append(flat)
        offset += int(flat.size)

    prefix_flat = np.concatenate(prefix_chunks).astype(np.int32, copy=False)
    return prefix_flat, prefix_offsets


def batch_high_fidelity_cuda(X, Y, alpha=0.6, symmetric=False, max_pairs_per_chunk=32):
    """Batched high_fidelity pstats/cstats using native CUDA for the DP grid core.

    Ranking, x-ordering, and cut-prefix preparation are performed on the host;
    the expensive adaptive x-partition dynamic program is executed in CUDA for
    each pair/grid block.

    The host-side prefix construction for chunk N+1 is prepared in a background
    thread while chunk N's CUDA kernel is running (the native extension releases
    the GIL for the duration of the kernel call), instead of the two phases
    running strictly back-to-back. This overlap is what actually speeds things
    up — it doesn't reduce total CPU or GPU work, it just lets them happen at
    the same time instead of one blocking the other.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays with variables in rows")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of samples")
    if symmetric and X.shape[0] != Y.shape[0]:
        raise ValueError("symmetric=True requires matching variable counts")

    nx, n = X.shape
    ny = Y.shape[0]
    native_cuda = cuda_native_module()

    rx_list, ry_list, ry_values, ry_index, cuts = _high_fidelity_grid_spec(n, alpha)
    n_grids = int(len(rx_list))
    n_rys = int(len(ry_values))

    if symmetric:
        pair_list = [(i, j) for i in range(nx) for j in range(i + 1, ny)]
    else:
        pair_list = [(i, j) for i in range(nx) for j in range(ny)]

    mic = np.zeros((nx, ny), dtype=np.float64)
    tic = np.zeros((nx, ny), dtype=np.float64)
    if symmetric:
        np.fill_diagonal(mic, 1.0)
        np.fill_diagonal(tic, 1.0)

    if not pair_list:
        return mic, tic

    order_x_cache = [np.argsort(X[i], kind="stable") for i in range(nx)]
    y_rank_cache = [_rank_stable(Y[j]) for j in range(ny)]

    max_pairs_per_chunk = int(max(1, max_pairs_per_chunk))
    chunks = [
        pair_list[start: start + max_pairs_per_chunk]
        for start in range(0, len(pair_list), max_pairs_per_chunk)
    ]

    with ThreadPoolExecutor(max_workers=1) as pool:
        next_future = pool.submit(
            _prepare_chunk_prefix, chunks[0], y_rank_cache, order_x_cache, ry_values, cuts, n_rys
        )

        for chunk_idx, chunk in enumerate(chunks):
            prefix_flat, prefix_offsets = next_future.result()

            if chunk_idx + 1 < len(chunks):
                next_future = pool.submit(
                    _prepare_chunk_prefix,
                    chunks[chunk_idx + 1], y_rank_cache, order_x_cache, ry_values, cuts, n_rys,
                )

            # native_cuda releases the GIL for the duration of the kernel call,
            # so the submitted next-chunk prep above genuinely runs concurrently
            # with this GPU call rather than waiting for it.
            mic_pair, tic_pair = native_cuda.batch_high_fidelity_mic_tic(
                prefix_flat,
                prefix_offsets.ravel(),
                rx_list,
                ry_list,
                ry_index,
                cuts,
                len(chunk),
                n_rys,
                n,
            )
            tic_pair = tic_pair / float(n_grids)

            for local_idx, (i, j) in enumerate(chunk):
                mic_val = float(mic_pair[local_idx])
                tic_val = float(tic_pair[local_idx])
                mic[i, j] = mic_val
                tic[i, j] = tic_val
                if symmetric:
                    mic[j, i] = mic_val
                    tic[j, i] = tic_val

    return mic, tic
