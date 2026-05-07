"""
Public API for cumine.
"""

from __future__ import annotations

import numpy as np

from ._backend import resolve_device, set_last_backend
from ._kernels import compute_characteristic_matrix


_VALID_ESTS = {"high_fidelity", "fast"}
_EST_ALIASES = {
    "mic": "high_fidelity",
    "default": "high_fidelity",
    "hf": "high_fidelity",
    "sc": "high_fidelity",
    "superclumps": "high_fidelity",
    "mic_sc": "high_fidelity",
    "approx": "fast",
    "equifreq": "fast",
    "mic_approx": "fast",
}


def _normalize_est(est):
    if est is None:
        return "high_fidelity"
    e = str(est).strip().lower()
    e = _EST_ALIASES.get(e, e)
    if e not in _VALID_ESTS:
        raise ValueError("est must be one of: 'high_fidelity', 'fast'")
    return e


def _auto_requested(device) -> bool:
    return device is None or str(device).strip().lower() == "auto"


def _rank_stable_int32(a):
    return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.int32)


def _rank_matrix_rows(X):
    X = np.asarray(X, dtype=np.float64)
    ranks = np.empty(X.shape, dtype=np.int32)
    for i in range(X.shape[0]):
        ranks[i] = _rank_stable_int32(X[i])
    return ranks


def _constant_rows(X):
    return np.std(np.asarray(X, dtype=np.float64), axis=1) == 0


def _apply_pstats_constant_mask(mic, tic, const_mask):
    const_mask = np.asarray(const_mask, dtype=bool)
    if not const_mask.any():
        return mic, tic
    n_vars = mic.shape[0]
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and (const_mask[i] or const_mask[j]):
                mic[i, j] = 0.0
                tic[i, j] = 0.0
    np.fill_diagonal(mic, 1.0)
    np.fill_diagonal(tic, 1.0)
    return mic, tic


def _apply_cstats_constant_mask(mic, tic, const_x, const_y):
    const_x = np.asarray(const_x, dtype=bool)
    const_y = np.asarray(const_y, dtype=bool)
    if not const_x.any() and not const_y.any():
        return mic, tic
    bad = const_x[:, None] | const_y[None, :]
    mic[bad] = 0.0
    tic[bad] = 0.0
    return mic, tic


class MINE:
    def __init__(self, alpha=0.6, c=15, est="high_fidelity", device="auto"):
        if not (0 < float(alpha) <= 1):
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = float(alpha)
        self.c = c
        self.est = _normalize_est(est)
        self.device = device
        self.resolved_device_ = None
        self._M = None

    def compute_score(self, x, y):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if x_arr.ravel().shape[0] != y_arr.ravel().shape[0]:
            raise ValueError(
                f"x and y must have the same length ({x_arr.ravel().shape[0]} vs {y_arr.ravel().shape[0]})"
            )

        self.resolved_device_ = resolve_device(
            self.device,
            n_samples=x_arr.ravel().shape[0],
            workload="single",
        )
        self._M = compute_characteristic_matrix(
            x_arr,
            y_arr,
            self.alpha,
            self.c,
            self.est,
            device=self.resolved_device_,
        )
        return None

    def is_computed(self):
        return self._M is not None

    def _require_computed(self):
        if self._M is None:
            raise RuntimeError("compute_score(x, y) must be called before reading statistics")

    def get_score(self):
        self._require_computed()
        return self._M

    def mic(self):
        self._require_computed()
        if not self._M:
            return 0.0
        return float(max((np.max(row) if len(row) else 0.0) for row in self._M))

    def tic(self, norm=False):
        self._require_computed()
        vals = [float(v) for row in self._M for v in row]
        if not vals:
            return 0.0
        total = float(np.sum(vals))
        return total / len(vals) if norm else total

    def mas(self):
        self._require_computed()
        best = 0.0
        for i, row in enumerate(self._M):
            rx = i + 2
            for j, val in enumerate(row):
                ry = j + 2
                i2 = ry - 2
                j2 = rx - 2
                if 0 <= i2 < len(self._M) and 0 <= j2 < len(self._M[i2]):
                    best = max(best, abs(float(val) - float(self._M[i2][j2])))
        return float(best)

    def mev(self):
        self._require_computed()
        if not self._M:
            return 0.0
        vals = []
        vals.extend(float(v) for v in self._M[0])
        for row in self._M:
            if len(row):
                vals.append(float(row[0]))
        return float(max(vals)) if vals else 0.0

    def mcn(self, eps=0):
        self._require_computed()
        target = (1.0 - float(eps)) * self.mic()
        best = None
        for i, row in enumerate(self._M):
            rx = i + 2
            for j, val in enumerate(row):
                ry = j + 2
                if float(val) >= target:
                    score = np.log2(rx * ry)
                    best = score if best is None else min(best, score)
        return float(best) if best is not None else 0.0

    def mcn_general(self):
        return self.mcn(0)

    def gmic(self, p=-1):
        self._require_computed()
        return self.mic()

    def __repr__(self):
        scored = "computed" if self.is_computed() else "not computed"
        backend = self.resolved_device_ or self.device
        return (
            f"MINE(alpha={self.alpha}, c={self.c}, est='{self.est}', "
            f"device='{self.device}', backend='{backend}', {scored})"
        )


def _pstats_loop(X, alpha=0.6, c=15, est="high_fidelity", device="cpu"):
    est = _normalize_est(est)
    n_vars = X.shape[0]
    mic = np.eye(n_vars, dtype=np.float64)
    tic = np.eye(n_vars, dtype=np.float64)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mine = MINE(alpha=alpha, c=c, est=est, device=device)
            mine.compute_score(X[i], X[j])
            mic_ij = mine.mic()
            tic_ij = mine.tic(norm=True)
            mic[i, j] = mic[j, i] = mic_ij
            tic[i, j] = tic[j, i] = tic_ij

    return mic, tic


def _cstats_loop(X, Y, alpha=0.6, c=15, est="high_fidelity", device="cpu"):
    est = _normalize_est(est)
    mic = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)
    tic = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            mine = MINE(alpha=alpha, c=c, est=est, device=device)
            mine.compute_score(X[i], Y[j])
            mic[i, j] = mine.mic()
            tic[i, j] = mine.tic(norm=True)

    return mic, tic


def pstats(X, alpha=0.6, c=15, est="high_fidelity", device="auto"):
    est = _normalize_est(est)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with variables in rows")

    n_vars, n_samples = X.shape
    n_pairs = n_vars * (n_vars - 1) // 2

    resolved = resolve_device(device, n_samples=n_samples, workload="pstats", n_pairs=n_pairs)

    if resolved == "cuda" and est == "high_fidelity":
        from ._kernels import batch_high_fidelity_cuda

        const_mask = _constant_rows(X)
        mic, tic = batch_high_fidelity_cuda(X, X, alpha=float(alpha), symmetric=True)
        mic, tic = _apply_pstats_constant_mask(mic, tic, const_mask)
        set_last_backend("cuda")
        return mic, tic

    if resolved == "cuda" and est == "fast":
        from ._backend import cuda_native_module

        ranks = _rank_matrix_rows(X)
        const_mask = _constant_rows(X)
        cuda_native = cuda_native_module()
        mic, tic = cuda_native.batch_mic_tic_fast(ranks, ranks, alpha=float(alpha), symmetric=True)
        mic, tic = _apply_pstats_constant_mask(mic, tic, const_mask)
        set_last_backend("cuda")
        return mic, tic

    return _pstats_loop(X, alpha=alpha, c=c, est=est, device=resolved)


def cstats(X, Y, alpha=0.6, c=15, est="high_fidelity", device="auto"):
    est = _normalize_est(est)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays with variables in rows")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of samples")

    n_pairs = X.shape[0] * Y.shape[0]

    resolved = resolve_device(device, n_samples=X.shape[1], workload="cstats", n_pairs=n_pairs)

    if resolved == "cuda" and est == "high_fidelity":
        from ._kernels import batch_high_fidelity_cuda

        const_x = _constant_rows(X)
        const_y = _constant_rows(Y)
        mic, tic = batch_high_fidelity_cuda(X, Y, alpha=float(alpha), symmetric=False)
        mic, tic = _apply_cstats_constant_mask(mic, tic, const_x, const_y)
        set_last_backend("cuda")
        return mic, tic

    if resolved == "cuda" and est == "fast":
        from ._backend import cuda_native_module

        ranks_x = _rank_matrix_rows(X)
        ranks_y = _rank_matrix_rows(Y)
        const_x = _constant_rows(X)
        const_y = _constant_rows(Y)
        cuda_native = cuda_native_module()
        mic, tic = cuda_native.batch_mic_tic_fast(ranks_x, ranks_y, alpha=float(alpha), symmetric=False)
        mic, tic = _apply_cstats_constant_mask(mic, tic, const_x, const_y)
        set_last_backend("cuda")
        return mic, tic

    return _cstats_loop(X, Y, alpha=alpha, c=c, est=est, device=resolved)
