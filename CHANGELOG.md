# Changelog

## Unreleased

### Performance: overlap host-side prefix construction with the CUDA kernel in `batch_high_fidelity_cuda`

**Problem.** `nvidia-smi` showed only ~65-66% GPU utilization during real `high_fidelity` CUDA runs
(observed while running a 16,383x16,383 gene-association screen), while the Python worker process
sat at a steady 100% of a single CPU core. Root cause: `batch_high_fidelity_cuda` (in `_kernels.py`)
processes pairs in small batches (`max_pairs_per_chunk`, default 32) and, for each batch, first runs
a serial Python loop building rank-based cut-prefix arrays on the host (`_high_fidelity_prefix_single`,
one pair at a time), *then* calls the native CUDA kernel (`batch_high_fidelity_mic_tic`) and blocks
until it returns. The two phases never overlapped — compounded by the four native entry points in
`cuda_ext.cpp` never releasing the GIL during their blocking CUDA calls, so even if something else had
tried to run concurrently, nothing could.

**Fix (two changes, both required):**

1. `cumine/cuda_ext.cpp` — wrapped the blocking CUDA calls in all four native entry points
   (`py_norm_mi`, `py_batch_mic_tic_equifreq`, `py_high_fidelity_scores`, `py_batch_high_fidelity_mic_tic`)
   in `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`. Safe here because nothing between those macros
   touches a Python object — only raw pointers already extracted from the NumPy arrays beforehand.
2. `cumine/_kernels.py` — `batch_high_fidelity_cuda` now uses a `ThreadPoolExecutor(max_workers=1)` to
   prepare chunk *N+1*'s host-side prefix arrays (`_prepare_chunk_prefix`, factored out of the main loop)
   while chunk *N*'s CUDA kernel is in flight, instead of doing the two phases strictly back-to-back.
   This only helps because of change (1) — without the GIL being released during the kernel call, a
   background thread would never actually get to run concurrently with it.

**Measured impact** (500x500 gene-pair block, n=147 samples, `high_fidelity`, RTX 3090):

| | Before | After |
|---|---|---|
| Wall clock | 130.0s | 91.9s (~1.41x faster) |
| `nvidia-smi` GPU utilization | ~65-66% | ~93% |
| Worker CPU usage | ~100% (1 core) | ~140% (overlap confirmed) |

No change in output values (same MIC range observed before/after on identical input). All 35
existing tests pass unchanged (1 pre-existing skip, unrelated).

**Files touched:** `cumine/cuda_ext.cpp`, `cumine/_kernels.py`.

**Not yet done / ideas for a follow-up pass:**
- The same serialize-then-launch pattern likely affects `high_fidelity_scores` (single-pair path,
  `_char_matrix_high_fidelity_cuda`) and the `fast`-estimator batch path — not fixed here since they
  weren't the measured bottleneck for this workload, but worth the same treatment.
- `_prepare_chunk_prefix` itself is still single-threaded per chunk; for very large `max_pairs_per_chunk`
  it could additionally be parallelized *across* pairs within a chunk (e.g. via multiple worker threads
  or a process pool), on top of the current N/N+1 chunk overlap — untried, no measured need yet at
  `max_pairs_per_chunk=32`.
- **Fully GPU-side preprocessing (port rank/binning/cut-prefix construction to CUDA kernels too), so
  `batch_high_fidelity_cuda` never touches the CPU at all.** Technically doable — rank computation
  (argsort), equal-frequency binning, and cut-prefix count construction are all fairly parallelizable
  (sorting, bucketing, segmented prefix sums), just currently written as host-side NumPy rather than
  CUDA kernels. **Estimated payoff is small at this point, though** — back-of-envelope from the
  before/after numbers above: unpatched total (130.0s) = CPU-phase + GPU-phase; patched total (91.9s)
  ≈ `max(CPU-phase, GPU-phase)`. That implies GPU-phase ≈ 91.9s and CPU-phase ≈ 38.1s — i.e. the CPU
  phase is already shorter than the GPU phase and is now almost entirely hidden by the overlap fix
  above. Eliminating it entirely would only close the remaining ~7% utilization gap (mostly first-chunk
  startup latency, since there's nothing to overlap with before the very first GPU call), not deliver
  another ~1.4x jump. Would require writing/debugging a parallel sort, parallel binning, and a batched
  segmented-prefix-sum kernel — real effort for a single-digit-percent gain *unless* a future faster
  GPU or larger CPU workload shifts the balance (i.e. if CPU-phase ever exceeds GPU-phase, this jumps
  from "diminishing returns" to "the actual bottleneck," worth re-measuring before committing to it).

  **IMPORTANT SCOPING NOTE:** cuMINE's goal is to be a general-purpose minepy replacement (arbitrary
  sample counts `n`, not just this project's n=147) with GPU support added on top — so any kernel
  design here must not assume small `n`. An earlier draft of this note sketched an n=147-specific
  approach (O(n^2) counting-based rank, naive serial-scan prefix construction) that is a fine *fast
  path* for small `n` but would be a bad, potentially very slow choice at large `n` (thousands+
  samples) if used unconditionally — it does not belong as the *only* implementation. Corrected below
  to a size-adaptive design with two paths and a runtime dispatch between them.

  **Hard requirement, not just an implementation detail: no artificial cap on `n`, matching minepy.**
  minepy's own ApproxMaxMI/DP core has no hard-coded sample-count limit — it's polynomial in `n` and
  just gets slower as `n` grows (single-threaded C, no parallelism to hide that growth). cuMINE must
  have the same property. Concretely: both the small-`n` and large-`n` paths below must be **correct
  for any `n`** — the crossover threshold between them is purely a *speed* decision (which path
  finishes faster), never a *capability* boundary. Neither path should refuse to run, error out, or
  silently produce wrong answers above/below some size; the small-n path is allowed to become slow at
  large `n` (real work grows, same as minepy), but never unavailable. The only ceiling either path can
  legitimately hit is memory (per-gene arrays and batch results fitting in available RAM/VRAM) — that's
  a hardware constraint shared with minepy, not a cuMINE-specific limitation, and not something to
  design around with an artificial size cap.

  1. **Rank computation** (replaces `_rank_stable`, currently `argsort(argsort(x))`).
     - *Small-`n` path*: one thread per element `i`, each counts how many other elements are smaller
       (with an index tiebreak for stability): `rank[i] = sum(x[j] < x[i] or (x[j] == x[i] and j < i)
       for j in range(n))`. O(n^2) work per gene, but trivial to implement correctly and fast when
       n is small enough that O(n^2) total work is still less than a sort's overhead.
     - *Large-`n` path*: use `thrust::sort_by_key` (radix or merge sort, ships with the CUDA toolkit —
       no new dependency) to get order, then derive ranks from sorted position with the same
       index-tiebreak logic for stability. O(n log n).
     - *Dispatch*: pick a crossover threshold empirically (benchmark both paths across a range of n
       on real hardware — do not guess a number without measuring) and branch on it at call time.
  2. **Equal-frequency binning** (replaces `_equifreq_bins`). Scale-agnostic already —
     `bin[i] = min(rank[i] * k // n, k - 1)` is a pure elementwise op, one thread per element, O(n)
     total regardless of size. No size-adaptive design needed here, just move it into a kernel.
  3. **Cut-prefix count construction** (replaces `_cut_prefix_counts`). This is a one-hot-then-cumsum
     along the sample axis per bin (see the actual NumPy: `prefix[arange(1,n+1), yb_ordered] = 1;
     cumsum(axis=0)`) — a batched, per-bin running count evaluated only at the specific `cuts`
     positions needed.
     - *Small-`n` path*: one thread per (pair, grid-resolution, bin) triple, each doing a plain
       serial O(n) scan over its own sequence. Parallelism comes from the sheer number of
       (pair x grid-resolution x bin) combinations in a chunk, not from optimizing the inner scan.
     - *Large-`n` path*: a per-triple serial O(n) scan stops being competitive as n grows — switch to
       a real parallel scan per triple (`cub::BlockScan`/`cub::DeviceScan`, or a custom Hillis-Steele/
       Blelloch scan), since CUB ships with the CUDA toolkit and is built for exactly this
       batched/segmented-scan shape.
     - *Dispatch*: same crossover-threshold approach as (1); the two operations likely share a similar
       n-dependent crossover point since both trade "more total work, no sync overhead" against
       "less total work, real cross-thread synchronization" — but confirm independently, don't assume
       they match.

  All three integrate the same way the existing DP kernel does: build/copy the small per-gene arrays
  to device once per chunk (or cache like `order_x_cache`/`y_rank_cache` already do host-side), then
  launch one kernel per stage instead of the current three Python/NumPy passes in
  `_high_fidelity_prefix_single`. Before committing to this, re-run the same before/after benchmark
  harness used above (500x500 block, n=147, `high_fidelity`) for the small-n case *and* a large-n
  benchmark (representative of cuMINE's broader intended use, not just this project) to confirm real
  gain in both regimes — the small-n result alone (this project's scale) is not sufficient evidence
  for a general library decision.
