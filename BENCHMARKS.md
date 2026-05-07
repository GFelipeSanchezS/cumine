# Benchmark checkpoint

Hardware checkpoint used for the current development numbers:

- GPU: RTX 2080 Ti class device
- CUDA toolkit: CUDA 12.x environment
- Python: 3.10
- Package mode: editable install with `CUMINE_BUILD_CUDA=1`

Run the full validation set before trusting benchmark numbers:

```bash
pytest -qv
CUMINE_DEVICE=cuda pytest -qv
python tools/characterize_estimators.py
```

## Fast estimator

Commands:

```bash
python benchmarks/pairwise.py --est fast --device cpu
python benchmarks/pairwise.py --est fast --device cuda
```

Representative results:

| Workload | CPU | Native CUDA | Approx. speedup |
|---|---:|---:|---:|
| `pstats`, 10 vars, 1000 samples, 45 pairs | 0.448 s | 0.361 s | 1.2× |
| `pstats`, 25 vars, 2000 samples, 300 pairs | 6.515 s | 0.061 s | 107× |
| `pstats`, 50 vars, 5000 samples, 1225 pairs | 93.656 s | 0.594 s | 158× |
| `pstats`, 100 vars, 5000 samples, 4950 pairs | 373.055 s | 2.289 s | 163× |
| `cstats`, 10×8 vars, 1000 samples, 80 pairs | 0.782 s | 0.007 s | 112× |
| `cstats`, 25×20 vars, 2000 samples, 500 pairs | 10.766 s | 0.075 s | 144× |
| `cstats`, 50×40 vars, 5000 samples, 2000 pairs | 147.843 s | 0.951 s | 155× |

## High-fidelity estimator

Commands:

```bash
CUMINE_DEVICE=cpu python benchmarks/high_fidelity.py
CUMINE_DEVICE=cuda python benchmarks/high_fidelity.py
python benchmarks/pairwise.py --est high_fidelity --device cpu
python benchmarks/pairwise.py --est high_fidelity --device cuda
python benchmarks/pairwise.py --est high_fidelity --device cuda --scale large
```

Representative single-pair results:

| Samples | CPU | Native CUDA | Approx. speedup |
|---:|---:|---:|---:|
| 300 | 0.195 s | 0.401 s | CPU faster; CUDA overhead dominates |
| 800 | 0.916 s | 0.027 s | 34× |
| 2000 | 4.699 s | 0.154 s | 31× |
| 5000 | 24.162 s | 0.990 s | 24× |

Representative pairwise results:

| Workload | CPU | Native CUDA | Approx. speedup |
|---|---:|---:|---:|
| `pstats`, 6 vars, 300 samples, 15 pairs | 2.728 s | 0.410 s | 6.7× |
| `pstats`, 8 vars, 500 samples, 28 pairs | 10.746 s | 0.128 s | 84× |
| `pstats`, 10 vars, 800 samples, 45 pairs | 39.739 s | 0.491 s | 81× |
| `cstats`, 5×4 vars, 300 samples, 20 pairs | 3.645 s | 0.036 s | 101× |
| `cstats`, 8×6 vars, 500 samples, 48 pairs | 18.412 s | 0.168 s | 110× |
| `cstats`, 10×8 vars, 800 samples, 80 pairs | 70.909 s | 0.853 s | 83× |

Large CUDA stress checkpoint:

| Workload | Native CUDA |
|---|---:|
| `pstats`, 16 vars, 1000 samples, 120 pairs | 1.805 s |
| `pstats`, 25 vars, 1500 samples, 300 pairs | 15.333 s |
| `cstats`, 16×12 vars, 1000 samples, 192 pairs | 2.906 s |
| `cstats`, 25×20 vars, 1500 samples, 500 pairs | 25.785 s |

## Notes

- Small CUDA workloads can be slower than CPU because kernel launch/setup overhead dominates.
- `high_fidelity` CUDA accelerates the adaptive DP scoring core. Host-side rank/order and prefix preparation remain CPU-side.
- Benchmark numbers vary by CPU, GPU, CUDA toolkit, driver, and architecture flags.
