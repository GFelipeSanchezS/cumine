#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>

static void set_err(char* err_msg, int err_msg_len, const char* msg) {
    if (err_msg != nullptr && err_msg_len > 0) {
        std::snprintf(err_msg, static_cast<size_t>(err_msg_len), "%s", msg);
    }
}

static void set_cuda_err(char* err_msg, int err_msg_len, const char* prefix, cudaError_t err) {
    if (err_msg != nullptr && err_msg_len > 0) {
        std::snprintf(
            err_msg,
            static_cast<size_t>(err_msg_len),
            "%s: %s",
            prefix,
            cudaGetErrorString(err)
        );
    }
}

// ---------------------------------------------------------------------------
// Existing single-grid normalized MI kernel
// ---------------------------------------------------------------------------

__global__ void hist_kernel(
    const int* __restrict__ xb,
    const int* __restrict__ yb,
    int n,
    int ry,
    int cells,
    int* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        int x = xb[i];
        int y = yb[i];
        int idx = x * ry + y;
        if (idx >= 0 && idx < cells) {
            atomicAdd(&counts[idx], 1);
        }
    }
}

extern "C" int norm_mi_cuda_impl(
    const int* xb_host,
    const int* yb_host,
    int rx,
    int ry,
    int n,
    double* out_value
) {
    if (xb_host == nullptr || yb_host == nullptr || out_value == nullptr) {
        return 1;
    }

    const int cells = rx * ry;
    const size_t n_bytes = static_cast<size_t>(n) * sizeof(int);
    const size_t cells_bytes = static_cast<size_t>(cells) * sizeof(int);

    int* xb_dev = nullptr;
    int* yb_dev = nullptr;
    int* counts_dev = nullptr;
    int* counts_host = nullptr;

    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&xb_dev), n_bytes);
    if (err != cudaSuccess) return 10;

    err = cudaMalloc(reinterpret_cast<void**>(&yb_dev), n_bytes);
    if (err != cudaSuccess) {
        cudaFree(xb_dev);
        return 11;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&counts_dev), cells_bytes);
    if (err != cudaSuccess) {
        cudaFree(xb_dev);
        cudaFree(yb_dev);
        return 12;
    }

    counts_host = static_cast<int*>(std::calloc(static_cast<size_t>(cells), sizeof(int)));
    if (counts_host == nullptr) {
        cudaFree(xb_dev);
        cudaFree(yb_dev);
        cudaFree(counts_dev);
        return 13;
    }

    err = cudaMemcpy(xb_dev, xb_host, n_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail_copy;

    err = cudaMemcpy(yb_dev, yb_host, n_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail_copy;

    err = cudaMemset(counts_dev, 0, cells_bytes);
    if (err != cudaSuccess) goto fail_copy;

    {
        const int block = 256;
        int grid = (n + block - 1) / block;
        if (grid < 1) grid = 1;
        if (grid > 4096) grid = 4096;

        hist_kernel<<<grid, block>>>(xb_dev, yb_dev, n, ry, cells, counts_dev);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail_copy;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto fail_copy;
    }

    err = cudaMemcpy(counts_host, counts_dev, cells_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail_copy;

    {
        double* row = static_cast<double*>(std::calloc(static_cast<size_t>(rx), sizeof(double)));
        double* col = static_cast<double*>(std::calloc(static_cast<size_t>(ry), sizeof(double)));
        if (row == nullptr || col == nullptr) {
            std::free(row);
            std::free(col);
            std::free(counts_host);
            cudaFree(xb_dev);
            cudaFree(yb_dev);
            cudaFree(counts_dev);
            return 14;
        }

        const double inv_n = 1.0 / static_cast<double>(n);

        for (int i = 0; i < rx; ++i) {
            for (int j = 0; j < ry; ++j) {
                const int count = counts_host[i * ry + j];
                const double p = static_cast<double>(count) * inv_n;
                row[i] += p;
                col[j] += p;
            }
        }

        double mi = 0.0;
        for (int i = 0; i < rx; ++i) {
            for (int j = 0; j < ry; ++j) {
                const int count = counts_host[i * ry + j];
                if (count <= 0) continue;
                const double pxy = static_cast<double>(count) * inv_n;
                const double denom = row[i] * col[j];
                if (denom > 0.0) {
                    mi += pxy * std::log(pxy / denom);
                }
            }
        }

        const double normalizer = std::log(static_cast<double>(rx < ry ? rx : ry));
        *out_value = (normalizer > 0.0) ? (mi / normalizer) : 0.0;

        std::free(row);
        std::free(col);
    }

    std::free(counts_host);
    cudaFree(xb_dev);
    cudaFree(yb_dev);
    cudaFree(counts_dev);
    return 0;

fail_copy:
    std::free(counts_host);
    cudaFree(xb_dev);
    cudaFree(yb_dev);
    cudaFree(counts_dev);
    return 20 + static_cast<int>(err);
}

// ---------------------------------------------------------------------------
// Batched fast-estimator pstats/cstats CUDA path
// ---------------------------------------------------------------------------

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(static_cast<long long>(assumed));
        if (old_val >= val) {
            break;
        }
        old = atomicCAS(
            address_as_ull,
            assumed,
            static_cast<unsigned long long int>(__double_as_longlong(val))
        );
    } while (assumed != old);

    return __longlong_as_double(static_cast<long long>(old));
}


__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull =
        reinterpret_cast<unsigned long long int*>(address);

    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(static_cast<long long>(assumed));
        double new_val = old_val + val;
        old = atomicCAS(
            address_as_ull,
            assumed,
            static_cast<unsigned long long int>(__double_as_longlong(new_val))
        );
    } while (assumed != old);

    return __longlong_as_double(static_cast<long long>(old));
}

__global__ void batch_mi_grid_kernel(
    const int* __restrict__ ranks_x,
    const int* __restrict__ ranks_y,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ rx_list,
    const int* __restrict__ ry_list,
    int n,
    int ny,
    int n_pairs,
    int n_grids,
    double* __restrict__ mic_pair,
    double* __restrict__ tic_pair
) {
    const int block_id = blockIdx.x;
    const int pair_idx = block_id / n_grids;
    const int grid_idx = block_id - pair_idx * n_grids;

    if (pair_idx >= n_pairs || grid_idx >= n_grids) {
        return;
    }

    const int rx = rx_list[grid_idx];
    const int ry = ry_list[grid_idx];
    const int cells = rx * ry;

    extern __shared__ unsigned char smem[];
    int* counts = reinterpret_cast<int*>(smem);
    const size_t counts_bytes = static_cast<size_t>(cells) * sizeof(int);
    const size_t aligned_offset = (counts_bytes + 7u) & ~static_cast<size_t>(7u);
    double* row = reinterpret_cast<double*>(smem + aligned_offset);
    double* col = row + rx;

    for (int idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        counts[idx] = 0;
    }
    for (int idx = threadIdx.x; idx < rx; idx += blockDim.x) {
        row[idx] = 0.0;
    }
    for (int idx = threadIdx.x; idx < ry; idx += blockDim.x) {
        col[idx] = 0.0;
    }
    __syncthreads();

    const int xi = pair_i[pair_idx];
    const int yj = pair_j[pair_idx];
    const int* xrank = ranks_x + static_cast<size_t>(xi) * n;
    const int* yrank = ranks_y + static_cast<size_t>(yj) * n;

    for (int s = threadIdx.x; s < n; s += blockDim.x) {
        int xb = static_cast<int>((static_cast<long long>(xrank[s]) * rx) / n);
        int yb = static_cast<int>((static_cast<long long>(yrank[s]) * ry) / n);
        if (xb >= rx) xb = rx - 1;
        if (yb >= ry) yb = ry - 1;
        if (xb < 0) xb = 0;
        if (yb < 0) yb = 0;
        atomicAdd(&counts[xb * ry + yb], 1);
    }
    __syncthreads();

    // The cell count is small relative to n. Use a single thread for the MI math
    // to keep the first batched CUDA path simple and deterministic.
    if (threadIdx.x == 0) {
        const double inv_n = 1.0 / static_cast<double>(n);

        for (int cell = 0; cell < cells; ++cell) {
            const int cnt = counts[cell];
            if (cnt <= 0) continue;
            const int xbin = cell / ry;
            const int ybin = cell - xbin * ry;
            const double p = static_cast<double>(cnt) * inv_n;
            row[xbin] += p;
            col[ybin] += p;
        }

        double mi = 0.0;
        for (int cell = 0; cell < cells; ++cell) {
            const int cnt = counts[cell];
            if (cnt <= 0) continue;
            const int xbin = cell / ry;
            const int ybin = cell - xbin * ry;
            const double pxy = static_cast<double>(cnt) * inv_n;
            const double denom = row[xbin] * col[ybin];
            if (denom > 0.0) {
                mi += pxy * log(pxy / denom);
            }
        }

        const double normalizer = log(static_cast<double>(rx < ry ? rx : ry));
        const double score = (normalizer > 0.0) ? (mi / normalizer) : 0.0;

        atomicMaxDouble(&mic_pair[pair_idx], score);
        atomicAddDouble(&tic_pair[pair_idx], score);
    }
}

static void make_grid_list(int n, double alpha, std::vector<int>& rx_list, std::vector<int>& ry_list) {
    const int B = std::max(static_cast<int>(std::pow(static_cast<double>(n), alpha)), 4);
    int max_rx = std::min(static_cast<int>(std::floor(static_cast<double>(B) / 2.0)) + 1, n / 2 + 1);
    max_rx = std::max(max_rx, 2);

    for (int rx = 2; rx <= max_rx; ++rx) {
        int max_ry = std::min(static_cast<int>(std::floor(static_cast<double>(B) / static_cast<double>(rx))) + 1, n / 2 + 1);
        max_ry = std::max(max_ry, 2);
        for (int ry = 2; ry <= max_ry; ++ry) {
            rx_list.push_back(rx);
            ry_list.push_back(ry);
        }
    }
}

static void make_pairs(int nx, int ny, int symmetric, std::vector<int>& pair_i, std::vector<int>& pair_j) {
    if (symmetric) {
        for (int i = 0; i < nx; ++i) {
            for (int j = i + 1; j < ny; ++j) {
                pair_i.push_back(i);
                pair_j.push_back(j);
            }
        }
    } else {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                pair_i.push_back(i);
                pair_j.push_back(j);
            }
        }
    }
}

extern "C" int batch_mic_tic_equifreq_cuda_impl(
    const int* ranks_x_host,
    const int* ranks_y_host,
    int nx,
    int ny,
    int n,
    double alpha,
    int symmetric,
    double* mic_out_host,
    double* tic_out_host,
    char* err_msg,
    int err_msg_len
) {
    if (ranks_x_host == nullptr || ranks_y_host == nullptr || mic_out_host == nullptr || tic_out_host == nullptr) {
        set_err(err_msg, err_msg_len, "null input pointer");
        return 1;
    }
    if (nx < 1 || ny < 1 || n < 2 || alpha <= 0.0 || alpha > 1.0) {
        set_err(err_msg, err_msg_len, "invalid dimensions or alpha");
        return 2;
    }
    if (symmetric && nx != ny) {
        set_err(err_msg, err_msg_len, "symmetric mode requires nx == ny");
        return 3;
    }

    std::vector<int> rx_list_host;
    std::vector<int> ry_list_host;
    make_grid_list(n, alpha, rx_list_host, ry_list_host);
    const int n_grids = static_cast<int>(rx_list_host.size());
    if (n_grids < 1) {
        set_err(err_msg, err_msg_len, "empty grid list");
        return 4;
    }

    std::vector<int> pair_i_host;
    std::vector<int> pair_j_host;
    make_pairs(nx, ny, symmetric, pair_i_host, pair_j_host);
    const int n_pairs = static_cast<int>(pair_i_host.size());

    // Initialize output matrices. For pstats, diagonal is a perfect self-association.
    const int matrix_size = nx * ny;
    for (int idx = 0; idx < matrix_size; ++idx) {
        mic_out_host[idx] = 0.0;
        tic_out_host[idx] = 0.0;
    }
    if (symmetric) {
        for (int i = 0; i < nx; ++i) {
            mic_out_host[i * ny + i] = 1.0;
            tic_out_host[i * ny + i] = 1.0;
        }
    }

    if (n_pairs == 0) {
        return 0;
    }

    int max_rx = 0;
    int max_ry = 0;
    int max_cells = 0;
    for (int g = 0; g < n_grids; ++g) {
        max_rx = std::max(max_rx, rx_list_host[g]);
        max_ry = std::max(max_ry, ry_list_host[g]);
        max_cells = std::max(max_cells, rx_list_host[g] * ry_list_host[g]);
    }
    const size_t counts_bytes = static_cast<size_t>(max_cells) * sizeof(int);
    const size_t aligned_offset = (counts_bytes + 7u) & ~static_cast<size_t>(7u);
    const size_t shared_bytes = aligned_offset + static_cast<size_t>(max_rx + max_ry) * sizeof(double);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        set_cuda_err(err_msg, err_msg_len, "cudaGetDeviceProperties failed", err);
        return 10;
    }
    if (shared_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        set_err(err_msg, err_msg_len, "grid requires more shared memory than the active GPU allows");
        return 11;
    }

    int* ranks_x_dev = nullptr;
    int* ranks_y_dev = nullptr;
    int* pair_i_dev = nullptr;
    int* pair_j_dev = nullptr;
    int* rx_list_dev = nullptr;
    int* ry_list_dev = nullptr;
    double* mic_pair_dev = nullptr;
    double* tic_pair_dev = nullptr;

    const size_t ranks_x_bytes = static_cast<size_t>(nx) * n * sizeof(int);
    const size_t ranks_y_bytes = static_cast<size_t>(ny) * n * sizeof(int);
    const size_t pairs_bytes = static_cast<size_t>(n_pairs) * sizeof(int);
    const size_t grids_bytes = static_cast<size_t>(n_grids) * sizeof(int);
    const size_t pair_scores_bytes = static_cast<size_t>(n_pairs) * sizeof(double);

    err = cudaMalloc(reinterpret_cast<void**>(&ranks_x_dev), ranks_x_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc ranks_x failed", err); return 20; }
    err = cudaMalloc(reinterpret_cast<void**>(&ranks_y_dev), ranks_y_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc ranks_y failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&pair_i_dev), pairs_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc pair_i failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&pair_j_dev), pairs_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc pair_j failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&rx_list_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc rx_list failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&ry_list_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc ry_list failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&mic_pair_dev), pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc mic_pair failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&tic_pair_dev), pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc tic_pair failed", err); goto fail; }

    err = cudaMemcpy(ranks_x_dev, ranks_x_host, ranks_x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy ranks_x failed", err); goto fail; }
    err = cudaMemcpy(ranks_y_dev, ranks_y_host, ranks_y_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy ranks_y failed", err); goto fail; }
    err = cudaMemcpy(pair_i_dev, pair_i_host.data(), pairs_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy pair_i failed", err); goto fail; }
    err = cudaMemcpy(pair_j_dev, pair_j_host.data(), pairs_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy pair_j failed", err); goto fail; }
    err = cudaMemcpy(rx_list_dev, rx_list_host.data(), grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy rx_list failed", err); goto fail; }
    err = cudaMemcpy(ry_list_dev, ry_list_host.data(), grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy ry_list failed", err); goto fail; }
    err = cudaMemset(mic_pair_dev, 0, pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "memset mic_pair failed", err); goto fail; }
    err = cudaMemset(tic_pair_dev, 0, pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "memset tic_pair failed", err); goto fail; }

    {
        const int block = 256;
        const long long total_blocks_ll = static_cast<long long>(n_pairs) * static_cast<long long>(n_grids);
        if (total_blocks_ll > 2147483647LL) {
            set_err(err_msg, err_msg_len, "too many pair-grid blocks for one CUDA launch");
            goto fail;
        }
        const int grid = static_cast<int>(total_blocks_ll);

        batch_mi_grid_kernel<<<grid, block, shared_bytes>>>(
            ranks_x_dev,
            ranks_y_dev,
            pair_i_dev,
            pair_j_dev,
            rx_list_dev,
            ry_list_dev,
            n,
            ny,
            n_pairs,
            n_grids,
            mic_pair_dev,
            tic_pair_dev
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "batch kernel launch failed", err); goto fail; }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "batch kernel synchronize failed", err); goto fail; }
    }

    {
        std::vector<double> mic_pair_host(static_cast<size_t>(n_pairs), 0.0);
        std::vector<double> tic_pair_host(static_cast<size_t>(n_pairs), 0.0);
        err = cudaMemcpy(mic_pair_host.data(), mic_pair_dev, pair_scores_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy mic_pair failed", err); goto fail; }
        err = cudaMemcpy(tic_pair_host.data(), tic_pair_dev, pair_scores_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy tic_pair failed", err); goto fail; }

        for (int p = 0; p < n_pairs; ++p) {
            const int i = pair_i_host[p];
            const int j = pair_j_host[p];
            const double mic = mic_pair_host[p];
            const double tic_norm = tic_pair_host[p] / static_cast<double>(n_grids);
            mic_out_host[i * ny + j] = mic;
            tic_out_host[i * ny + j] = tic_norm;
            if (symmetric) {
                mic_out_host[j * ny + i] = mic;
                tic_out_host[j * ny + i] = tic_norm;
            }
        }
    }

    cudaFree(ranks_x_dev);
    cudaFree(ranks_y_dev);
    cudaFree(pair_i_dev);
    cudaFree(pair_j_dev);
    cudaFree(rx_list_dev);
    cudaFree(ry_list_dev);
    cudaFree(mic_pair_dev);
    cudaFree(tic_pair_dev);
    return 0;

fail:
    cudaFree(ranks_x_dev);
    cudaFree(ranks_y_dev);
    cudaFree(pair_i_dev);
    cudaFree(pair_j_dev);
    cudaFree(rx_list_dev);
    cudaFree(ry_list_dev);
    cudaFree(mic_pair_dev);
    cudaFree(tic_pair_dev);
    return 100;
}

// ---------------------------------------------------------------------------
// Batched high_fidelity estimator CUDA DP core
// ---------------------------------------------------------------------------

__device__ double hf_interval_score_device(
    const int* __restrict__ pc,
    const int* __restrict__ cuts,
    int K,
    int ry,
    int n,
    int a,
    int b
) {
    const int length = cuts[b] - cuts[a];
    if (length <= 0) return -1.0e300;

    const double inv_n = 1.0 / static_cast<double>(n);
    const double p_col = static_cast<double>(length) * inv_n;
    double val = 0.0;

    const int total_row = K - 1;
    const int* pc_a = pc + static_cast<size_t>(a) * ry;
    const int* pc_b = pc + static_cast<size_t>(b) * ry;
    const int* pc_total = pc + static_cast<size_t>(total_row) * ry;

    for (int yy = 0; yy < ry; ++yy) {
        const int cnt = pc_b[yy] - pc_a[yy];
        if (cnt <= 0) continue;

        const double pxy = static_cast<double>(cnt) * inv_n;
        const double py = static_cast<double>(pc_total[yy]) * inv_n;
        const double expected = p_col * py;
        if (expected > 0.0) {
            val += pxy * log(pxy / expected);
        }
    }
    return val;
}

__global__ void high_fidelity_dp_kernel(
    const int* __restrict__ prefix_flat,
    const int* __restrict__ prefix_offsets,
    const int* __restrict__ rx_list,
    const int* __restrict__ ry_list,
    const int* __restrict__ ry_index,
    const int* __restrict__ cuts,
    int K,
    int n,
    int n_pairs,
    int n_rys,
    int n_grids,
    double* __restrict__ grid_scores,
    double* __restrict__ mic_pair,
    double* __restrict__ tic_pair
) {
    const int block_id = blockIdx.x;
    const int pair_idx = block_id / n_grids;
    const int grid_idx = block_id - pair_idx * n_grids;

    if (pair_idx >= n_pairs || grid_idx >= n_grids) return;

    const int rx = rx_list[grid_idx];
    const int ry = ry_list[grid_idx];
    const int ridx = ry_index[grid_idx];
    const int prefix_offset = prefix_offsets[pair_idx * n_rys + ridx];
    const int* pc = prefix_flat + static_cast<size_t>(prefix_offset);

    extern __shared__ unsigned char smem[];
    double* dp0 = reinterpret_cast<double*>(smem);
    double* dp1 = dp0 + K;

    for (int b = threadIdx.x; b < K; b += blockDim.x) {
        dp0[b] = (b == 0) ? 0.0 : -1.0e300;
        dp1[b] = -1.0e300;
    }
    __syncthreads();

    double* prev = dp0;
    double* curr = dp1;

    for (int kk = 1; kk <= rx; ++kk) {
        for (int b = threadIdx.x; b < K; b += blockDim.x) {
            double best = -1.0e300;
            if (b >= kk) {
                for (int a = kk - 1; a < b; ++a) {
                    const double prev_val = prev[a];
                    if (prev_val <= -1.0e299) continue;
                    const double interval = hf_interval_score_device(pc, cuts, K, ry, n, a, b);
                    const double candidate = prev_val + interval;
                    if (candidate > best) best = candidate;
                }
            }
            curr[b] = best;
        }
        __syncthreads();
        double* tmp = prev;
        prev = curr;
        curr = tmp;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double score = 0.0;
        const double mi = prev[K - 1];
        const int min_dim = rx < ry ? rx : ry;
        const double denom = log(static_cast<double>(min_dim));
        if (denom > 0.0 && mi > -1.0e299) {
            score = mi / denom;
            if (score < 0.0) score = 0.0;
            if (score > 1.0) score = 1.0;
        }

        if (grid_scores != nullptr && n_pairs == 1) {
            grid_scores[grid_idx] = score;
        }
        if (mic_pair != nullptr) {
            atomicMaxDouble(&mic_pair[pair_idx], score);
        }
        if (tic_pair != nullptr) {
            atomicAddDouble(&tic_pair[pair_idx], score);
        }
    }
}

extern "C" int high_fidelity_scores_cuda_impl(
    const int* prefix_flat_host,
    const int* prefix_offsets_host,
    const int* rx_list_host,
    const int* ry_list_host,
    const int* ry_index_host,
    const int* cuts_host,
    int prefix_flat_len,
    int n_rys,
    int n_grids,
    int K,
    int n,
    double* scores_out_host,
    char* err_msg,
    int err_msg_len
) {
    if (prefix_flat_host == nullptr || prefix_offsets_host == nullptr || rx_list_host == nullptr ||
        ry_list_host == nullptr || ry_index_host == nullptr || cuts_host == nullptr || scores_out_host == nullptr) {
        set_err(err_msg, err_msg_len, "null input pointer");
        return 1;
    }
    if (prefix_flat_len < 1 || n_rys < 1 || n_grids < 1 || K < 2 || n < 2) {
        set_err(err_msg, err_msg_len, "invalid high_fidelity dimensions");
        return 2;
    }

    int* prefix_flat_dev = nullptr;
    int* prefix_offsets_dev = nullptr;
    int* rx_list_dev = nullptr;
    int* ry_list_dev = nullptr;
    int* ry_index_dev = nullptr;
    int* cuts_dev = nullptr;
    double* scores_dev = nullptr;

    cudaError_t err;
    const size_t prefix_bytes = static_cast<size_t>(prefix_flat_len) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(n_rys) * sizeof(int);
    const size_t grids_bytes = static_cast<size_t>(n_grids) * sizeof(int);
    const size_t cuts_bytes = static_cast<size_t>(K) * sizeof(int);
    const size_t scores_bytes = static_cast<size_t>(n_grids) * sizeof(double);

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaGetDeviceProperties failed", err); return 10; }
    const size_t shared_bytes = static_cast<size_t>(2 * K) * sizeof(double);
    if (shared_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        set_err(err_msg, err_msg_len, "high_fidelity DP requires more shared memory than the active GPU allows");
        return 11;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&prefix_flat_dev), prefix_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc prefix_flat failed", err); return 20; }
    err = cudaMalloc(reinterpret_cast<void**>(&prefix_offsets_dev), offsets_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc prefix_offsets failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&rx_list_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc rx_list failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&ry_list_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc ry_list failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&ry_index_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc ry_index failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&cuts_dev), cuts_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc cuts failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&scores_dev), scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc scores failed", err); goto fail; }

    err = cudaMemcpy(prefix_flat_dev, prefix_flat_host, prefix_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy prefix_flat failed", err); goto fail; }
    err = cudaMemcpy(prefix_offsets_dev, prefix_offsets_host, offsets_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy prefix_offsets failed", err); goto fail; }
    err = cudaMemcpy(rx_list_dev, rx_list_host, grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy rx_list failed", err); goto fail; }
    err = cudaMemcpy(ry_list_dev, ry_list_host, grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy ry_list failed", err); goto fail; }
    err = cudaMemcpy(ry_index_dev, ry_index_host, grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy ry_index failed", err); goto fail; }
    err = cudaMemcpy(cuts_dev, cuts_host, cuts_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy cuts failed", err); goto fail; }

    err = cudaMemset(scores_dev, 0, scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "memset scores failed", err); goto fail; }

    {
        const int block = 256;
        high_fidelity_dp_kernel<<<n_grids, block, shared_bytes>>>(
            prefix_flat_dev,
            prefix_offsets_dev,
            rx_list_dev,
            ry_list_dev,
            ry_index_dev,
            cuts_dev,
            K,
            n,
            1,
            n_rys,
            n_grids,
            scores_dev,
            nullptr,
            nullptr
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "high_fidelity score kernel launch failed", err); goto fail; }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "high_fidelity score kernel sync failed", err); goto fail; }
    }

    err = cudaMemcpy(scores_out_host, scores_dev, scores_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy scores failed", err); goto fail; }

    cudaFree(prefix_flat_dev);
    cudaFree(prefix_offsets_dev);
    cudaFree(rx_list_dev);
    cudaFree(ry_list_dev);
    cudaFree(ry_index_dev);
    cudaFree(cuts_dev);
    cudaFree(scores_dev);
    return 0;

fail:
    cudaFree(prefix_flat_dev);
    cudaFree(prefix_offsets_dev);
    cudaFree(rx_list_dev);
    cudaFree(ry_list_dev);
    cudaFree(ry_index_dev);
    cudaFree(cuts_dev);
    cudaFree(scores_dev);
    return 100;
}

extern "C" int batch_high_fidelity_mic_tic_cuda_impl(
    const int* prefix_flat_host,
    const int* prefix_offsets_host,
    const int* rx_list_host,
    const int* ry_list_host,
    const int* ry_index_host,
    const int* cuts_host,
    int prefix_flat_len,
    int n_pairs,
    int n_rys,
    int n_grids,
    int K,
    int n,
    double* mic_pair_out_host,
    double* tic_pair_out_host,
    char* err_msg,
    int err_msg_len
) {
    if (prefix_flat_host == nullptr || prefix_offsets_host == nullptr || rx_list_host == nullptr ||
        ry_list_host == nullptr || ry_index_host == nullptr || cuts_host == nullptr ||
        mic_pair_out_host == nullptr || tic_pair_out_host == nullptr) {
        set_err(err_msg, err_msg_len, "null input pointer");
        return 1;
    }
    if (prefix_flat_len < 1 || n_pairs < 1 || n_rys < 1 || n_grids < 1 || K < 2 || n < 2) {
        set_err(err_msg, err_msg_len, "invalid batched high_fidelity dimensions");
        return 2;
    }

    int* prefix_flat_dev = nullptr;
    int* prefix_offsets_dev = nullptr;
    int* rx_list_dev = nullptr;
    int* ry_list_dev = nullptr;
    int* ry_index_dev = nullptr;
    int* cuts_dev = nullptr;
    double* mic_pair_dev = nullptr;
    double* tic_pair_dev = nullptr;

    cudaError_t err;
    const size_t prefix_bytes = static_cast<size_t>(prefix_flat_len) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(n_pairs) * static_cast<size_t>(n_rys) * sizeof(int);
    const size_t grids_bytes = static_cast<size_t>(n_grids) * sizeof(int);
    const size_t cuts_bytes = static_cast<size_t>(K) * sizeof(int);
    const size_t pair_scores_bytes = static_cast<size_t>(n_pairs) * sizeof(double);

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaGetDeviceProperties failed", err); return 10; }
    const size_t shared_bytes = static_cast<size_t>(2 * K) * sizeof(double);
    if (shared_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        set_err(err_msg, err_msg_len, "high_fidelity DP requires more shared memory than the active GPU allows");
        return 11;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&prefix_flat_dev), prefix_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc prefix_flat failed", err); return 20; }
    err = cudaMalloc(reinterpret_cast<void**>(&prefix_offsets_dev), offsets_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc prefix_offsets failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&rx_list_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc rx_list failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&ry_list_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc ry_list failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&ry_index_dev), grids_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc ry_index failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&cuts_dev), cuts_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc cuts failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&mic_pair_dev), pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc mic_pair failed", err); goto fail; }
    err = cudaMalloc(reinterpret_cast<void**>(&tic_pair_dev), pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "cudaMalloc tic_pair failed", err); goto fail; }

    err = cudaMemcpy(prefix_flat_dev, prefix_flat_host, prefix_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy prefix_flat failed", err); goto fail; }
    err = cudaMemcpy(prefix_offsets_dev, prefix_offsets_host, offsets_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy prefix_offsets failed", err); goto fail; }
    err = cudaMemcpy(rx_list_dev, rx_list_host, grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy rx_list failed", err); goto fail; }
    err = cudaMemcpy(ry_list_dev, ry_list_host, grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy ry_list failed", err); goto fail; }
    err = cudaMemcpy(ry_index_dev, ry_index_host, grids_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy ry_index failed", err); goto fail; }
    err = cudaMemcpy(cuts_dev, cuts_host, cuts_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy cuts failed", err); goto fail; }
    err = cudaMemset(mic_pair_dev, 0, pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "memset mic_pair failed", err); goto fail; }
    err = cudaMemset(tic_pair_dev, 0, pair_scores_bytes);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "memset tic_pair failed", err); goto fail; }

    {
        const int block = 256;
        const long long total_blocks_ll = static_cast<long long>(n_pairs) * static_cast<long long>(n_grids);
        if (total_blocks_ll > 2147483647LL) {
            set_err(err_msg, err_msg_len, "too many high_fidelity pair-grid blocks for one CUDA launch");
            goto fail;
        }
        const int grid = static_cast<int>(total_blocks_ll);

        high_fidelity_dp_kernel<<<grid, block, shared_bytes>>>(
            prefix_flat_dev,
            prefix_offsets_dev,
            rx_list_dev,
            ry_list_dev,
            ry_index_dev,
            cuts_dev,
            K,
            n,
            n_pairs,
            n_rys,
            n_grids,
            nullptr,
            mic_pair_dev,
            tic_pair_dev
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "batched high_fidelity kernel launch failed", err); goto fail; }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "batched high_fidelity kernel sync failed", err); goto fail; }
    }

    err = cudaMemcpy(mic_pair_out_host, mic_pair_dev, pair_scores_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy mic_pair failed", err); goto fail; }
    err = cudaMemcpy(tic_pair_out_host, tic_pair_dev, pair_scores_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_cuda_err(err_msg, err_msg_len, "copy tic_pair failed", err); goto fail; }

    cudaFree(prefix_flat_dev);
    cudaFree(prefix_offsets_dev);
    cudaFree(rx_list_dev);
    cudaFree(ry_list_dev);
    cudaFree(ry_index_dev);
    cudaFree(cuts_dev);
    cudaFree(mic_pair_dev);
    cudaFree(tic_pair_dev);
    return 0;

fail:
    cudaFree(prefix_flat_dev);
    cudaFree(prefix_offsets_dev);
    cudaFree(rx_list_dev);
    cudaFree(ry_list_dev);
    cudaFree(ry_index_dev);
    cudaFree(cuts_dev);
    cudaFree(mic_pair_dev);
    cudaFree(tic_pair_dev);
    return 100;
}
