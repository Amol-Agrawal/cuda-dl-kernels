// kernels_wmma.cuh - FP16 tensor-core GEMM via WMMA (sm_70+; T4 = sm_75).
// Mixed precision: A, B in FP16, accumulate + output in FP32. Row-major.
// C = A(MxK) * B(KxN)
#pragma once
#include <mma.h>
#include <cuda_fp16.h>
#include "common.cuh"

using namespace nvcuda;

// ---------------------------------------------------------------------------
// TC-0: naive WMMA. One warp computes one 16x16 tile of C, loading fragments
// straight from global memory. No data reuse between warps.
// Block = 4 warps arranged 2x2 -> 32x32 tile of C per block.
// Requires M, N, K multiples of 16.
// ---------------------------------------------------------------------------
__global__ void hgemm_wmma_naive(int M, int N, int K, const half* A,
                                 const half* B, float* C) {
  const int warp = threadIdx.x / 32;
  const int tileRow = (blockIdx.y * 2 + warp / 2) * 16;
  const int tileCol = (blockIdx.x * 2 + warp % 2) * 16;
  if (tileRow >= M || tileCol >= N) return;  // warp-uniform exit

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.f);

  for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a, A + tileRow * K + k, K);
    wmma::load_matrix_sync(b, B + k * N + tileCol, N);
    wmma::mma_sync(c, a, b, c);
  }
  wmma::store_matrix_sync(C + tileRow * N + tileCol, c, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Thread-block swizzling (used by TC-3).
// Blocks are dispatched roughly in linear blockIdx order (x fastest). With the
// plain mapping, the ~100 blocks resident at once cover a few block-rows x ALL
// block-columns, so they stream almost all of B through the 4 MB L2. Grouped
// ordering walks GROUP_M block-rows down before moving right, so the resident
// blocks form a squarer patch and share more A and B panels in L2.
// ---------------------------------------------------------------------------
template <bool SWIZZLE, int GROUP_M = 8>
__device__ __forceinline__ void block_tile_origin(int BMv, int BNv, int& bm, int& bn) {
  if (!SWIZZLE) {
    bm = blockIdx.y * BMv;
    bn = blockIdx.x * BNv;
    return;
  }
  const int num_m = gridDim.y, num_n = gridDim.x;
  const int pid = blockIdx.y * num_n + blockIdx.x;
  const int per_group = GROUP_M * num_n;
  const int first_m = (pid / per_group) * GROUP_M;
  const int gsize = min(num_m - first_m, GROUP_M);
  const int local = pid % per_group;
  bm = (first_m + local % gsize) * BMv;
  bn = (local / gsize) * BNv;
}

// ---------------------------------------------------------------------------
// TC-1: shared-memory tiled WMMA.
// Block tile 128x128, K-step 32, 8 warps (2 x 4). Each warp owns a 64x32
// sub-tile = 4x2 accumulator fragments, so every A fragment it loads from
// shared memory is reused 2x and every B fragment 4x.
// Global->shared copies use 16-byte (8 x half) vector loads.
// Shared rows are padded by 8 halfs to break bank conflicts while keeping
// the 32-byte fragment alignment WMMA needs.
// Requires M % 128 == 0, N % 128 == 0, K % 32 == 0.
// ---------------------------------------------------------------------------
namespace tc1 {
constexpr int BM = 128, BN = 128, BK = 32, PAD = 8;
constexpr int WARPS_M = 2, WARPS_N = 4;          // 8 warps
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64 x 32 per warp
constexpr int FM = WM / 16, FN = WN / 16;          // 4 x 2 fragments
constexpr int THREADS = WARPS_M * WARPS_N * 32;    // 256
}  // namespace tc1

template <bool SWIZZLE>
__global__ void __launch_bounds__(tc1::THREADS)
hgemm_wmma_smem(int M, int N, int K, const half* A, const half* B, float* C) {
  using namespace tc1;
  __shared__ __align__(32) half As[BM][BK + PAD];
  __shared__ __align__(32) half Bs[BK][BN + PAD];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int wr = warp / WARPS_N;  // 0..1
  const int wc = warp % WARPS_N;  // 0..3
  int bm, bn;
  block_tile_origin<SWIZZLE>(BM, BN, bm, bn);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[FN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile: 128 x 32 halfs = 512 int4 vectors, 2 per thread (4 per row).
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * THREADS;
      const int row = v / (BK / 8), col = (v % (BK / 8)) * 8;
      *reinterpret_cast<int4*>(&As[row][col]) =
          *reinterpret_cast<const int4*>(&A[(bm + row) * K + k0 + col]);
    }
    // B tile: 32 x 128 halfs = 512 int4 vectors, 2 per thread (16 per row).
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * THREADS;
      const int row = v / (BN / 8), col = (v % (BN / 8)) * 8;
      *reinterpret_cast<int4*>(&Bs[row][col]) =
          *reinterpret_cast<const int4*>(&B[(k0 + row) * N + bn + col]);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &As[wr * WM + i * 16][kk], BK + PAD);
#pragma unroll
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &Bs[kk][wc * WN + j * 16], BN + PAD);
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      const int r = bm + wr * WM + i * 16;
      const int c = bn + wc * WN + j * 16;
      wmma::store_matrix_sync(C + r * N + c, acc[i][j], N, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// TC-2: double-buffered WMMA (software pipelining).
// Same tiling as TC-1, but two shared-memory buffers. While the tensor cores
// work on tile k from buffer `buf`, each thread already has tile k+1 in flight
// from global memory into registers; it is written to buffer `buf ^ 1` after
// the MMAs, so global-load latency hides behind compute. One __syncthreads per
// k-step instead of two. (T4 / sm_75 has no cp.async, so the prefetch goes
// through registers.)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(tc1::THREADS)
hgemm_wmma_dbuf(int M, int N, int K, const half* A, const half* B, float* C) {
  using namespace tc1;
  __shared__ __align__(32) half As[2][BM][BK + PAD];
  __shared__ __align__(32) half Bs[2][BK][BN + PAD];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int wr = warp / WARPS_N;
  const int wc = warp % WARPS_N;
  const int bm = blockIdx.y * BM;
  const int bn = blockIdx.x * BN;

  // Per-thread load slots: 2 int4 of A and 2 int4 of B per k-step.
  int aRow[2], aCol[2], bRow[2], bCol[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v = tid + i * THREADS;
    aRow[i] = v / (BK / 8); aCol[i] = (v % (BK / 8)) * 8;
    bRow[i] = v / (BN / 8); bCol[i] = (v % (BN / 8)) * 8;
  }
  int4 ra[2], rb[2];
  auto load_global = [&](int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ra[i] = *reinterpret_cast<const int4*>(&A[(bm + aRow[i]) * K + k0 + aCol[i]]);
      rb[i] = *reinterpret_cast<const int4*>(&B[(k0 + bRow[i]) * N + bn + bCol[i]]);
    }
  };
  auto store_shared = [&](int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      *reinterpret_cast<int4*>(&As[buf][aRow[i]][aCol[i]]) = ra[i];
      *reinterpret_cast<int4*>(&Bs[buf][bRow[i]][bCol[i]]) = rb[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[FN];

  // Prologue: tile 0 -> buffer 0.
  load_global(0);
  store_shared(0);
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool has_next = k0 + BK < K;
    if (has_next) load_global(k0 + BK);  // in flight during the MMAs below

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &As[buf][wr * WM + i * 16][kk], BK + PAD);
#pragma unroll
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &Bs[buf][kk][wc * WN + j * 16], BN + PAD);
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }

    // buf^1 was last read in the previous k-step, which every warp finished
    // before the previous __syncthreads, so it is safe to overwrite now.
    if (has_next) store_shared(buf ^ 1);
    __syncthreads();
    buf ^= 1;
  }

#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      const int r = bm + wr * WM + i * 16;
      const int c = bn + wc * WN + j * 16;
      wmma::store_matrix_sync(C + r * N + c, acc[i][j], N, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
inline bool launch_wmma_naive(int M, int N, int K, const half* A,
                              const half* B, float* C) {
  if (M % 16 || N % 16 || K % 16) return false;
  dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
  hgemm_wmma_naive<<<grid, 128>>>(M, N, K, A, B, C);
  return true;
}

inline bool launch_wmma_smem(int M, int N, int K, const half* A,
                             const half* B, float* C) {
  if (M % tc1::BM || N % tc1::BN || K % tc1::BK) return false;
  dim3 grid(N / tc1::BN, M / tc1::BM);
  hgemm_wmma_smem<false><<<grid, tc1::THREADS>>>(M, N, K, A, B, C);
  return true;
}

// TC-3: TC-1 with grouped (swizzled) block order for L2 reuse.
inline bool launch_wmma_swizzle(int M, int N, int K, const half* A,
                                const half* B, float* C) {
  if (M % tc1::BM || N % tc1::BN || K % tc1::BK) return false;
  dim3 grid(N / tc1::BN, M / tc1::BM);
  hgemm_wmma_smem<true><<<grid, tc1::THREADS>>>(M, N, K, A, B, C);
  return true;
}

inline bool launch_wmma_dbuf(int M, int N, int K, const half* A,
                             const half* B, float* C) {
  if (M % tc1::BM || N % tc1::BN || K % tc1::BK) return false;
  dim3 grid(N / tc1::BN, M / tc1::BM);
  hgemm_wmma_dbuf<<<grid, tc1::THREADS>>>(M, N, K, A, B, C);
  return true;
}

// cuBLAS FP16 inputs, FP32 accumulate + output, tensor cores. Row-major trick
// as in SGEMM: compute C^T = B^T * A^T in column-major.
inline void launch_cublas_fp16(cublasHandle_t h, int M, int N, int K,
                               const half* A, const half* B, float* C) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                            CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}
