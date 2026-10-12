// kernels_softmax.cuh - row-wise softmax over a (rows x cols) row-major matrix.
// y[r][c] = exp(x[r][c] - max_r) / sum_c' exp(x[r][c'] - max_r)
#pragma once
#include "common.cuh"

constexpr int SM_THREADS = 256;  // one block per row
constexpr int SM_WARPS = SM_THREADS / 32;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Block-wide reductions: shuffle inside each warp, then warp 0 reduces the
// per-warp partials. `sh` must hold SM_WARPS floats. Every thread gets the result.
__device__ __forceinline__ float block_max(float v, float* sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0) sh[w] = v;
  __syncthreads();
  if (w == 0) {
    v = lane < SM_WARPS ? sh[lane] : -INFINITY;
    v = warp_max(v);
    if (lane == 0) sh[0] = v;
  }
  __syncthreads();
  v = sh[0];
  __syncthreads();  // sh is reused by the next reduction
  return v;
}
__device__ __forceinline__ float block_sum(float v, float* sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[w] = v;
  __syncthreads();
  if (w == 0) {
    v = lane < SM_WARPS ? sh[lane] : 0.f;
    v = warp_sum(v);
    if (lane == 0) sh[0] = v;
  }
  __syncthreads();
  v = sh[0];
  __syncthreads();
  return v;
}

// ---------------------------------------------------------------------------
// Naive: 3 kernels, 3 full passes over x in global memory.
//   1) row max (shared-memory tree reduction)   reads x
//   2) row sum of exp(x - max)                  reads x
//   3) normalize                                reads x, writes y
// Traffic ~ 4 * rows * cols * 4 bytes.
// ---------------------------------------------------------------------------
__global__ void softmax_rowmax(const float* x, float* rowmax, int cols) {
  __shared__ float sh[SM_THREADS];
  const float* xr = x + (size_t)blockIdx.x * cols;
  float m = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += SM_THREADS) m = fmaxf(m, xr[c]);
  sh[threadIdx.x] = m;
  __syncthreads();
  for (int s = SM_THREADS / 2; s > 0; s >>= 1) {  // classic tree reduction
    if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) rowmax[blockIdx.x] = sh[0];
}

__global__ void softmax_rowsum(const float* x, const float* rowmax,
                               float* rowsum, int cols) {
  __shared__ float sh[SM_THREADS];
  const float* xr = x + (size_t)blockIdx.x * cols;
  const float m = rowmax[blockIdx.x];
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += SM_THREADS) s += expf(xr[c] - m);
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int st = SM_THREADS / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) rowsum[blockIdx.x] = sh[0];
}

__global__ void softmax_normalize(const float* x, const float* rowmax,
                                  const float* rowsum, float* y, int cols) {
  const size_t base = (size_t)blockIdx.x * cols;
  const float m = rowmax[blockIdx.x];
  const float inv = 1.f / rowsum[blockIdx.x];
  for (int c = threadIdx.x; c < cols; c += SM_THREADS)
    y[base + c] = expf(x[base + c] - m) * inv;
}

// ---------------------------------------------------------------------------
// Fused: 1 kernel, 1 read of x + 1 write of y.
// The whole row is held in registers (VPT values per thread, coalesced
// stride-SM_THREADS layout), max and sum use warp-shuffle block reductions,
// and the exp is computed once and reused for the output.
// Traffic ~ 2 * rows * cols * 4 bytes. Supports cols <= VPT * SM_THREADS.
// ---------------------------------------------------------------------------
template <int VPT>
__global__ void __launch_bounds__(SM_THREADS)
softmax_fused(const float* x, float* y, int cols) {
  __shared__ float sh[SM_WARPS];
  const size_t base = (size_t)blockIdx.x * cols;
  float v[VPT];
  float m = -INFINITY;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = threadIdx.x + i * SM_THREADS;
    v[i] = c < cols ? x[base + c] : -INFINITY;
    m = fmaxf(m, v[i]);
  }
  m = block_max(m, sh);

  float s = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = threadIdx.x + i * SM_THREADS;
    v[i] = c < cols ? expf(v[i] - m) : 0.f;
    s += v[i];
  }
  s = block_sum(s, sh);

  const float inv = 1.f / s;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = threadIdx.x + i * SM_THREADS;
    if (c < cols) y[base + c] = v[i] * inv;
  }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
// scratch must hold 2 * rows floats (row max + row sum).
inline void launch_softmax_naive(const float* x, float* y, float* scratch,
                                 int rows, int cols) {
  float* rowmax = scratch;
  float* rowsum = scratch + rows;
  softmax_rowmax<<<rows, SM_THREADS>>>(x, rowmax, cols);
  softmax_rowsum<<<rows, SM_THREADS>>>(x, rowmax, rowsum, cols);
  softmax_normalize<<<rows, SM_THREADS>>>(x, rowmax, rowsum, y, cols);
}

// Returns false if cols is too wide to keep a row in registers.
inline bool launch_softmax_fused(const float* x, float* y, int rows, int cols) {
  const int vpt = ceil_div(cols, SM_THREADS);
  if (vpt <= 1)       softmax_fused<1><<<rows, SM_THREADS>>>(x, y, cols);
  else if (vpt <= 2)  softmax_fused<2><<<rows, SM_THREADS>>>(x, y, cols);
  else if (vpt <= 4)  softmax_fused<4><<<rows, SM_THREADS>>>(x, y, cols);
  else if (vpt <= 8)  softmax_fused<8><<<rows, SM_THREADS>>>(x, y, cols);
  else if (vpt <= 16) softmax_fused<16><<<rows, SM_THREADS>>>(x, y, cols);
  else return false;
  return true;
}
