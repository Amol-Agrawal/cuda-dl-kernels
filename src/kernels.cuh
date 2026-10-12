// kernels.cuh - SGEMM optimization ladder.
// All matrices are row-major. C = alpha * A(MxK) * B(KxN) + beta * C(MxN).
#pragma once
#include "common.cuh"

// ---------------------------------------------------------------------------
// Stage 0: naive, UNCOALESCED.
// threadIdx.x walks down ROWS, so the 32 threads of a warp read A and write C
// with a stride of K / N floats -> each access is its own memory transaction.
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ---------------------------------------------------------------------------
// Stage 1: global memory coalescing.
// Same math; threadIdx.x now walks along COLUMNS, so a warp reads 32
// consecutive floats of B / writes 32 consecutive floats of C (one 128B line),
// and all 32 threads read the SAME A element (a broadcast).
// ---------------------------------------------------------------------------
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta,
                                float* C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ---------------------------------------------------------------------------
// Stage 2: shared-memory tiling.
// A 32x32 block loads a 32x32 tile of A and of B into shared memory, then
// every thread does 32 FMAs out of shared memory. Each global element is now
// loaded once per tile instead of once per FMA -> global traffic / 32.
// Inner loop: As[ty][k] is a broadcast (one ty per warp), Bs[k][tx] hits 32
// consecutive banks -> no bank conflicts.
// Bottleneck left: 2 shared loads per FMA, so it becomes shared-memory bound.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ---------------------------------------------------------------------------
// Stage 3: 2D register blocking.
// Block tile BM x BN, K-step BK, each thread computes a TM x TN micro-tile
// held in registers. Per k-step a thread loads TM values of A and TN of B into
// registers and does TM*TN FMAs: 16 shared loads feed 64 FMAs (was 2 per 1).
// Requires M % BM == 0, N % BN == 0, K % BK == 0.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_regblock(int M, int N, int K, float alpha, const float* A,
               const float* B, float beta, float* C) {
  constexpr int THREADS = (BM * BN) / (TM * TN);
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int tid = threadIdx.x;
  const int threadRow = tid / (BN / TN);
  const int threadCol = tid % (BN / TN);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Cooperative load mapping (coalesced along the row of each tile).
  const int rowA = tid / BK, colA = tid % BK;
  constexpr int strideA = THREADS / BK;
  const int rowB = tid / BN, colB = tid % BN;
  constexpr int strideB = THREADS / BN;

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += strideA)
      As[(rowA + off) * BK + colA] = A[(rowA + off) * K + colA];
#pragma unroll
    for (int off = 0; off < BK; off += strideB)
      Bs[(rowB + off) * BN + colB] = B[(rowB + off) * N + colB];
    __syncthreads();
    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      float* c = &C[(threadRow * TM + i) * N + threadCol * TN + j];
      *c = alpha * acc[i * TN + j] + beta * *c;
    }
}

// ---------------------------------------------------------------------------
// Stage 4: vectorized float4 loads/stores + transposed A tile.
// Same tiling as stage 3, but:
//  - A and B are read from global as float4 (4x fewer load instructions,
//    128-bit transactions);
//  - the A tile is stored TRANSPOSED in shared memory (As[k][m]) so each
//    thread's TM values for one k are contiguous and can be read as float4;
//  - C is read/written as float4.
// Requires M % BM == 0, N % BN == 0, K % BK == 0, and N, K multiples of 4.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_vec4(int M, int N, int K, float alpha, const float* A, const float* B,
           float beta, float* C) {
  constexpr int THREADS = (BM * BN) / (TM * TN);
  static_assert(THREADS * 4 == BM * BK, "one float4 of A per thread per k-step");
  static_assert(THREADS * 4 == BK * BN, "one float4 of B per thread per k-step");
  __shared__ __align__(16) float As[BK * BM];  // transposed: As[k * BM + m]
  __shared__ __align__(16) float Bs[BK * BN];

  const int tid = threadIdx.x;
  const int threadRow = tid / (BN / TN);
  const int threadCol = tid % (BN / TN);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  const int rowA = tid / (BK / 4), colA = tid % (BK / 4);
  const int rowB = tid / (BN / 4), colB = tid % (BN / 4);

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    const float4 a4 = *reinterpret_cast<const float4*>(&A[rowA * K + colA * 4]);
    As[(colA * 4 + 0) * BM + rowA] = a4.x;
    As[(colA * 4 + 1) * BM + rowA] = a4.y;
    As[(colA * 4 + 2) * BM + rowA] = a4.z;
    As[(colA * 4 + 3) * BM + rowA] = a4.w;
    *reinterpret_cast<float4*>(&Bs[rowB * BN + colB * 4]) =
        *reinterpret_cast<const float4*>(&B[rowB * N + colB * 4]);
    __syncthreads();
    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; i += 4) {
        const float4 t = *reinterpret_cast<const float4*>(&As[k * BM + threadRow * TM + i]);
        regM[i] = t.x; regM[i + 1] = t.y; regM[i + 2] = t.z; regM[i + 3] = t.w;
      }
#pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const float4 t = *reinterpret_cast<const float4*>(&Bs[k * BN + threadCol * TN + j]);
        regN[j] = t.x; regN[j + 1] = t.y; regN[j + 2] = t.z; regN[j + 3] = t.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      float4* cp = reinterpret_cast<float4*>(
          &C[(threadRow * TM + i) * N + threadCol * TN + j]);
      float4 c = *cp;
      c.x = alpha * acc[i * TN + j + 0] + beta * c.x;
      c.y = alpha * acc[i * TN + j + 1] + beta * c.y;
      c.z = alpha * acc[i * TN + j + 2] + beta * c.z;
      c.w = alpha * acc[i * TN + j + 3] + beta * c.w;
      *cp = c;
    }
}

// ---------------------------------------------------------------------------
// Launchers. Return false if the size is not supported by the kernel's tiling.
// ---------------------------------------------------------------------------
namespace rb {  // tile config shared by stages 3 and 4: 256 threads, 8x8 per thread
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int THREADS = (BM * BN) / (TM * TN);
}  // namespace rb

inline bool launch_naive(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

inline bool launch_coalesced(int M, int N, int K, float alpha, const float* A,
                             const float* B, float beta, float* C) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
  sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

inline bool launch_smem(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
  sgemm_smem<32><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

inline bool launch_regblock(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  using namespace rb;
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_regblock<BM, BN, BK, TM, TN><<<grid, THREADS>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

inline bool launch_vec4(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C) {
  using namespace rb;
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_vec4<BM, BN, BK, TM, TN><<<grid, THREADS>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

// cuBLAS is column-major. Row-major C = A*B is column-major C^T = B^T * A^T,
// so pass B first and swap M/N; no transposes needed.
inline void launch_cublas(cublasHandle_t h, int M, int N, int K, float alpha,
                          const float* A, const float* B, float beta,
                          float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                           A, K, &beta, C, N));
}
