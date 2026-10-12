// ops_cuda.cu - raw-pointer entry points for the PyTorch extension.
// Kept free of torch headers so nvcc only compiles our kernels; ops.cpp does
// the tensor handling. Each function returns nullptr on success, or an error.
#include "../src/kernels.cuh"
#include "../src/kernels_wmma.cuh"
#include "../src/kernels_softmax.cuh"

static const char* last_cuda_error() {
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}

// variant: 0 naive, 1 coalesced, 2 smem tiling, 3 register blocking, 4 vec4
const char* dlk_sgemm(int variant, int M, int N, int K, const float* A,
                      const float* B, float* C) {
  const float alpha = 1.f, beta = 0.f;
  bool ok = false;
  switch (variant) {
    case 0: ok = launch_naive(M, N, K, alpha, A, B, beta, C); break;
    case 1: ok = launch_coalesced(M, N, K, alpha, A, B, beta, C); break;
    case 2: ok = launch_smem(M, N, K, alpha, A, B, beta, C); break;
    case 3: ok = launch_regblock(M, N, K, alpha, A, B, beta, C); break;
    case 4: ok = launch_vec4(M, N, K, alpha, A, B, beta, C); break;
    default: return "unknown sgemm variant (expected 0-4)";
  }
  if (!ok) return "matrix size not supported by this variant (needs multiples of 128)";
  return last_cuda_error();
}

// variant: 0 naive WMMA, 1 smem-tiled WMMA, 2 double-buffered WMMA, 3 swizzled
const char* dlk_hgemm(int variant, int M, int N, int K, const void* A,
                      const void* B, float* C) {
  const half* a = static_cast<const half*>(A);
  const half* b = static_cast<const half*>(B);
  bool ok = false;
  switch (variant) {
    case 0: ok = launch_wmma_naive(M, N, K, a, b, C); break;
    case 1: ok = launch_wmma_smem(M, N, K, a, b, C); break;
    case 2: ok = launch_wmma_dbuf(M, N, K, a, b, C); break;
    case 3: ok = launch_wmma_swizzle(M, N, K, a, b, C); break;
    default: return "unknown hgemm variant (expected 0-3)";
  }
  if (!ok) return "matrix size not supported by this variant (needs multiples of 16 / 128)";
  return last_cuda_error();
}

// fused: 0 naive 3-kernel (needs scratch of 2*rows floats), 1 fused
const char* dlk_softmax(int fused, const float* x, float* y, float* scratch,
                        int rows, int cols) {
  if (fused) {
    if (!launch_softmax_fused(x, y, rows, cols)) return "cols too large for fused kernel (max 4096)";
  } else {
    launch_softmax_naive(x, y, scratch, rows, cols);
  }
  return last_cuda_error();
}
