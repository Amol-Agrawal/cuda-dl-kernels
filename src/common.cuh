// common.cuh - error checking, timing, init, verification helpers
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
              __FILE__, __LINE__);                                            \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,       \
              __LINE__);                                                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Uniform random in [-1, 1], fixed seed for reproducibility.
inline void fill_random(std::vector<float>& v, unsigned seed) {
  srand(seed);
  for (auto& x : v) x = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
}

// Max relative error vs reference. Different summation order => small FP32 diffs.
inline float max_rel_err(const std::vector<float>& out,
                         const std::vector<float>& ref) {
  float worst = 0.f;
  for (size_t i = 0; i < out.size(); ++i) {
    float e = fabsf(out[i] - ref[i]) / (1.0f + fabsf(ref[i]));
    worst = std::max(worst, e);
  }
  return worst;
}

// Times `fn` (which launches GPU work) with CUDA events.
// Returns the median of `iters` runs in milliseconds, after `warmup` runs.
template <typename F>
float time_ms(F fn, int warmup = 3, int iters = 10) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> t(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&t[i], start, stop));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(t.begin(), t.end());
  return t[iters / 2];
}
