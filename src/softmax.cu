// softmax.cu - correctness + benchmark harness: naive 3-kernel vs fused softmax.
// Usage: ./softmax                (default shapes, ~8M elements each)
// Appends to results/softmax.csv. Reports effective bandwidth (GB/s) against the
// minimum traffic of the op (read x once + write y once).
#include <string>
#include "kernels_softmax.cuh"

int main() {
  // (rows, cols): attention-like shapes, rows = batch * heads * seq_len.
  const int shapes[][2] = {{16384, 512}, {8192, 1024}, {4096, 2048}, {2048, 4096}};

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);

  FILE* csv = fopen("results/softmax.csv", "a");
  printf("%-14s %6s %6s %9s %9s %8s %10s\n", "kernel", "rows", "cols", "ms",
         "GB/s", "speedup", "max_err");

  for (auto& sh : shapes) {
    const int rows = sh[0], cols = sh[1];
    const size_t n = (size_t)rows * cols;
    std::vector<float> hx(n), hy(n), href(n);
    fill_random(hx, 7);
    for (auto& v : hx) v *= 8.f;  // logits in [-8, 8]

    // CPU reference in double precision.
    for (int r = 0; r < rows; ++r) {
      const float* xr = &hx[(size_t)r * cols];
      double m = -1e300, s = 0;
      for (int c = 0; c < cols; ++c) m = std::max(m, (double)xr[c]);
      for (int c = 0; c < cols; ++c) s += exp((double)xr[c] - m);
      for (int c = 0; c < cols; ++c)
        href[(size_t)r * cols + c] = (float)(exp((double)xr[c] - m) / s);
    }

    float *dx, *dy, *dscratch;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dscratch, 2 * rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const double min_bytes = 2.0 * n * sizeof(float);

    auto check = [&]() {
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
      float e = 0.f;  // absolute error: outputs are probabilities in [0, 1]
      for (size_t i = 0; i < n; ++i) e = std::max(e, fabsf(hy[i] - href[i]));
      return e;
    };

    // Naive
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    launch_softmax_naive(dx, dy, dscratch, rows, cols);
    const float err_naive = check();
    const float ms_naive = time_ms([&] { launch_softmax_naive(dx, dy, dscratch, rows, cols); });
    printf("%-14s %6d %6d %9.4f %9.1f %8s %10.2e%s\n", "naive_3kernel", rows, cols,
           ms_naive, min_bytes / (ms_naive * 1e6), "1.00x", err_naive,
           err_naive > 1e-5f ? "  <-- WRONG" : "");
    if (csv) fprintf(csv, "%s,naive_3kernel,%d,%d,%.5f,%.1f,1.00,%.2e\n", prop.name,
                     rows, cols, ms_naive, min_bytes / (ms_naive * 1e6), err_naive);

    // Fused
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    if (!launch_softmax_fused(dx, dy, rows, cols)) {
      printf("%-14s %6d %6d  skipped (row too wide for registers)\n", "fused", rows, cols);
    } else {
      const float err_fused = check();
      const float ms_fused = time_ms([&] { launch_softmax_fused(dx, dy, rows, cols); });
      printf("%-14s %6d %6d %9.4f %9.1f %7.2fx %10.2e%s\n", "fused", rows, cols,
             ms_fused, min_bytes / (ms_fused * 1e6), ms_naive / ms_fused, err_fused,
             err_fused > 1e-5f ? "  <-- WRONG" : "");
      if (csv) fprintf(csv, "%s,fused,%d,%d,%.5f,%.1f,%.2f,%.2e\n", prop.name, rows,
                       cols, ms_fused, min_bytes / (ms_fused * 1e6),
                       ms_naive / ms_fused, err_fused);
    }

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dscratch));
  }
  if (csv) fclose(csv);
  return 0;
}
