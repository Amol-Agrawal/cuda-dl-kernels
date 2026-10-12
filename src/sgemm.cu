// sgemm.cu - benchmark + correctness harness for the SGEMM ladder.
// Usage: ./sgemm [size ...]        (default: 1024 2048 4096)
// Prints a CSV row per (kernel, size) and appends it to results/sgemm.csv.
#include <string>
#include <functional>
#include "kernels.cuh"

struct Kernel {
  std::string name;
  std::function<bool(int, int, int, float, const float*, const float*, float,
                     float*)> run;
};

int main(int argc, char** argv) {
  std::vector<int> sizes = {1024, 2048, 4096};
  if (argc > 1) {
    sizes.clear();
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s | SMs: %d | sm_%d%d\n", prop.name, prop.multiProcessorCount,
         prop.major, prop.minor);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // Keep cuBLAS on plain FP32 (no TF32 / tensor-core shortcuts) for a fair baseline.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  std::vector<Kernel> kernels = {
      {"0_naive", launch_naive},
      {"1_coalesced", launch_coalesced},
      {"2_smem_tiling", launch_smem},
      {"3_reg_blocking", launch_regblock},
      {"4_vec4", launch_vec4},
  };

  FILE* csv = fopen("results/sgemm.csv", "a");
  printf("%-15s %6s %10s %10s %9s %10s\n", "kernel", "size", "ms", "GFLOPS",
         "%cuBLAS", "max_err");

  for (int n : sizes) {
    const int M = n, N = n, K = n;
    const float alpha = 1.f, beta = 0.f;
    const double flops = 2.0 * M * N * K;
    const size_t bytesA = sizeof(float) * M * K, bytesB = sizeof(float) * K * N,
                 bytesC = sizeof(float) * M * N;

    std::vector<float> hA(M * K), hB(K * N), hRef(M * N), hOut(M * N);
    fill_random(hA, 1);
    fill_random(hB, 2);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    // Reference + baseline: cuBLAS.
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    launch_cublas(handle, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    float ms_cublas = time_ms(
        [&] { launch_cublas(handle, M, N, K, alpha, dA, dB, beta, dC); });
    double gf_cublas = flops / (ms_cublas * 1e6);
    printf("%-15s %6d %10.3f %10.1f %8.1f%% %10s\n", "cublas", n, ms_cublas,
           gf_cublas, 100.0, "-");
    if (csv) fprintf(csv, "%s,cublas,%d,%.4f,%.1f,100.0,0\n", prop.name, n,
                     ms_cublas, gf_cublas);

    for (auto& kern : kernels) {
      // Naive at 4096 takes seconds per run; fewer iters keeps it bearable.
      int iters = (kern.name == "0_naive" && n >= 4096) ? 3 : 10;
      CUDA_CHECK(cudaMemset(dC, 0, bytesC));
      if (!kern.run(M, N, K, alpha, dA, dB, beta, dC)) {
        printf("%-15s %6d  skipped (size not a multiple of the tile)\n",
               kern.name.c_str(), n);
        continue;
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytesC, cudaMemcpyDeviceToHost));
      float err = max_rel_err(hOut, hRef);

      float ms = time_ms(
          [&] { kern.run(M, N, K, alpha, dA, dB, beta, dC); }, 1, iters);
      double gf = flops / (ms * 1e6);
      printf("%-15s %6d %10.3f %10.1f %8.1f%% %10.2e%s\n", kern.name.c_str(),
             n, ms, gf, 100.0 * gf / gf_cublas, err,
             err > 1e-3f ? "  <-- WRONG" : "");
      if (csv) fprintf(csv, "%s,%s,%d,%.4f,%.1f,%.1f,%.2e\n", prop.name,
                       kern.name.c_str(), n, ms, gf, 100.0 * gf / gf_cublas,
                       err);
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  if (csv) fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
