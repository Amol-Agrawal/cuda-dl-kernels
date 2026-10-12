// hgemm.cu - benchmark + correctness harness for the WMMA tensor-core track.
// Usage: ./hgemm [size ...]        (default: 1024 2048 4096)
// FP16 inputs, FP32 accumulate/output. Reference = cuBLAS GemmEx (tensor cores).
#include <string>
#include <functional>
#include "kernels_wmma.cuh"

struct TcKernel {
  std::string name;
  std::function<bool(int, int, int, const half*, const half*, float*)> run;
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
  if (prop.major < 7) {
    fprintf(stderr, "WMMA needs sm_70+. On Kaggle pick 'GPU T4 x2', not P100.\n");
    return 1;
  }

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  std::vector<TcKernel> kernels = {
      {"tc0_wmma_naive", launch_wmma_naive},
      {"tc1_wmma_smem", launch_wmma_smem},
      {"tc2_wmma_dbuf", launch_wmma_dbuf},
      {"tc3_wmma_swizzle", launch_wmma_swizzle},
  };

  FILE* csv = fopen("results/hgemm.csv", "a");
  printf("%-16s %6s %10s %10s %9s %10s\n", "kernel", "size", "ms", "TFLOPS",
         "%cuBLAS", "max_err");

  for (int n : sizes) {
    const int M = n, N = n, K = n;
    const double flops = 2.0 * M * N * K;

    std::vector<float> fA(M * K), fB(K * N);
    fill_random(fA, 1);
    fill_random(fB, 2);
    std::vector<half> hA(M * K), hB(K * N);
    for (size_t i = 0; i < fA.size(); ++i) hA[i] = __float2half(fA[i]);
    for (size_t i = 0; i < fB.size(); ++i) hB[i] = __float2half(fB[i]);
    std::vector<float> hRef(M * N), hOut(M * N);

    half *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(half) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(half) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(half) * K * N, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * M * N));
    launch_cublas_fp16(handle, M, N, K, dA, dB, dC);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    float ms_ref = time_ms([&] { launch_cublas_fp16(handle, M, N, K, dA, dB, dC); });
    double tf_ref = flops / (ms_ref * 1e9);
    printf("%-16s %6d %10.3f %10.2f %8.1f%% %10s\n", "cublas_fp16", n, ms_ref,
           tf_ref, 100.0, "-");
    if (csv) fprintf(csv, "%s,cublas_fp16,%d,%.4f,%.3f,100.0,0\n", prop.name, n,
                     ms_ref, tf_ref);

    for (auto& kern : kernels) {
      CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * M * N));
      if (!kern.run(M, N, K, dA, dB, dC)) {
        printf("%-16s %6d  skipped (size not a multiple of the tile)\n",
               kern.name.c_str(), n);
        continue;
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(hOut.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
      float err = max_rel_err(hOut, hRef);

      float ms = time_ms([&] { kern.run(M, N, K, dA, dB, dC); });
      double tf = flops / (ms * 1e9);
      printf("%-16s %6d %10.3f %10.2f %8.1f%% %10.2e%s\n", kern.name.c_str(),
             n, ms, tf, 100.0 * tf / tf_ref, err,
             err > 1e-2f ? "  <-- WRONG" : "");
      if (csv) fprintf(csv, "%s,%s,%d,%.4f,%.3f,%.1f,%.2e\n", prop.name,
                       kern.name.c_str(), n, ms, tf, 100.0 * tf / tf_ref, err);
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  if (csv) fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
