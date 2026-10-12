// occupancy.cu - per-kernel resource usage and theoretical occupancy, from the
// CUDA runtime (no performance counters needed, so it works on Kaggle/Colab).
// For each kernel: registers/thread, static shared memory/block, local-memory
// spills, max resident blocks per SM, theoretical occupancy, and the limiter.
// Usage: ./occupancy   -> table on stdout + results/occupancy.csv
#include "kernels.cuh"
#include "kernels_wmma.cuh"
#include "kernels_softmax.cuh"

static cudaDeviceProp prop;
static FILE* csv = nullptr;

template <typename Kern>
void report(const char* name, Kern kernel, int threads) {
  cudaFuncAttributes a;
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));

  // Approximate per-resource limits, to name the limiter.
  const int warps_per_block = (threads + 31) / 32;
  const int regs_per_warp = ((a.numRegs * 32 + 255) / 256) * 256;  // 256-reg allocation unit
  const int by_regs = a.numRegs ? prop.regsPerMultiprocessor / (regs_per_warp * warps_per_block) : 99;
  const int by_smem = a.sharedSizeBytes ? (int)(prop.sharedMemPerMultiprocessor / a.sharedSizeBytes) : 99;
  const int by_threads = prop.maxThreadsPerMultiProcessor / threads;
  const int by_blocks = prop.maxBlocksPerMultiProcessor;
  const char* limiter = "blocks";
  int lim = by_blocks;
  if (by_threads < lim) { lim = by_threads; limiter = "threads"; }
  if (by_regs < lim)    { lim = by_regs;    limiter = "registers"; }
  if (by_smem < lim)    { lim = by_smem;    limiter = "shared mem"; }

  const int max_warps = prop.maxThreadsPerMultiProcessor / 32;
  const double occ = 100.0 * blocks * warps_per_block / max_warps;
  printf("%-18s %7d %6d %9zu %7zu %10d %9.0f%%  %s\n", name, threads, a.numRegs,
         a.sharedSizeBytes, a.localSizeBytes, blocks, occ, limiter);
  if (csv) fprintf(csv, "%s,%d,%d,%zu,%zu,%d,%.1f,%s\n", name, threads, a.numRegs,
                   a.sharedSizeBytes, a.localSizeBytes, blocks, occ, limiter);
}

int main() {
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s | regs/SM %d | smem/SM %zu B | max threads/SM %d | max blocks/SM %d\n\n",
         prop.name, prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor,
         prop.maxThreadsPerMultiProcessor, prop.maxBlocksPerMultiProcessor);
  csv = fopen("results/occupancy.csv", "w");
  if (csv) fprintf(csv, "kernel,threads,regs,smem_bytes,spill_bytes,blocks_per_sm,occupancy_pct,limiter\n");
  printf("%-18s %7s %6s %9s %7s %10s %10s  %s\n", "kernel", "threads", "regs",
         "smem(B)", "spill", "blocks/SM", "occupancy", "limiter");

  using namespace rb;
  report("0_naive", sgemm_naive, 1024);
  report("1_coalesced", sgemm_coalesced, 1024);
  report("2_smem_tiling", sgemm_smem<32>, 1024);
  report("3_reg_blocking", sgemm_regblock<BM, BN, BK, TM, TN>, rb::THREADS);
  report("4_vec4", sgemm_vec4<BM, BN, BK, TM, TN>, rb::THREADS);
  report("tc0_wmma_naive", hgemm_wmma_naive, 128);
  report("tc1_wmma_smem", hgemm_wmma_smem<false>, tc1::THREADS);
  report("tc2_wmma_dbuf", hgemm_wmma_dbuf, tc1::THREADS);
  report("tc3_wmma_swizzle", hgemm_wmma_smem<true>, tc1::THREADS);
  report("softmax_rowmax", softmax_rowmax, SM_THREADS);
  report("softmax_fused<4>", softmax_fused<4>, SM_THREADS);
  report("softmax_fused<16>", softmax_fused<16>, SM_THREADS);

  if (csv) fclose(csv);
  return 0;
}
