# cuda-dl-kernels

Optimizing deep learning operators in CUDA on an NVIDIA T4 (Turing, sm_75):

- **FP32 SGEMM** optimization ladder: naive -> coalescing -> shared-memory tiling -> 2D register blocking -> float4 vectorization, vs cuBLAS.
- **FP16 tensor-core GEMM** with WMMA: naive -> shared-memory tiled -> double-buffered (software-pipelined), vs cuBLAS FP16.
- **Fused softmax**: single global pass with warp-shuffle block reductions, vs a naive 3-kernel version and torch.softmax.
- **PyTorch C++/CUDA extension** exposing all kernels, with correctness tests against torch.
- **Analytical performance model + roofline** predicting each kernel's ceiling from its tiling, compared with measurements.

## Layout

```
src/common.cuh           error checks, CUDA-event timing (median), init, verification
src/kernels.cuh          FP32 SGEMM stages 0-4 + cuBLAS FP32 baseline
src/kernels_wmma.cuh     FP16 WMMA GEMMs TC0-TC2 + cuBLAS FP16 baseline
src/kernels_softmax.cuh  naive 3-kernel and fused softmax
src/{sgemm,hgemm,softmax}.cu   C++ benchmark harnesses -> results/*.csv
ext/ops_cuda.cu, ext/ops.cpp   PyTorch extension (dlk.sgemm / dlk.hgemm / dlk.softmax)
python/test_ops.py       correctness vs torch.matmul / torch.softmax
python/bench_torch.py    timing vs torch -> results/torch_compare.csv
python/roofline.py       performance model + roofline -> results/perf_model.csv, roofline.png
cuda_dl_kernels_kaggle.ipynb   runs everything on Kaggle (GPU T4 x2)
```

## Run

```
make ARCH=sm_75
./sgemm 1024 2048 4096 && ./hgemm 1024 2048 4096 && ./softmax
pip install ninja && python python/test_ops.py && python python/bench_torch.py && python python/roofline.py
```

## Kernels

| Stage | Idea | Expected bottleneck after it |
|---|---|---|
| 0 naive | 1 thread per output, threadIdx.x along rows | uncoalesced global loads |
| 1 coalesced | threadIdx.x along columns: 128B lines, A broadcast | global bandwidth / latency |
| 2 smem tiling | 32x32 tiles in shared memory, global traffic / 32 | 2 shared loads per FMA |
| 3 register blocking | 128x128 block tile, 8x8 outputs per thread in registers | shared-memory bank conflicts, instruction count |
| 4 vec4 | float4 global loads/stores, transposed A tile for float4 shared reads | latency (no double buffering) |
| TC0 naive WMMA | 1 warp per 16x16 C tile, fragments from global | global traffic |
| TC1 smem WMMA | 128x128 block tile, 64x32 per warp (4x2 fragments), padded smem | load latency between k-steps |
| TC2 double-buffered | prefetch tile k+1 into registers during MMAs on tile k | - |
| softmax fused | row in registers, warp-shuffle max/sum, 1 read + 1 write | DRAM bandwidth (target) |

## Results (T4) - fill in from results/*.csv

| Kernel | 4096^3 perf | % of cuBLAS |
|---|---|---|
| 0_naive | | |
| 1_coalesced | | |
| 2_smem_tiling | | |
| 3_reg_blocking | | |
| 4_vec4 | | |
| tc0_wmma_naive | | |
| tc1_wmma_smem | | |
| tc2_wmma_dbuf | | |
