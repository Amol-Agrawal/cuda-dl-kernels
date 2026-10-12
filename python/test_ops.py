"""Correctness tests of every kernel against PyTorch. Run: python python/test_ops.py"""
import sys
import torch
from dlk import ops

torch.backends.cuda.matmul.allow_tf32 = False  # exact FP32 reference
torch.manual_seed(0)
dev = "cuda"
failures = 0


def report(name, err, tol):
    global failures
    ok = err <= tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<40} max_err={err:.2e} (tol {tol:.0e})")


def rel_err(out, ref):
    return ((out - ref).abs() / (1 + ref.abs())).max().item()


# FP32 SGEMM: square and rectangular shapes (stages 3-4 need multiples of 128).
for (M, N, K) in [(128, 128, 128), (1024, 1024, 1024), (256, 512, 384), (2048, 1024, 512)]:
    A = torch.randn(M, K, device=dev)
    B = torch.randn(K, N, device=dev)
    ref = A @ B
    for v, name in enumerate(["naive", "coalesced", "smem", "regblock", "vec4"]):
        report(f"sgemm[{name}] {M}x{N}x{K}", rel_err(ops.sgemm(A, B, v), ref), 1e-3)

# Odd sizes for the stages that support them.
A = torch.randn(100, 70, device=dev)
B = torch.randn(70, 33, device=dev)
for v, name in enumerate(["naive", "coalesced", "smem"]):
    report(f"sgemm[{name}] 100x33x70 (odd)", rel_err(ops.sgemm(A, B, v), A @ B), 1e-3)

# FP16 tensor-core GEMM. Reference: the same FP16 values multiplied in FP32.
for (M, N, K) in [(128, 128, 128), (1024, 1024, 1024), (256, 512, 384), (1280, 384, 256)]:
    A = torch.randn(M, K, device=dev).half()
    B = torch.randn(K, N, device=dev).half()
    ref = A.float() @ B.float()
    for v, name in enumerate(["wmma_naive", "wmma_smem", "wmma_dbuf", "wmma_swizzle"]):
        report(f"hgemm[{name}] {M}x{N}x{K}", rel_err(ops.hgemm(A, B, v), ref), 1e-2)

# Softmax vs torch.softmax, including widths that are not multiples of 256.
for (rows, cols) in [(4096, 1024), (37, 1000), (512, 4096), (8, 3), (1024, 2050)]:
    x = torch.randn(rows, cols, device=dev) * 8
    ref = torch.softmax(x, dim=1)
    for fused in (False, True):
        out = ops.softmax(x, fused)
        report(f"softmax[{'fused' if fused else 'naive'}] {rows}x{cols}", (out - ref).abs().max().item(), 1e-5)

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
