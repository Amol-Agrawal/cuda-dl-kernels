"""Benchmark our kernels against PyTorch through the extension.
Run: python python/bench_torch.py   -> prints a table, writes results/torch_compare.csv"""
import csv
import os
import torch
from dlk import ops

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]  # median ms


rows = []
n = 4096
flops = 2 * n**3

A = torch.randn(n, n, device=dev)
B = torch.randn(n, n, device=dev)
t_ref = bench(lambda: A @ B)
t_ours = bench(lambda: ops.sgemm(A, B, 4))
rows.append(["sgemm fp32 4096", "torch.matmul", t_ref, flops / t_ref / 1e9, "TFLOPS"])
rows.append(["sgemm fp32 4096", "ours (vec4)", t_ours, flops / t_ours / 1e9, "TFLOPS"])

Ah, Bh = A.half(), B.half()
t_ref = bench(lambda: Ah @ Bh)
for v, name in [(1, "ours (wmma_smem)"), (2, "ours (wmma_dbuf)"), (3, "ours (wmma_swizzle)")]:
    t = bench(lambda: ops.hgemm(Ah, Bh, v))
    rows.append(["hgemm fp16 4096", name, t, flops / t / 1e9, "TFLOPS"])
rows.append(["hgemm fp16 4096", "torch.matmul (fp16 out)", t_ref, flops / t_ref / 1e9, "TFLOPS"])

for (r, c) in [(16384, 512), (8192, 1024), (4096, 2048), (2048, 4096)]:
    x = torch.randn(r, c, device=dev)
    gb = 2 * x.numel() * 4 / 1e9  # read once + write once
    for name, fn in [("torch.softmax", lambda: torch.softmax(x, dim=1)),
                     ("ours (naive 3-kernel)", lambda: ops.softmax(x, False)),
                     ("ours (fused)", lambda: ops.softmax(x, True))]:
        t = bench(fn)
        rows.append([f"softmax {r}x{c}", name, t, gb / (t / 1e3), "GB/s"])

print(f"{'op':<20} {'impl':<26} {'ms':>9} {'perf':>9}")
for op, impl, t, perf, unit in rows:
    print(f"{op:<20} {impl:<26} {t:>9.3f} {perf:>8.2f} {unit}")

os.makedirs("results", exist_ok=True)
with open("results/torch_compare.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["op", "impl", "ms", "perf", "unit"])
    w.writerows(rows)
