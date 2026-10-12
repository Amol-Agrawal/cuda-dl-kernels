"""End-to-end view: where does GPU time go in a real transformer?

Profiles GPT-2 (124M) FP16 inference on the T4 with torch.profiler (CUPTI
tracing, which works without the performance-counter permission Nsight Compute
needs), in two attention modes:
  - eager: explicit QK^T -> softmax -> PV (separate kernels)
  - sdpa:  PyTorch's fused scaled_dot_product_attention
and groups GPU kernel time into GEMM / attention / softmax / layernorm / other.
Then benchmarks our fused softmax against torch.softmax at GPT-2's real
attention-score shape (batch*heads*seq rows x seq cols).

Run: python python/profile_gpt2.py -> tables on stdout, results/gpt2_profile.csv
"""
import csv
import os
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import GPT2LMHeadModel

BATCH, SEQ, HEADS = 8, 1024, 12
dev = "cuda"
os.makedirs("results", exist_ok=True)


def category(name):
    n = name.lower()
    if any(k in n for k in ("fmha", "flash", "attention", "attn", "mem_eff")):
        return "attention (fused)"
    if "softmax" in n:
        return "softmax"
    if any(k in n for k in ("gemm", "cutlass", "s1688", "h1688", "s884", "h884", "sm75_xmma", "matmul")):
        return "GEMM"
    if "layer_norm" in n or "layernorm" in n:
        return "layernorm"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    return "other (elementwise, GELU, masks, ...)"


def self_gpu_us(evt):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0


def load(mode):
    try:
        m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation=mode)
    except Exception as err:  # older transformers: no switch / no SDPA for GPT-2
        print(f"[{mode}] attn_implementation not supported ({err}); using the default")
        m = GPT2LMHeadModel.from_pretrained("gpt2")
    return m.to(dev).half().eval()


ids = torch.randint(0, 50257, (BATCH, SEQ), device=dev)
rows_out = []

for mode in ["eager", "sdpa"]:
    model = load(mode)
    with torch.no_grad():
        for _ in range(3):  # warmup (cuBLAS heuristics, allocator)
            model(ids)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                model(ids)
            torch.cuda.synchronize()

    def is_gpu(e):  # keep only device-side events so nothing is double counted
        dt = getattr(e, "device_type", None)
        return dt is None or "CUDA" in str(dt)
    kernels = [(e.key, self_gpu_us(e)) for e in prof.key_averages() if is_gpu(e) and self_gpu_us(e) > 0]
    total = sum(t for _, t in kernels)
    by_cat = {}
    for name, t in kernels:
        by_cat[category(name)] = by_cat.get(category(name), 0.0) + t

    print(f"\n=== GPT-2 FP16 inference, batch {BATCH} x seq {SEQ}, attention = {mode} ===")
    print(f"GPU time per forward pass: {total / 3 / 1000:.1f} ms")
    for cat, t in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:<40} {100 * t / total:5.1f}%   {t / 3 / 1000:7.2f} ms")
        rows_out.append([mode, cat, round(t / 3 / 1000, 3), round(100 * t / total, 1)])
    print("  top 5 kernels:")
    for name, t in sorted(kernels, key=lambda x: -x[1])[:5]:
        print(f"    {100 * t / total:5.1f}%  {name[:90]}")
    del model
    torch.cuda.empty_cache()

with open("results/gpt2_profile.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attention_mode", "category", "ms_per_forward", "pct"])
    w.writerows(rows_out)

# ---- our softmax at GPT-2's attention-score shape ----
from dlk import ops  # noqa: E402  (builds the extension if needed)


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[iters // 2]


rows = 2 * HEADS * SEQ  # batch 2 keeps the FP32 scores at ~100 MB
x = torch.randn(rows, SEQ, device=dev) * 4
gb = 2 * x.numel() * 4 / 1e9
assert (ops.softmax(x, True) - torch.softmax(x, dim=1)).abs().max().item() < 1e-5
t_torch = bench(lambda: torch.softmax(x, dim=1))
t_ours = bench(lambda: ops.softmax(x, True))
print(f"\n=== Softmax at GPT-2 attention shape ({rows} x {SEQ}, FP32) ===")
print(f"  torch.softmax   {t_torch:.3f} ms  {gb / (t_torch / 1e3):6.1f} GB/s")
print(f"  ours (fused)    {t_ours:.3f} ms  {gb / (t_ours / 1e3):6.1f} GB/s   ({t_torch / t_ours:.2f}x)")
with open("results/gpt2_profile.csv", "a", newline="") as f:
    w = csv.writer(f)
    w.writerow(["softmax_bench", f"torch.softmax {rows}x{SEQ}", round(t_torch, 4), ""])
    w.writerow(["softmax_bench", f"ours fused {rows}x{SEQ}", round(t_ours, 4), ""])
