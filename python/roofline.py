"""Analytical performance model + roofline for the GEMM kernels.

For each kernel we derive its global-memory traffic from the tiling (assuming
no cache reuse beyond what the kernel stages itself), giving an arithmetic
intensity AI = FLOPs / bytes. The roofline then predicts a ceiling:
    predicted = min(peak_compute, AI * peak_bandwidth)
and we compare it with the measured number from results/*.csv.

Run: python python/roofline.py  -> table on stdout, results/roofline.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# T4 datasheet peaks. Sustained is lower: the card is power-capped at 70 W.
PEAK_FP32 = 8100.0      # GFLOPS
PEAK_FP16_TC = 65000.0  # GFLOPS, tensor cores, FP32 accumulate
PEAK_BW = 320.0         # GB/s

# Arithmetic intensity (FLOP / byte of global loads) derived from each tiling.
#   2-flop FMA; a C tile of BM x BN reloads (BM + BN) * K elements of A and B.
#   AI = 2 / (bytes_per_elem * (1/BM + 1/BN))
def tile_ai(bm, bn, bytes_per_elem):
    return 2.0 / (bytes_per_elem * (1.0 / bm + 1.0 / bn))

MODEL = {
    # kernel: (track, AI, note)
    "0_naive":        ("FP32", 2.0 / 8.0, "2 global loads per FMA, uncoalesced"),
    "1_coalesced":    ("FP32", 2.0 / 8.0, "2 loads per FMA, but coalesced + L1 broadcast"),
    "2_smem_tiling":  ("FP32", tile_ai(32, 32, 4), "32x32 tile, then shared-memory bound"),
    "3_reg_blocking": ("FP32", tile_ai(128, 128, 4), "128x128 tile, 8x8 per thread"),
    "4_vec4":         ("FP32", tile_ai(128, 128, 4), "same tile, 128-bit loads"),
    "tc0_wmma_naive": ("FP16", tile_ai(16, 16, 2), "one 16x16 tile per warp, no reuse"),
    "tc1_wmma_smem":  ("FP16", tile_ai(128, 128, 2), "128x128 block tile in shared memory"),
    "tc2_wmma_dbuf":  ("FP16", tile_ai(128, 128, 2), "same tile, global loads overlapped"),
    "tc3_wmma_swizzle": ("FP16", tile_ai(128, 128, 2), "TC1 + grouped block order for L2 reuse"),
}

cols = ["gpu", "kernel", "size", "ms", "perf", "pct_of_cublas", "max_err"]
frames = []
if os.path.exists("results/sgemm.csv"):
    frames.append(pd.read_csv("results/sgemm.csv", names=cols))
if os.path.exists("results/hgemm.csv"):
    h = pd.read_csv("results/hgemm.csv", names=cols)
    h["perf"] = h["perf"] * 1000.0  # TFLOPS -> GFLOPS
    frames.append(h)
df = pd.concat(frames).groupby(["kernel", "size"]).tail(1)
size = df["size"].max()
df = df[df["size"] == size]

rows = []
for k, (track, ai, note) in MODEL.items():
    m = df[df.kernel == k]
    if m.empty:
        continue
    peak = PEAK_FP32 if track == "FP32" else PEAK_FP16_TC
    pred = min(peak, ai * PEAK_BW)
    bound = "memory" if ai * PEAK_BW < peak else "compute"
    meas = float(m.perf.iloc[0])
    rows.append(dict(kernel=k, track=track, AI=round(ai, 2), bound=bound,
                     predicted_gflops=round(pred), measured_gflops=round(meas),
                     pct_of_model=round(100 * meas / pred, 1), note=note))
table = pd.DataFrame(rows)
print(f"Performance model vs measured, {size}^3 GEMM on T4")
print(f"Ridge points: FP32 {PEAK_FP32 / PEAK_BW:.0f} FLOP/B, FP16 tensor {PEAK_FP16_TC / PEAK_BW:.0f} FLOP/B")
print(table.to_string(index=False))
print("pct_of_model > 100 means L1/L2 caches supply reuse the kernel does not stage itself;"
      " < 100 means a bottleneck other than global traffic (shared-memory bandwidth, latency,"
      " occupancy). Confirm which with Nsight Compute.")
table.to_csv("results/perf_model.csv", index=False)

# ---- roofline plot ----
COLORS = {"FP32": "#2a78d6", "FP16": "#eb6834"}  # categorical slots 1, 2
MARKERS = {"FP32": "o", "FP16": "s"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
xs = [0.1, 1000.0]
for peak, label in [(PEAK_FP32, "FP32 peak 8.1 TFLOPS"), (PEAK_FP16_TC, "FP16 tensor peak 65 TFLOPS")]:
    ridge = peak / PEAK_BW
    ax.plot([xs[0], ridge, xs[1]], [xs[0] * PEAK_BW, peak, peak], color=MUTED, lw=1.5)
    ax.text(xs[1], peak * 1.12, label, ha="right", va="bottom", fontsize=8, color=MUTED)
ax.text(0.5, 0.5 * PEAK_BW * 1.4, "DRAM 320 GB/s", rotation=33, fontsize=8, color=MUTED)

for track in ["FP32", "FP16"]:
    t = table[table.track == track]
    ax.scatter(t.AI, t.measured_gflops, s=64, marker=MARKERS[track], color=COLORS[track],
               edgecolors="white", linewidths=1.5, zorder=3, label=f"{track} kernels (measured)")
    # Kernels with the same AI stack vertically; offset labels so they don't collide.
    seen = {}
    for _, r in t.iterrows():
        n = seen.get(r.AI, 0)
        seen[r.AI] = n + 1
        ax.annotate(r.kernel, (r.AI, r.measured_gflops), xytext=(8, -4 + 10 * n),
                    textcoords="offset points", fontsize=7.5, color=INK)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*xs)
ax.set_ylim(10, 120000)
ax.set_xlabel("Arithmetic intensity (FLOP / byte of global loads)", color=INK)
ax.set_ylabel("GFLOPS", color=INK)
ax.set_title(f"Roofline: SGEMM ladder and WMMA GEMM on T4 ({size}^3)", color=INK, loc="left")
ax.grid(True, which="major", color=GRID, lw=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("results/roofline.png")
print("saved results/roofline.png")
