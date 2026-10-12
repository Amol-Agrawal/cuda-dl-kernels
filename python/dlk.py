"""JIT-builds and loads the PyTorch extension (first import takes ~1-2 min)."""
import os
from torch.utils.cpp_extension import load

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ops = load(
    name="dlk",
    sources=[os.path.join(_ROOT, "ext", "ops.cpp"), os.path.join(_ROOT, "ext", "ops_cuda.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_75,code=sm_75", "-lineinfo"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)
