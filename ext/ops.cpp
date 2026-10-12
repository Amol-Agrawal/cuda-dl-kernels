// ops.cpp - PyTorch bindings: dlk.sgemm, dlk.hgemm, dlk.softmax
#include <torch/extension.h>

const char* dlk_sgemm(int variant, int M, int N, int K, const float* A,
                      const float* B, float* C);
const char* dlk_hgemm(int variant, int M, int N, int K, const void* A,
                      const void* B, float* C);
const char* dlk_softmax(int fused, const float* x, float* y, float* scratch,
                        int rows, int cols);

static void check_matrix(const torch::Tensor& t, torch::ScalarType dt, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(t.scalar_type() == dt, name, " has the wrong dtype");
}

// C = A @ B, FP32. variant 0-4 selects the optimization stage.
torch::Tensor sgemm(torch::Tensor A, torch::Tensor B, int64_t variant) {
  check_matrix(A, torch::kFloat32, "A");
  check_matrix(B, torch::kFloat32, "B");
  TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions differ");
  A = A.contiguous();
  B = B.contiguous();
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::zeros({M, N}, A.options());  // kernels compute alpha*AB + beta*C
  const char* err = dlk_sgemm((int)variant, M, N, K, A.data_ptr<float>(),
                              B.data_ptr<float>(), C.data_ptr<float>());
  TORCH_CHECK(err == nullptr, "sgemm: ", err ? err : "");
  return C;
}

// C = A @ B with FP16 inputs on tensor cores, FP32 accumulate and output.
torch::Tensor hgemm(torch::Tensor A, torch::Tensor B, int64_t variant) {
  check_matrix(A, torch::kFloat16, "A");
  check_matrix(B, torch::kFloat16, "B");
  TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions differ");
  A = A.contiguous();
  B = B.contiguous();
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
  const char* err = dlk_hgemm((int)variant, M, N, K, A.data_ptr(), B.data_ptr(),
                              C.data_ptr<float>());
  TORCH_CHECK(err == nullptr, "hgemm: ", err ? err : "");
  return C;
}

// Row-wise softmax of a 2-D FP32 tensor. fused=true -> single-pass kernel.
torch::Tensor softmax(torch::Tensor x, bool fused) {
  check_matrix(x, torch::kFloat32, "x");
  x = x.contiguous();
  const int rows = x.size(0), cols = x.size(1);
  auto y = torch::empty_like(x);
  auto scratch = torch::empty({2 * rows}, x.options());
  const char* err = dlk_softmax(fused ? 1 : 0, x.data_ptr<float>(), y.data_ptr<float>(),
                                scratch.data_ptr<float>(), rows, cols);
  TORCH_CHECK(err == nullptr, "softmax: ", err ? err : "");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgemm", &sgemm, "FP32 GEMM (variant 0-4)", py::arg("A"), py::arg("B"), py::arg("variant") = 4);
  m.def("hgemm", &hgemm, "FP16 tensor-core GEMM (variant 0-3)", py::arg("A"), py::arg("B"), py::arg("variant") = 2);
  m.def("softmax", &softmax, "row-wise softmax", py::arg("x"), py::arg("fused") = true);
}
