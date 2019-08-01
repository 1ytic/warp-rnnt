#include <cstring>

#include <tuple>
#include <sstream>
#include <string>

#include <torch/extension.h>
#include "core.h"

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Parallel.h>

#include <THC/THC.h>


#define CHECK_CONTIGUOUS(x)                                     \
  AT_CHECK((x).is_contiguous(),                                 \
           #x " must be contiguous")

#define CHECK_CUDA(x)                                           \
  AT_CHECK((x).device().is_cuda(),                              \
           #x " must be located in the CUDA")

#define CHECK_FLOAT(x)                                          \
  AT_CHECK((x).type().scalarType() == at::ScalarType::Float,    \
           #x " must be a Float tensor")

#define CHECK_INT(x)                                            \
  AT_CHECK((x).type().scalarType() == at::ScalarType::Int,      \
           #x " must be a Int tensor")


std::tuple<at::Tensor, at::Tensor> rnnt_loss(
        const at::Tensor& xs, const at::Tensor& ys,
        const at::Tensor& xn, const at::Tensor& yn,
        const int blank) {
    // Check contiguous
    CHECK_CONTIGUOUS(xs);
    CHECK_CONTIGUOUS(ys);
    CHECK_CONTIGUOUS(xn);
    CHECK_CONTIGUOUS(yn);
    // Check types
    CHECK_FLOAT(xs);
    CHECK_INT(ys);
    CHECK_INT(xn);
    CHECK_INT(yn);
    // Check device
    CHECK_CUDA(xs);
    CHECK_CUDA(ys);
    CHECK_CUDA(xn);
    CHECK_CUDA(yn);
    // Check number of dimensions and elements
    AT_CHECK(xs.dim() == 4, "xs must have 4 dimensions")
    AT_CHECK(xn.numel() == xs.size(0), "xn shape must be equal (N,)")
    AT_CHECK(yn.numel() == xs.size(0), "yn shape must be equal (N,)")
    AT_CHECK(xs.size(2) == ys.size(1) + 1, "ys shape (N, U-1) mismatched with xs (N, T, U, V)")

    const auto N = xs.size(0);
    const auto T = xs.size(1);
    const auto U = xs.size(2);
    const auto V = xs.size(3);

    at::Tensor grads = at::zeros_like(xs);

    at::TensorOptions buffer_opts(xs.device());
    at::TensorOptions counts_opts(xs.device());
    at::TensorOptions costs_opts(xs.device());

    counts_opts = counts_opts.dtype(at::ScalarType::Int);
    buffer_opts = buffer_opts.dtype(at::ScalarType::Float);
    costs_opts = costs_opts.dtype(at::ScalarType::Float);

    auto counts_shape = {N, U * 2};
    auto buffer_shape = {N, T, U};
    auto costs_shape = {N};

    torch::Tensor costs = torch::empty(costs_shape, costs_opts);
    at::Tensor counts = at::zeros(counts_shape, counts_opts);
    at::Tensor alphas = at::empty(buffer_shape, buffer_opts);
    at::Tensor betas = at::empty(buffer_shape, buffer_opts);

    auto stream = at::cuda::getCurrentCUDAStream(xs.device().index());

    auto status = run_warp_rnnt(stream,
        (unsigned int *)counts.data<int>(),
        alphas.data<float>(), betas.data<float>(),
        ys.data<int>(), xs.data<float>(),
        grads.data<float>(), costs.data<float>(),
        xn.data<int>(), yn.data<int>(),
        N, T, U, V, blank
    );

    AT_CHECK(status == RNNT_STATUS_SUCCESS, "rnnt_loss status " + std::to_string(status));

    return std::make_tuple(costs, grads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rnnt_loss",
        &rnnt_loss,
        "CUDA-Warp RNN-Transducer loss (forward and backward).",
        pybind11::arg("xs"),
        pybind11::arg("ys"),
        pybind11::arg("xn"),
        pybind11::arg("yn"),
        pybind11::arg("blank") = 0
    );
}
