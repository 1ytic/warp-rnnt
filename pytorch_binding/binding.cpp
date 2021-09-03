#include <tuple>
#include <string>
// #include <iostream>

#include <THC/THC.h>

#include <torch/types.h>
#include <torch/extension.h>

#include "core.h"

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CONTIGUOUS(x)          \
    TORCH_CHECK((x).is_contiguous(), \
                #x " must be contiguous")

#define CHECK_CUDA(x)                   \
    TORCH_CHECK((x).device().is_cuda(), \
                #x " must be located in the CUDA")

#define CHECK_FLOAT(x)                                      \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, \
                #x " must be a Float tensor")

#define CHECK_INT(x)                                      \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Int, \
                #x " must be a Int tensor")

#define None torch::indexing::None
#define Slice torch::indexing::Slice

std::tuple<at::Tensor, at::Tensor> rnnt_loss(
    const at::Tensor &xs, const at::Tensor &ys,
    const at::Tensor &xn, const at::Tensor &yn,
    const int blank, const float fastemit_lambda)
{
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
    TORCH_CHECK(xs.dim() == 4, "xs must have 4 dimensions")
    TORCH_CHECK(xn.numel() == xs.size(0), "xn shape must be equal (N,)")
    TORCH_CHECK(yn.numel() == xs.size(0), "yn shape must be equal (N,)")
    TORCH_CHECK(xs.size(2) == ys.size(1) + 1, "ys shape (N, U-1) mismatched with xs (N, T, U, V)")

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

    rnntStatus_t status;

    if (blank == -1)
    {

        TORCH_CHECK(V == 2, "xs must have values only for blank and label")

        status = run_warp_rnnt_gather(stream,
                                      (unsigned int *)counts.data_ptr<int>(),
                                      alphas.data_ptr<float>(), betas.data_ptr<float>(),
                                      xs.data_ptr<float>(),
                                      grads.data_ptr<float>(), costs.data_ptr<float>(),
                                      xn.data_ptr<int>(), yn.data_ptr<int>(),
                                      N, T, U, fastemit_lambda);
    }
    else
    {

        status = run_warp_rnnt(stream,
                               (unsigned int *)counts.data_ptr<int>(),
                               alphas.data_ptr<float>(), betas.data_ptr<float>(),
                               ys.data_ptr<int>(), xs.data_ptr<float>(),
                               grads.data_ptr<float>(), costs.data_ptr<float>(),
                               xn.data_ptr<int>(), yn.data_ptr<int>(),
                               N, T, U, V, blank, fastemit_lambda);
    }

    TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "rnnt_loss status " + std::to_string(status));

    return std::make_tuple(costs, grads);
}

std::tuple<at::Tensor, at::Tensor> rnnt_loss_compact(
    const at::Tensor &xs, const at::Tensor &ys,
    const at::Tensor &xn, const at::Tensor &yn,
    const int blank, const float fastemit_lambda)
{
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
    TORCH_CHECK(xs.dim() == 2, "xs must have 2 dimensions")
    TORCH_CHECK(xn.size(0) == yn.size(0), "xn and yn shape must be equal (N,)")
    TORCH_CHECK(ys.numel() == yn.sum().item<int64_t>(), "ys shape must be equal to (sum(yn), )")

    const auto N = xn.size(0);
    const auto Tm = xn.max().item<int64_t>();     // max of {T_i}
    const auto Um = yn.max().item<int64_t>() + 1; // max of {U_i}
    const auto V = xs.size(1);

    auto memPref = (xn * (yn + 1)).cumsum(0, at::ScalarType::Int); // count of frames by current batch
    auto labelPref = yn.cumsum(0, at::ScalarType::Int);            // copy yn

    int64_t STU = memPref[-1].item<int64_t>();
    TORCH_CHECK(xs.size(0) == STU, "xs shape mismatch with (\\sum{xn*(yn+1)}, )")

    // set begin of memory location of each sequence
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}).clone();
        auto cumsumLablePref = labelPref.index({Slice(0, -1, None)}).clone();
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
        labelPref.index_put_({Slice(1, None, None)}, cumsumLablePref);
    }
    memPref[0] = 0;
    labelPref[0] = 0;

    at::TensorOptions buffer_opts(xs.device());
    at::TensorOptions counts_opts(xs.device());
    at::TensorOptions costs_opts(xs.device());

    counts_opts = counts_opts.dtype(at::ScalarType::Int);
    buffer_opts = buffer_opts.dtype(at::ScalarType::Float);
    costs_opts = costs_opts.dtype(at::ScalarType::Float);

    auto counts_shape = {ys.numel() * 2 + 2 * N}; // 2 * \sum_{(U_i+1)}
    auto buffer_shape = {STU};                    // \sum_{T_i*(U_i+1)}
    auto costs_shape = {N};

    torch::Tensor costs = torch::empty(costs_shape, costs_opts); // the negtive log likelihood
    at::Tensor counts = at::zeros(counts_shape, counts_opts);    //  for maintain the execute status of forward/backward calculation
    at::Tensor alphas = at::empty(buffer_shape, buffer_opts);    // forward variable of RNN-T
    at::Tensor betas = at::empty(buffer_shape, buffer_opts);     // backward variable of RNN-T
    at::Tensor grads;

    auto stream = at::cuda::getCurrentCUDAStream(xs.device().index());

    rnntStatus_t status;

    if (blank < 0)
    {
        // gather mode
        int real_blank = (-1) - blank;

        at::TensorOptions gather_xs_opts(xs.device());
        gather_xs_opts = gather_xs_opts.dtype(at::ScalarType::Float);

        auto gather_xs_shape = {STU, 2L}; // (\sum_{T_i*(U_i+1)}, 2)
        at::Tensor gather_xs = at::empty(gather_xs_shape, gather_xs_opts);

        status = run_gather(stream, xs.data_ptr<float>(), ys.data_ptr<int>(),
                            (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
                            gather_xs.data_ptr<float>(), (unsigned int *)memPref.data_ptr<int>(), (unsigned int *)labelPref.data_ptr<int>(),
                            N, Tm, Um, V, real_blank);

        TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "gather status " + std::to_string(status));

        grads = at::zeros_like(gather_xs);

        status = run_warp_rnnt_compact_gather(stream,
                                              (unsigned int *)counts.data_ptr<int>(),
                                              alphas.data_ptr<float>(), betas.data_ptr<float>(),
                                              gather_xs.data_ptr<float>(),
                                              grads.data_ptr<float>(), costs.data_ptr<float>(),
                                              (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
                                              (unsigned int *)memPref.data_ptr<int>(), (unsigned int *)labelPref.data_ptr<int>(),
                                              N, Tm, Um, fastemit_lambda);
    }
    else
    {
        grads = at::zeros_like(xs);
        memPref *= V;
        status = run_warp_rnnt_compact(stream,
                                       (unsigned int *)counts.data_ptr<int>(),
                                       alphas.data_ptr<float>(), betas.data_ptr<float>(),
                                       ys.data_ptr<int>(), xs.data_ptr<float>(),
                                       grads.data_ptr<float>(), costs.data_ptr<float>(),
                                       xn.data_ptr<int>(), yn.data_ptr<int>(),
                                       memPref.data_ptr<int>(), labelPref.data_ptr<int>(),
                                       N, Tm, Um, V, blank, fastemit_lambda);
    }

    TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "rnnt_loss status " + std::to_string(status));

    return std::make_tuple(costs, grads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "rnnt_loss",
        &rnnt_loss,
        "CUDA-Warp RNN-Transducer loss (forward and backward).",
        pybind11::arg("xs"),
        pybind11::arg("ys"),
        pybind11::arg("xn"),
        pybind11::arg("yn"),
        pybind11::arg("blank") = 0,
        pybind11::arg("fastemit_lambda") = 0.0);

    m.def(
        "rnnt_loss_compact",
        &rnnt_loss_compact,
        "CUDA-Warp RNN-Transducer loss with compact memory layout",
        pybind11::arg("xs"),
        pybind11::arg("ys"),
        pybind11::arg("xn"),
        pybind11::arg("yn"),
        pybind11::arg("blank") = 0,
        pybind11::arg("fastemit_lambda") = 0.0);
}
