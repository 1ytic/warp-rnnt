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

// return (costs, grad, loc, blank)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> rnnt_loss_compact_forward(
    const torch::Tensor &xs, const torch::Tensor &ys,
    const torch::Tensor &xn, const torch::Tensor &yn,
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

    const auto device = xs.device();
    // the negtive log likelihood
    torch::Tensor costs = torch::empty({N}, torch::dtype(torch::kFloat32).device(device));
    //  for maintain the execute status of forward/backward calculation
    torch::Tensor counts = torch::zeros({ys.numel() * 2 + 2 * N}, torch::dtype(torch::kInt32).device(device));
    // forward variable of RNN-T
    torch::Tensor alphas = torch::empty({STU}, torch::dtype(torch::kFloat32).device(device));
    // backward variable of RNN-T
    torch::Tensor betas = torch::empty_like(alphas);
    torch::Tensor grads;

    auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    rnntStatus_t status;

    if (blank < 0)
    {
        // gather mode
        int real_blank = (-1) - blank;

        torch::Tensor gather_xs = torch::empty({STU, 2L}, torch::dtype(torch::kFloat32).device(device));
        torch::Tensor loc = torch::zeros({STU}, torch::dtype(torch::kInt64).device(device));

        status = run_gather(stream, xs.data_ptr<float>(), ys.data_ptr<int>(),
                            (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
                            gather_xs.data_ptr<float>(), loc.data_ptr<long>(),
                            (unsigned int *)memPref.data_ptr<int>(), (unsigned int *)labelPref.data_ptr<int>(),
                            N, Tm, Um, V, real_blank);

        TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "gather status " + std::to_string(status));

        grads = torch::zeros_like(gather_xs);

        status = run_warp_rnnt_compact_gather(stream,
                                              (unsigned int *)counts.data_ptr<int>(),
                                              alphas.data_ptr<float>(), betas.data_ptr<float>(),
                                              gather_xs.data_ptr<float>(),
                                              grads.data_ptr<float>(), costs.data_ptr<float>(),
                                              (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
                                              (unsigned int *)memPref.data_ptr<int>(), (unsigned int *)labelPref.data_ptr<int>(),
                                              N, Tm, Um, fastemit_lambda);
        TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "rnnt_loss status " + std::to_string(status));

        return std::make_tuple(costs, grads, loc, blank);
    }
    else
    {
        grads = torch::zeros_like(xs);
        memPref *= V;
        status = run_warp_rnnt_compact(stream,
                                       (unsigned int *)counts.data_ptr<int>(),
                                       alphas.data_ptr<float>(), betas.data_ptr<float>(),
                                       (unsigned int *)ys.data_ptr<int>(), xs.data_ptr<float>(),
                                       grads.data_ptr<float>(), costs.data_ptr<float>(),
                                       (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
                                       (unsigned int *)memPref.data_ptr<int>(), (unsigned int *)labelPref.data_ptr<int>(),
                                       N, Tm, Um, V, blank, fastemit_lambda);
        TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "rnnt_loss status " + std::to_string(status));

        // non gather mode, only (costs, grad) is useful.
        return std::make_tuple(costs, grads, grads, blank);
    }
}

torch::Tensor rnnt_loss_compact_backward(
    const torch::Tensor &grad_cost, torch::Tensor &grad, const torch::Tensor &cumSum,
    const torch::Tensor &loc, long V, int blank)
{
    // Check contiguous
    CHECK_CONTIGUOUS(grad_cost);
    CHECK_CONTIGUOUS(grad);
    // Check types
    CHECK_FLOAT(grad_cost);
    CHECK_FLOAT(grad);
    // Check device
    CHECK_CUDA(grad_cost);
    CHECK_CUDA(grad);
    // Check number of dimensions and elements
    TORCH_CHECK(grad_cost.dim() == 1, "grad_cost must have 1 dimensions") // (N,)
    TORCH_CHECK(grad.dim() == 2, "grad must have 2 dimensions")           // (STU, 2)

    const auto N = grad_cost.size(0);
    const auto STU = grad.size(0);

    auto stream = c10::cuda::getCurrentCUDAStream(grad_cost.device().index());
    rnntStatus_t status;

    const auto device = grad_cost.device();

    if (blank < 0)
    {
        CHECK_CONTIGUOUS(loc);
        TORCH_CHECK(loc.scalar_type() == at::ScalarType::Long, "loc must be a Long tensor");
        CHECK_CUDA(loc);
        TORCH_CHECK(grad.size(0) == loc.size(0), " grad and loc must be equal in dim=0")

        int real_blank = -1 - blank;

        torch::Tensor scatter_grad = torch::zeros({STU, V}, torch::dtype(torch::kFloat32).device(device));

        status = run_scatter_grad(stream, grad_cost.data_ptr<float>(), grad.data_ptr<float>(),
                                  loc.data_ptr<long>(), (unsigned int *)cumSum.data_ptr<int>(), scatter_grad.data_ptr<float>(),
                                  STU, N, V, real_blank);

        TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "scatter status " + std::to_string(status));
        return scatter_grad;
    }
    else
    {

        status = run_backward_compact(stream, grad_cost.data_ptr<float>(), grad.data_ptr<float>(),
                                      (unsigned int *)cumSum.data_ptr<int>(), STU, N, V);
        TORCH_CHECK(status == RNNT_STATUS_SUCCESS, "grad accum status " + std::to_string(status));

        return grad;
    }
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
        "rnnt_loss_compact_forward",
        &rnnt_loss_compact_forward,
        "CUDA-Warp RNN-Transducer loss with compact memory layout",
        pybind11::arg("xs"),
        pybind11::arg("ys"),
        pybind11::arg("xn"),
        pybind11::arg("yn"),
        pybind11::arg("blank") = 0,
        pybind11::arg("fastemit_lambda") = 0.0);

    m.def(
        "rnnt_loss_compact_backward",
        &rnnt_loss_compact_backward,
        "Compact RNN-T loss backward",
        pybind11::arg("grad_cost"),
        pybind11::arg("grad"),
        pybind11::arg("cumSum"),
        pybind11::arg("loc"),
        pybind11::arg("V"),
        pybind11::arg("blank"));
}
