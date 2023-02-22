#include <string>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "core.h"

#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be located in the CUDA")

#define CHECK_FLOAT(x)                                                         \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                      \
                #x " must be a Float tensor")

#define CHECK_INT(x)                                                           \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                        \
                #x " must be a Int tensor")

#define None torch::indexing::None
#define Slice torch::indexing::Slice

std::tuple<at::Tensor, at::Tensor>
rnnt_loss(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &xn,
          const at::Tensor &yn, const int blank, const float fastemit_lambda) {
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
    TORCH_CHECK(xs.size(2) == ys.size(1) + 1,
                "ys shape (N, U-1) mismatched with xs (N, T, U, V)")

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

    auto stream = c10::cuda::getCurrentCUDAStream(xs.device().index());

    rnntStatus_t status;

    if (blank == -1) {

        TORCH_CHECK(V == 2, "xs must have values only for blank and label")

        status = run_warp_rnnt_gather(
            stream, (unsigned int *)counts.data_ptr<int>(),
            alphas.data_ptr<float>(), betas.data_ptr<float>(),
            xs.data_ptr<float>(), grads.data_ptr<float>(),
            costs.data_ptr<float>(), xn.data_ptr<int>(), yn.data_ptr<int>(), N,
            T, U, fastemit_lambda);

    } else {

        status = run_warp_rnnt(
            stream, (unsigned int *)counts.data_ptr<int>(),
            alphas.data_ptr<float>(), betas.data_ptr<float>(),
            ys.data_ptr<int>(), xs.data_ptr<float>(), grads.data_ptr<float>(),
            costs.data_ptr<float>(), xn.data_ptr<int>(), yn.data_ptr<int>(), N,
            T, U, V, blank, fastemit_lambda);
    }

    TORCH_CHECK(status == RNNT_STATUS_SUCCESS,
                "rnnt_loss status " + std::to_string(status));

    return std::make_tuple(costs, grads);
}

// return (costs, grad, loc)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rnnt_loss_compact_forward(const torch::Tensor &xs, const torch::Tensor &ys,
                          const torch::Tensor &xn, const torch::Tensor &yn,
                          const int blank, const float fastemit_lambda,
                          const bool required_grad) {
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
    TORCH_CHECK(ys.numel() == yn.sum().item<int64_t>(),
                "ys shape must be equal to (sum(yn), )")
    const at::cuda::OptionalCUDAGuard device_guard(device_of(xs));

    const auto N = xn.size(0);
    const auto Tm = xn.max().item<int64_t>();     // max of {T_i}
    const auto Um = yn.max().item<int64_t>() + 1; // max of {U_i}
    const auto V = xs.size(1);

    auto memPref =
        (xn * (yn + 1))
            .cumsum(0, torch::kInt32); // count of frames by current batch
    auto labelPref = yn.cumsum(0, torch::kInt32); // copy yn

    int64_t STU = memPref[-1].item<int64_t>();
    TORCH_CHECK(xs.size(0) == STU,
                "xs shape mismatch with (\\sum{xn*(yn+1)}, )")

    // set begin of memory location of each sequence
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}).clone();
        auto cumsumLablePref = labelPref.index({Slice(0, -1, None)}).clone();
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
        labelPref.index_put_({Slice(1, None, None)}, cumsumLablePref);
    }
    memPref[0] = 0;
    labelPref[0] = 0;

    torch::Tensor gather_xs = torch::empty({STU, 2L}, xs.options());
    torch::Tensor loc =
        torch::zeros({STU}, torch::dtype(torch::kInt64).device(xs.device()));

    /* gather the labels & blank
     * from xs (STU, V) to gather_xs (STU, 2)
     * gather_xs[:, 0] is collected from  xs[:, blank]
     * ... and gather_xs[:, 1] is collected from xs representing the log probs of labels
     * this is similar to non-compact mode and gather=True
     */
    run_gather_for_compact(
        xs.data_ptr<float>(), ys.data_ptr<int>(),
        (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
        gather_xs.data_ptr<float>(), loc.data_ptr<long>(),
        (unsigned int *)memPref.data_ptr<int>(),
        (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, V, blank);

    // the negtive log likelihood
    torch::Tensor costs = torch::empty({N}, xs.options());
    //  for maintain the execute status of forward/backward calculation
    torch::Tensor counts = torch::zeros({ys.numel() * 2 + 2 * N}, ys.options());

    // backward variable of RNN-T
    torch::Tensor betas = torch::empty({STU}, xs.options());

    torch::Tensor grads, alphas;
    if (required_grad) {
        // forward variable of RNN-T
        alphas = torch::empty_like(betas);
        grads = torch::empty_like(gather_xs);
    } else {
        // refer unused alphas / grads to betas
        // otherwise following alphas.data_ptr<float>() would raise error
        alphas = betas;
        grads = betas;
    }

    run_warp_rnnt_compact(
        (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
        betas.data_ptr<float>(), gather_xs.data_ptr<float>(),
        grads.data_ptr<float>(), costs.data_ptr<float>(),
        (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
        (unsigned int *)memPref.data_ptr<int>(),
        (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, fastemit_lambda,
        required_grad);

    return std::make_tuple(costs, grads, loc);
}

torch::Tensor rnnt_loss_compact_backward(const torch::Tensor &grad_cost,
                                         const torch::Tensor &grad_xs,
                                         const torch::Tensor &cum_lens,
                                         const torch::Tensor &loc, long V,
                                         int blank) {
    // Check contiguous
    CHECK_CONTIGUOUS(grad_cost);
    CHECK_CONTIGUOUS(grad_xs);
    CHECK_CONTIGUOUS(loc);
    // Check types
    CHECK_FLOAT(grad_cost);
    CHECK_FLOAT(grad_xs);
    TORCH_CHECK(loc.scalar_type() == at::ScalarType::Long,
                "loc must be a Long tensor");
    // Check device
    CHECK_CUDA(grad_cost);
    CHECK_CUDA(grad_xs);
    CHECK_CUDA(cum_lens);
    CHECK_CUDA(loc);
    // Check number of dimensions and elements
    TORCH_CHECK(grad_cost.dim() == 1,
                "grad_cost must have 1 dimensions")                // (N,)
    TORCH_CHECK(grad_xs.dim() == 2, "grad must have 2 dimensions") // (STU, 2)
    TORCH_CHECK(grad_xs.size(0) == loc.size(0),
                "grad and loc must be equal in dim=0")
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_cost));

    const auto N = grad_cost.size(0);
    const auto STU = grad_xs.size(0);

    torch::Tensor scatter_grad = torch::zeros({STU, V}, grad_cost.options());

    run_scatter_grad_for_compact(
        grad_cost.data_ptr<float>(), grad_xs.data_ptr<float>(),
        loc.data_ptr<long>(), (int *)cum_lens.data_ptr<int>(),
        scatter_grad.data_ptr<float>(), STU, N, V, blank);

    return scatter_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rnnt_loss", &rnnt_loss,
          "CUDA-Warp RNN-Transducer loss (forward and backward).",
          pybind11::arg("xs"), pybind11::arg("ys"), pybind11::arg("xn"),
          pybind11::arg("yn"), pybind11::arg("blank") = 0,
          pybind11::arg("fastemit_lambda") = 0.0);

    m.def(
        "rnnt_loss_compact", &rnnt_loss_compact_forward,
        "CUDA-Warp RNN-Transducer loss (forward / backward) in compact layout.",
        pybind11::arg("xs"), pybind11::arg("ys"), pybind11::arg("xn"),
        pybind11::arg("yn"), pybind11::arg("blank") = 0,
        pybind11::arg("fastemit_lambda") = 0.0,
        pybind11::arg("required_grad") = true);

    m.def("rnnt_loss_compact_backward", &rnnt_loss_compact_backward,
          "CUDA-Warp RNN-Transducer loss backward for compact layout",
          pybind11::arg("grad_costs"), pybind11::arg("grad_xs"),
          pybind11::arg("cumSum"), pybind11::arg("loc"), pybind11::arg("V"),
          pybind11::arg("blank") = 0);
}
