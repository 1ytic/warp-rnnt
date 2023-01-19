#define EIGEN_USE_GPU

#include <cuda.h>
#include <iostream>
#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "core.h"

static const char* transducerGetStatusString(rnntStatus_t status) {
  switch (status) {
    case RNNT_STATUS_SUCCESS:
      return "no error";
    case RNNT_STATUS_WARP_FAILED:
      return "warp failed";
    case RNNT_STATUS_GRADS_BLANK_FAILED:
      return "grads blank failed";
    case RNNT_STATUS_GRADS_LABEL_FAILED:
      return "grads label failed";
    case RNNT_STATUS_COSTS_FAILED:
    default:
      return "unknown error";
  }
}

namespace tf = tensorflow;

REGISTER_OP("TransducerLoss")
.Input("log_probs: float32")
.Input("labels: int32")
.Input("frames_lengths: int32")
.Input("labels_lengths: int32")
.Attr("blank: int = 0")
.Attr("fastemit_lambda: float = 0.0")
.Output("costs: float32")
.Output("grads: float32")
.SetShapeFn([](tf::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(2));
  c->set_output(1, c->input(0));
  return tf::Status::OK();
});

namespace transducer {

class TransducerLossOpBase : public tf::OpKernel {
 public:
  explicit TransducerLossOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank", &blank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fastemit_lambda", &fastemit_lambda_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    rnntStatus_t status;
    auto cuda_stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
    const tf::Tensor* log_probs; // (N, T, U, V)
    const tf::Tensor* labels; // (N, U-1)
    const tf::Tensor* frames_lengths; // (N,)
    const tf::Tensor* labels_lengths; // (N,)

    OP_REQUIRES_OK(ctx, ctx->input("log_probs", &log_probs));
    OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
    OP_REQUIRES_OK(ctx, ctx->input("frames_lengths", &frames_lengths));
    OP_REQUIRES_OK(ctx, ctx->input("labels_lengths", &labels_lengths));

    OP_REQUIRES(ctx, log_probs->shape().dims() == 4,
                tf::errors::InvalidArgument("transducer_loss log_probs is not a 4-Tensor"));

    OP_REQUIRES(ctx, labels->shape().dims() == 2,
                tf::errors::InvalidArgument("transducer_loss labels is not a 2-Tensor"));

    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(frames_lengths->shape()),
                tf::errors::InvalidArgument("transducer_loss frames_lengths is not a vector"));

    OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(labels_lengths->shape()),
                tf::errors::InvalidArgument("transducer_loss labels_lengths is not a vector"));

    const auto& log_probs_shape = log_probs->shape();
    const auto N = log_probs_shape.dim_size(0);
    const auto T = log_probs_shape.dim_size(1);
    const auto U = log_probs_shape.dim_size(2);
    const auto V = log_probs_shape.dim_size(3);

    OP_REQUIRES(
      ctx, tf::FastBoundsCheck(V, std::numeric_limits<int>::max()),
      tf::errors::InvalidArgument("transducer_loss num_classes cannot exceed max int"));

    OP_REQUIRES(
      ctx, N == frames_lengths->dim_size(0),
      tf::errors::InvalidArgument("transducer_loss len(frames_lengths) != batch_size.  ",
                                  "len(input_length):  ", frames_lengths->dim_size(0),
                                  " batch_size: ", N));

    OP_REQUIRES(
      ctx, U - 1 == labels->dim_size(1),
      tf::errors::InvalidArgument("transducer_loss labels->dim_size(1) != he maximum number of output labels.  ",
                                  "labels->dim_size(1):  ", labels->dim_size(1),
                                  " the maximum number of output labels: ", U - 1));

    OP_REQUIRES(ctx, labels_lengths->dim_size(0) == frames_lengths->dim_size(0),
                tf::errors::InvalidArgument(
                  "transducer_loss labels_lengths and labels_values must contain the "
                  "same number of rows, but saw shapes: ",
                  labels_lengths->shape().DebugString(), " vs. ",
                  frames_lengths->shape().DebugString()));

    // allocate output memory.
    tf::Tensor* costs_ = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("costs", frames_lengths->shape(), &costs_));
    auto costs = costs_->vec<float>();

    tf::Tensor* grads_ = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("grads", log_probs->shape(), &grads_));
    cudaMemset(grads_->flat<float>().data(), 0, grads_->NumElements()*sizeof(float));
    auto grads = grads_->tensor<float, 4>();

    // allocate temp memory.
    tf::Tensor counts_;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_INT32, tf::TensorShape({N, U * 2}), &counts_));
    cudaMemset(counts_.flat<int32_t>().data(), 0, counts_.NumElements()*sizeof(int32_t));
    auto counts = counts_.tensor<int32_t, 2>();

    tf::Tensor alphas_;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_FLOAT, tf::TensorShape({N, T, U}), &alphas_));
    auto alphas = alphas_.tensor<float, 3>();

    tf::Tensor betas_;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_FLOAT, tf::TensorShape({N, T, U}), &betas_));
    auto betas = betas_.tensor<float, 3>();

    auto labels_t = labels->tensor<int32_t, 2>();
    auto log_probs_t = log_probs->tensor<float, 4>();

    auto labels_lengths_t = labels_lengths->vec<int32_t>();
    auto frames_lengths_t = frames_lengths->vec<int32_t>();

    if (blank_ == -1) {
      status = run_warp_rnnt_gather(cuda_stream,
                                    (unsigned int *)counts.data(),
                                    alphas.data(), betas.data(),
                                    log_probs_t.data(),
                                    grads.data(), costs.data(),
                                    frames_lengths_t.data(), labels_lengths_t.data(),
                                    N, T, U, fastemit_lambda_
                                   );
    } else {
      status = run_warp_rnnt(cuda_stream,
                             (unsigned int *)counts.data(),
                             alphas.data(), betas.data(),
                             labels_t.data(), log_probs_t.data(),
                             grads.data(), costs.data(),
                             frames_lengths_t.data(), labels_lengths_t.data(),
                             N, T, U, V, blank_, fastemit_lambda_
                            );
    }

    OP_REQUIRES(ctx, status == RNNT_STATUS_SUCCESS,
                tf::errors::Internal("transducer compute error in run_warp_rnnt: ", transducerGetStatusString(status))
               );
  }

 private:
  int blank_;
  float fastemit_lambda_;
  TF_DISALLOW_COPY_AND_ASSIGN(TransducerLossOpBase);
};

class TransducerLossOpGPU : public TransducerLossOpBase {
 public:
  explicit TransducerLossOpGPU(tf::OpKernelConstruction* ctx) : TransducerLossOpBase(ctx) {
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TransducerLossOpGPU);
};

REGISTER_KERNEL_BUILDER(Name("TransducerLoss").Device(::tensorflow::DEVICE_GPU)
                        .HostMemory("costs"),
                        TransducerLossOpGPU);

} /* namespace transducer */

#undef EIGEN_USE_GPU
