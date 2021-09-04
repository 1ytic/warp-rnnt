#ifndef RNNT_CORE_H
#define RNNT_CORE_H

#include <cuda_runtime.h>

typedef enum
{
    RNNT_STATUS_SUCCESS = 0,
    RNNT_STATUS_WARP_FAILED = 1,
    RNNT_STATUS_GRADS_BLANK_FAILED = 2,
    RNNT_STATUS_GRADS_LABEL_FAILED = 3,
    RNNT_STATUS_COSTS_FAILED = 4
} rnntStatus_t;

#ifdef __cplusplus
#include <cstddef>
extern "C"
{
#endif

    rnntStatus_t run_warp_rnnt(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                               const int *labels, const float *log_probs, float *grads, float *costs,
                               const int *xn, const int *yn, int N, int T, int U, int V, int blank, float fastemit_lambda);

    rnntStatus_t run_warp_rnnt_gather(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                                      const float *log_probs, float *grads, float *costs,
                                      const int *xn, const int *yn, int N, int T, int U, float fastemit_lambda);

    rnntStatus_t run_warp_rnnt_compact(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                                       const int *labels, const float *log_probs, float *grads, float *costs,
                                       const int *xn, const int *yn, const int *memPref, const int *labelPref,
                                       int N, int Tm, int Um, int V, int blank, float fastemit_lambda);

    rnntStatus_t run_warp_rnnt_compact_gather(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                                              const float *log_probs, float *grads, float *costs,
                                              const unsigned int *xn, const unsigned int *yn, const unsigned int *memPref, const unsigned int *labelPref,
                                              unsigned int N, unsigned int T, unsigned int U, float fastemit_lambda);

    rnntStatus_t run_gather(cudaStream_t stream, const float *xs, const int *ys, const unsigned int *xn, const unsigned int *yn,
                            float *gather_xs, long *loc,
                            const unsigned int *memPref, const unsigned int *labelPref,
                            unsigned int N, unsigned int T, unsigned int U, unsigned int V, unsigned int blank);

    rnntStatus_t run_scatter_grad(cudaStream_t stream, const float *grad_cost, const float *gather_grad,
                                  const long *loc, const unsigned int *cumSum,
                                  float *grad, unsigned int STU, unsigned int N, unsigned int V, unsigned int blank);

    rnntStatus_t run_backward_compact(cudaStream_t stream, const float *grad_cost, float *grad,
                                      const unsigned int *cumSum, unsigned int STU, unsigned N, unsigned int V);

#ifdef __cplusplus
}
#endif

#endif //RNNT_CORE_H
