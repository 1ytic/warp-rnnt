#ifndef RNNT_CORE_H
#define RNNT_CORE_H

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_KERNEL_STAT(s)                                                   \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, #s " error: %s\n", cudaGetErrorString(err));       \
            exit(-1);                                                          \
        }                                                                      \
    }

typedef enum {
    RNNT_STATUS_SUCCESS = 0,
    RNNT_STATUS_WARP_FAILED = 1,
    RNNT_STATUS_GRADS_BLANK_FAILED = 2,
    RNNT_STATUS_GRADS_LABEL_FAILED = 3,
    RNNT_STATUS_COSTS_FAILED = 4
} rnntStatus_t;

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

rnntStatus_t run_warp_rnnt(cudaStream_t stream, unsigned int *counts,
                           float *alphas, float *betas, const int *labels,
                           const float *log_probs, float *grads, float *costs,
                           const int *xn, const int *yn, int N, int T, int U,
                           int V, int blank, float fastemit_lambda);

rnntStatus_t run_warp_rnnt_gather(cudaStream_t stream, unsigned int *counts,
                                  float *alphas, float *betas,
                                  const float *log_probs, float *grads,
                                  float *costs, const int *xn, const int *yn,
                                  int N, int T, int U, float fastemit_lambda);

void run_gather_for_compact(const float *xs, const int *ys,
                            const unsigned int *xn, const unsigned int *yn,
                            float *gather_xs, long *loc,
                            const unsigned int *memPref,
                            const unsigned int *labelPref, unsigned int N,
                            unsigned int T, unsigned int U, unsigned int V,
                            unsigned int blank);
void run_warp_rnnt_compact(unsigned int *counts, float *alphas, float *betas,
                           const float *log_probs, float *grads, float *costs,
                           const unsigned int *xn, const unsigned int *yn,
                           const unsigned int *memPref,
                           const unsigned int *labelPref, unsigned int N,
                           unsigned int T, unsigned int U,
                           float fastemit_lambda, bool beta_only);

void run_scatter_grad_for_compact(const float *grad_cost,
                                  const float *gather_grad, const long *loc,
                                  const int *cum_lens, float *scatter_grad,
                                  unsigned int STU, unsigned int N,
                                  unsigned int V, unsigned int blank);

#ifdef __cplusplus
}
#endif

#endif // RNNT_CORE_H
