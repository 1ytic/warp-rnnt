#ifndef RNNT_CORE_H
#define RNNT_CORE_H

#include <cuda_runtime.h>

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

rnntStatus_t run_warp_rnnt(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                           const int *labels, const float *log_probs, float *grads, float *costs,
                           const int *xn, const int *yn, int N, int T, int U, int V, int blank);

#ifdef __cplusplus
}
#endif

#endif //RNNT_CORE_H