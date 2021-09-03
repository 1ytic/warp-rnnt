#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define W 64
#define H 16

__global__ void kernel_fill_gather(const float *xs, const int *ys, const unsigned int *xn, const unsigned int *yn,
                                   float *gather_xs, const unsigned int *memPref, const unsigned int *labelPref,
                                   unsigned int V, int blank)
{
    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int u = blockIdx.y * H + threadIdx.y;
    unsigned int n = blockIdx.z;

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t >= actual_t || u >= actual_u)
        return;

    unsigned int mem_loc = memPref[n];

    // l = ys(n, u)
    unsigned int l = ys[labelPref[n] + u];
    // gather_xs(n, t, u, 0) = xs(n, t, u, blank)
    gather_xs[(mem_loc << 1) + t * actual_u * 2 + u * 2] = xs[mem_loc * V + t * actual_u * V + u * V + blank];
    // gather_xs(n, t, u, 1) = xs(n, t, u, l)
    gather_xs[(mem_loc << 1) + t * actual_u * 2 + u * 2 + 1] = xs[mem_loc * V + t * actual_u * V + u * V + l];
}

rnntStatus_t run_gather(cudaStream_t stream, const float *xs, const int *ys, const unsigned int *xn, const unsigned int *yn,
                        float *gather_xs, const unsigned int *memPref, const unsigned int *labelPref,
                        unsigned int N, unsigned int T, unsigned int U, unsigned int V, int blank)
{

    dim3 threads1(W, H);
    dim3 blocks1((T + W - 1) / W, (U + H - 1) / H, N);

    kernel_fill_gather<<<blocks1, threads1, 0, stream>>>(xs, ys, xn, yn, gather_xs, memPref, labelPref, V, blank);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    return RNNT_STATUS_SUCCESS;
}