#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define WLL 1024
#define WL 512
#define W 64
#define H 16

__global__ void kernel_fill_gather(const float *xs, const int *ys, const unsigned int *xn, const unsigned int *yn,
                                   float *gather_xs, long *loc, const unsigned int *memPref, const unsigned int *labelPref,
                                   unsigned int V, unsigned int blank)
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
    loc[mem_loc + t * actual_u + u] = l;
    float *ptr_gather = gather_xs + ((mem_loc + t * actual_u + u) << 1);
    // gather_xs(n, t, u, 0) = xs(n, t, u, blank)
    *(ptr_gather++) = xs[(mem_loc + t * actual_u + u) * V + blank];
    // gather_xs(n, t, u, 1) = xs(n, t, u, l)
    *ptr_gather = xs[(mem_loc + t * actual_u + u) * V + l];
}

rnntStatus_t run_gather(cudaStream_t stream, const float *xs, const int *ys, const unsigned int *xn, const unsigned int *yn,
                        float *gather_xs, long *loc,
                        const unsigned int *memPref, const unsigned int *labelPref,
                        unsigned int N, unsigned int T, unsigned int U, unsigned int V, unsigned int blank)
{

    dim3 threads1(W, H);
    dim3 blocks1((T + W - 1) / W, (U + H - 1) / H, N);

    kernel_fill_gather<<<blocks1, threads1, 0, stream>>>(xs, ys, xn, yn, gather_xs, loc, memPref, labelPref, V, blank);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    return RNNT_STATUS_SUCCESS;
}

__global__ void kernel_fill_scatter_grad(const float *grad_cost, const float *gather_grad, const long *loc,
                                         const unsigned int *cumSum, float *scatter_grad,
                                         unsigned int STU, unsigned int V, unsigned int blank)
{
    unsigned int i = blockIdx.x * WL + threadIdx.x;
    if (i >= STU)
        return;

    unsigned int n = blockIdx.y;
    // ensure i in [cumSum[n-1], cumSum[n]]
    if (i >= cumSum[n] || (n > 0 && i < cumSum[n - 1]))
        return;

    // printf("(i, n)=(%d, %d)\n", i, n);
    if (threadIdx.y == 0)
    {
        // fill blank label grad
        scatter_grad[i * V + blank] = gather_grad[i << 1] * grad_cost[n];
    }
    else //if (threadIdx.y == 1)
    {
        // fill real label grad
        scatter_grad[i * V + loc[i]] = gather_grad[(i << 1) + 1] * grad_cost[n];
    }
}

rnntStatus_t run_scatter_grad(cudaStream_t stream, const float *grad_cost, const float *gather_grad,
                              const long *loc, const unsigned int *cumSum,
                              float *scatter_grad, unsigned int STU, unsigned int N, unsigned int V, unsigned int blank)
{
    // grad_cost (N, )
    // gather_grad (STU, 2)
    // scatter_grad (STU, V)

    dim3 threads1(WL, 2);
    dim3 blocks1((STU + WL - 1) / WL, N);

    kernel_fill_scatter_grad<<<blocks1, threads1, 0, stream>>>(grad_cost, gather_grad, loc, cumSum, scatter_grad, STU, V, blank);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    return RNNT_STATUS_SUCCESS;
}

__global__ void kernel_fill_grad(const float *grad_cost, float *grad, const unsigned int *cumSum, unsigned int STU, unsigned int V)
{

    unsigned int i = blockIdx.x * W + threadIdx.x;
    if (i >= STU)
        return;

    unsigned int n = blockIdx.z;
    // ensure i in [cumSum[n-1], cumSum[n]]
    if (i >= cumSum[n] || (n > 0 && i < cumSum[n - 1]))
        return;

    unsigned int v = blockIdx.y * H + threadIdx.y;
    if (v >= V)
        return;

    grad[i * V + v] *= grad_cost[n];
}

rnntStatus_t run_backward_compact(cudaStream_t stream, const float *grad_cost, float *grad,
                                  const unsigned int *cumSum, unsigned int STU, unsigned N, unsigned int V)
{
    // grad_cost (N, )
    // grad (STU, V)
    dim3 threads(W, H);
    dim3 blocks((STU + W - 1) / W, (V + H - 1) / V, N); // (N-1) redundancy

    kernel_fill_grad<<<blocks, threads, 0, stream>>>(grad_cost, grad, cumSum, STU, V);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    return RNNT_STATUS_SUCCESS;
}