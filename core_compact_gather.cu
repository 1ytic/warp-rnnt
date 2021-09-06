#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define W 32
#define G 1024
#define B 256

__forceinline__ __device__ static float log_sum_exp(float a, float b)
{
    float maximum, diff;
    if (a > b)
    {
        maximum = a;
        diff = b - a;
    }
    else
    {
        maximum = b;
        diff = a - b;
    }
    //if (diff > -42) {
    maximum += log1pf(expf(diff));
    //}
    return maximum;
}

__device__ void kernel_warp_alphas_compact(unsigned int *counts, volatile float *alphas,
                                           const float *log_probs,
                                           const unsigned int *xn, const unsigned int *yn,
                                           const unsigned int *memPref, const unsigned int *labelPref)
{

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    assert(d < W);
    assert(blockDim.x == W);

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;

    unsigned int *lock = counts + ((labelPref[n] + n) << 1) + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        // initialize the state as log(p) = 0.
        // alphas[n, 0, 0] = 0;
        alphas[mem_loc] = 0.0f;
    }

    if (blockIdx.x > 0)
    {
        // Wait previous row
        do
        {
        } while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0)
    {
        // Wait previous column
        do
        {
        } while (atomicAdd(lock - 1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u)
    {

        // Compute initial row value

        // a = alphas[n, 0, u-1]
        float a = alphas[mem_loc + u - 1];
        
        // b = log_probs[n, 0, u-1, 1]
        float b = log_probs[mem_beg + (u << 1) - 1]; // should be [mem_beg + 2 * (u-1) + 1] in a more readable manner.

        // alphas[n, 0, u] = a + b
        alphas[mem_loc + u] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t)
    {

        // Compute initial column with local scan algorithm

        float a;

        // b = log_probs[n, t-1, 0, 0]
        float b = log_probs[mem_beg + ((t - 1) * actual_u << 1)];

#pragma unroll
        for (unsigned int i = 1; i < W; i *= 2)
        {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d)
            {
                b += a;
            }
        }

        // a = alphas[n, p, 0]
        a = alphas[mem_loc + p * actual_u];

        // alphas[n, t, 0] = a + b;
        alphas[mem_loc + t * actual_u] = a + b;
    }

    if (t < actual_t && u < actual_u)
    {

        // Ready to compute alphas[t, u]

        // bias = log_probs[n, t-1, u, 0]
        float bias = log_probs[mem_beg + (((t - 1) * actual_u + u) << 1)];

        // skip = alphas[n, p, u] + bias
        float skip = alphas[mem_loc + p * actual_u + u] + bias;
        
        // emit = alphas[n, t, u-1] + log_probs[n, t, u-1, 1]
        float emit = alphas[mem_loc + t * actual_u + u - 1] + log_probs[mem_beg + ((t * actual_u + u) << 1) - 1];

        float r = log_sum_exp(skip, emit);
        float output = r;

        for (unsigned int i = 1; i < W; i++)
        {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d)
            {
                r = log_sum_exp(r + bias, emit);
                output = r;
            }
        }

        // alphas[n, t, u] = output
        alphas[mem_loc + t * actual_u + u] = output;
    }

    if (d == 0)
    {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__device__ void kernel_warp_betas_compact(unsigned int *counts, volatile float *betas,
                                          const float *log_probs,
                                          const unsigned int *xn, const unsigned int *yn,
                                          const unsigned int *memPref, const unsigned int *labelPref)
{

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    assert(d < W);
    assert(blockDim.x == W);

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    // int T1 = actual_t - 1;
    // int U1 = actual_u - 1;
    unsigned int _val = actual_t * actual_u - u;
    unsigned int _valm1 = _val - 1;
    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;

    unsigned int *lock = counts + ((labelPref[n] + n) << 1) + actual_u + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        // betas[n, T1, U1] = log_probs[n, T1, U1, 0]
        betas[mem_loc + _valm1 + u] = log_probs[mem_beg + ((_valm1 + u) << 1)];
    }

    if (blockIdx.x > 0)
    {
        // Wait previous row
        do
        {
        } while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0)
    {
        // Wait previous column
        do
        {
        } while (atomicAdd(lock - 1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u)
    {

        // Compute last row value

        // a = betas[n, T1, U1-u+1]
        float a = betas[mem_loc + _val];

        // b = log_probs[n, T1, U1-u, 1]
        float b = log_probs[mem_beg + (_val << 1) - 1];

        // betas[n, T1, U1-u] = a + b
        betas[mem_loc + _valm1] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t)
    {

        // Compute last column with local scan algorithm

        float a;

        // b = log_probs[n, T1-t, U1, 0]
        float b = log_probs[mem_beg + ((_valm1 + u - t * actual_u) << 1)];

#pragma unroll
        for (unsigned int i = 1; i < W; i *= 2)
        {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d)
            {
                b += a;
            }
        }

        // a = betas[n, T1-p, U1]
        a = betas[mem_loc + _valm1 + u - p * actual_u];

        // betas[n, T1 - t, U1] = a + b;
        betas[mem_loc + _valm1 + u - t * actual_u] = a + b;
    }

    if (t < actual_t && u < actual_u)
    {

        // Ready to compute betas[T1-t, U1-u]

        // bias = log_probs[n, T1 - t, U1 - u, 0];
        float bias = log_probs[mem_beg + ((_valm1 - t * actual_u) << 1)];
        
        // skip = betas[n, T1 - p, U1 - u] + bias;
        float skip = betas[mem_loc + _valm1 - p * actual_u] + bias;
        
        // emit = betas[n, T1 - t, U1 - u + 1] + log_probs[n, T1 - t, U1 - u, 1];
        float emit = betas[mem_loc + _val - t * actual_u] + log_probs[mem_beg + ((_val - t * actual_u) << 1) - 1];

        float r = log_sum_exp(skip, emit);
        float output = r;

        for (unsigned int i = 1; i < W; i++)
        {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d)
            {
                r = log_sum_exp(r + bias, emit);
                output = r;
            }
        }

        // betas[n, T1 - t, U1 - u] = output;
        betas[mem_loc + _valm1 - t * actual_u] = output;
    }

    if (d == 0)
    {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__global__ void kernel_warp_compact(unsigned int *counts, volatile float *alphas, volatile float *betas,
                                    const float *log_probs,
                                    const unsigned int *xn, const unsigned int *yn,
                                    const unsigned int *memPref, const unsigned int *labelPref)
{
    if (threadIdx.y == 0)
    {
        kernel_warp_alphas_compact(counts, alphas, log_probs, xn, yn, memPref, labelPref);
    }
    else if (threadIdx.y == 1)
    {
        kernel_warp_betas_compact(counts, betas, log_probs, xn, yn, memPref, labelPref);
    }
}

__global__ void kernel_grads_blank_compact(float *grads, const float *alphas, const float *betas,
                                           const float *log_probs,
                                           const unsigned int *xn, const unsigned int *yn,
                                           const unsigned int *memPref)
{

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = g * G + d;

    assert(d < G);
    assert(blockDim.x == G);

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t >= actual_t || u >= actual_u)
        return;

    if (t == actual_t - 1 && u < actual_u - 1)
        return;

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;
    // a = alphas[n, t, u];
    float a = alphas[mem_loc + t * actual_u + u];

    if (t < actual_t - 1)
    {
        // a += betas[n, t + 1, u];
        a += betas[mem_loc + (t + 1) * actual_u + u];
    }

    // index = (n, t, u, 0);
    unsigned int index = mem_beg + ((t * actual_u + u) << 1);

    // a = expf(a + log_probs[index] - betas[n, 0, 0]);
    a = expf(a + log_probs[index] - betas[mem_loc]);

    grads[index] = -a;
}

__global__ void kernel_grads_label_compact(float *grads, const float *alphas, const float *betas,
                                           const float *log_probs,
                                           const unsigned int *xn, const unsigned int *yn,
                                           const unsigned int *memPref, const unsigned int *labelPref,
                                           float fastemit_lambda)
{

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = g * G + d;

    assert(d < G);
    assert(blockDim.x == G);

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n];

    if (t >= actual_t || u >= actual_u)
        return;

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;
    unsigned int _index = t * (actual_u + 1) + u;

    // a = alphas[n, t, u] + betas[n, t, u + 1];
    float a = alphas[mem_loc + _index] + betas[mem_loc + _index + 1];

    // index = (n, t, u, 1);
    unsigned int index = mem_beg + (_index << 1) + 1;

    // a = expf(a + log_probs[index] - betas[n, 0, 0]);
    a = expf(a + log_probs[index] - betas[mem_loc]);

    // apply FastEmit regularization
    // https://arxiv.org/abs/2010.11148
    a = (1. + fastemit_lambda) * a;

    grads[index] = -a;
}

__global__ void kernel_fill_costs_compact(float *costs, float *grads, const float *alphas, const float *betas, const float *log_probs,
                                          const unsigned int *xn, const unsigned int *yn, const unsigned int *memPref,
                                          unsigned int N)
{

    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N)
        return;

    unsigned int t = xn[n] - 1;
    unsigned int u = yn[n];
    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;

    // a = alphas[n, t, u] + log_probs[n, t, u, 0]
    float a = alphas[mem_loc + t * (u + 1) + u] + log_probs[mem_beg + ((t * (u + 1) + u) << 1)];

    // b = betas[n, 0, 0]
    float b = betas[mem_loc];

    float ratio = fabsf(a - b) / fabsf(fmaxf(a, b));

    if (ratio > 0.001)
    {

        printf("\nWARNING: sample %d [%d, %d] has a forward/backward mismatch %f / %f\n",
               n, t + 1, u, a, b);

        // grads[n, 0, 0, 0]
        float *g = grads + mem_beg;

        for (unsigned int i = 0; i < t + 1; ++i)
        {
            for (unsigned int j = 0; j < u + 1; ++j)
            {
                for (unsigned int v = 0; v < 2; ++v, ++g)
                {
                    *g = 0;
                }
            }
        }

        b = (a + b) / 2.0f;
    }

    costs[n] = -b;
}

rnntStatus_t run_warp_rnnt_compact_gather(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                                          const float *log_probs, float *grads, float *costs,
                                          const unsigned int *xn, const unsigned int *yn, const unsigned int *memPref, const unsigned int *labelPref,
                                          unsigned int N, unsigned int T, unsigned int U, float fastemit_lambda)
{

    dim3 threads1(W, 2);
    dim3 blocks1((T + W - 1) / W, U, N);
    kernel_warp_compact<<<blocks1, threads1, 0, stream>>>(counts, alphas, betas, log_probs, xn, yn, memPref, labelPref);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    dim3 blocks2((T + G - 1) / G, U, N);
    kernel_grads_blank_compact<<<blocks2, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, memPref);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_GRADS_BLANK_FAILED;

    if (U > 1)
    {

        dim3 blocks3((T + G - 1) / G, U - 1, N);
        kernel_grads_label_compact<<<blocks3, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, memPref, labelPref, fastemit_lambda);
        if (cudaGetLastError() != cudaSuccess)
            return RNNT_STATUS_GRADS_LABEL_FAILED;
    }

    dim3 blocks4((N + B - 1) / B, 1, 1);
    kernel_fill_costs_compact<<<blocks4, B, 0, stream>>>(costs, grads, alphas, betas, log_probs, xn, yn, memPref, N);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_COSTS_FAILED;

    return RNNT_STATUS_SUCCESS;
}
