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
    // assert(u <= U);
    // assert(gridDim.y == U);
    assert(blockDim.x == W);

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;

    // unsigned int *lock = counts + n * U * 2 + blockIdx.y;
    unsigned int *lock = counts + ((labelPref[n] + n) << 1) + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        // alphas[idx3(n, 0, 0, T, U)] = 0;
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

        // float a = alphas[idx3(n, 0, u - 1, T, U)];
        float a = alphas[mem_loc + u - 1];
        // float b = log_probs[idx4(n, 0, u - 1, 1, T, U, 2)];
        float b = log_probs[mem_beg + (u << 1) - 1]; // should be [mem_beg + 2 * (u-1) + 1] in a more readable manner.

        // alphas[idx3(n, 0, u, T, U)] = a + b;
        alphas[mem_loc + u] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t)
    {

        // Compute initial column with local scan algorithm

        float a;
        // float b = log_probs[idx4(n, t - 1, 0, 0, T, U, 2)];
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

        // a = alphas[idx3(n, p, 0, T, U)];
        a = alphas[mem_loc + p * actual_u];

        // alphas[idx3(n, t, 0, T, U)] = a + b;
        alphas[mem_loc + t * actual_u] = a + b;
    }

    if (t < actual_t && u < actual_u)
    {

        // Ready to compute alphas[t, u]

        // float bias = log_probs[idx4(n, t - 1, u, 0, T, U, 2)];
        float bias = log_probs[mem_beg + (((t - 1) * actual_u + u) << 1)];
        // float skip = alphas[idx3(n, p, u, T, U)] + bias;
        float skip = alphas[mem_loc + p * actual_u + u] + bias;
        // float emit = alphas[idx3(n, t, u - 1, T, U)] + log_probs[idx4(n, t, u - 1, 1, T, U, 2)];
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

        // alphas[idx3(n, t, u, T, U)] = output;
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
    // assert(u <= U);
    // assert(gridDim.y == U);
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

    // unsigned int *lock = counts + n * U * 2 + U + blockIdx.y;
    unsigned int *lock = counts + ((labelPref[n] + n) << 1) + actual_u + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        // betas[idx3(n, T1, U1, T, U)] = log_probs[idx4(n, T1, U1, 0, T, U, 2)];
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

        // float a = betas[idx3(n, T1, U1 - u + 1, T, U)];
        float a = betas[mem_loc + _val];
        // float b = log_probs[idx4(n, T1, U1 - u, 1, T, U, 2)];
        float b = log_probs[mem_beg + (_val << 1) - 1];

        // betas[idx3(n, T1, U1 - u, T, U)] = a + b;
        betas[mem_loc + _valm1] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t)
    {

        // Compute last column with local scan algorithm

        float a;
        // float b = log_probs[idx4(n, T1 - t, U1, 0, T, U, 2)];
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

        // a = betas[idx3(n, T1 - p, U1, T, U)];
        a = betas[mem_loc + _valm1 + u - p * actual_u];

        // betas[idx3(n, T1 - t, U1, T, U)] = a + b;
        betas[mem_loc + _valm1 + u - t * actual_u] = a + b;
    }

    if (t < actual_t && u < actual_u)
    {

        // Ready to compute betas[T1-t, U1-u]

        // float bias = log_probs[idx4(n, T1 - t, U1 - u, 0, T, U, 2)];
        float bias = log_probs[mem_beg + ((_valm1 - t * actual_u) << 1)];
        // float skip = betas[idx3(n, T1 - p, U1 - u, T, U)] + bias;
        float skip = betas[mem_loc + _valm1 - p * actual_u] + bias;
        // float emit = betas[idx3(n, T1 - t, U1 - u + 1, T, U)] + log_probs[idx4(n, T1 - t, U1 - u, 1, T, U, 2)];
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

        // betas[idx3(n, T1 - t, U1 - u, T, U)] = output;
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
        // kernel_warp_alphas(counts, alphas, log_probs, xn, yn, T, U);
        kernel_warp_alphas_compact(counts, alphas, log_probs, xn, yn, memPref, labelPref);
    }
    else if (threadIdx.y == 1)
    {
        // kernel_warp_betas(counts, betas, log_probs, xn, yn, T, U);
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

    // assert(u < U);
    assert(d < G);
    assert(blockDim.x == G);
    // assert(gridDim.y == U);

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t >= actual_t || u >= actual_u)
        return;

    if (t == actual_t - 1 && u < actual_u - 1)
        return;

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;
    // float a = alphas[idx3(n, t, u, T, U)];
    float a = alphas[mem_loc + t * actual_u + u];

    if (t < actual_t - 1)
    {
        // a += betas[idx3(n, t + 1, u, T, U)];
        a += betas[mem_loc + (t + 1) * actual_u + u];
    }

    // unsigned int index = idx4(n, t, u, 0, T, U, 2);
    unsigned int index = mem_beg + ((t * actual_u + u) << 1);

    // a = expf(a + log_probs[index] - betas[idx3(n, 0, 0, T, U)]);
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

    // assert(u < U - 1);
    assert(d < G);
    assert(blockDim.x == G);
    // assert(gridDim.y == U - 1);

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n];

    if (t >= actual_t || u >= actual_u)
        return;

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;
    unsigned int _index = t * (actual_u + 1) + u;

    // float a = alphas[idx3(n, t, u, T, U)] + betas[idx3(n, t, u + 1, T, U)];
    float a = alphas[mem_loc + _index] + betas[mem_loc + _index + 1];

    // unsigned int index = idx4(n, t, u, 1, T, U, 2);
    unsigned int index = mem_beg + (_index << 1) + 1;

    // a = expf(a + log_probs[index] - betas[idx3(n, 0, 0, T, U)]);
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

    // float a = alphas[idx3(n, t, u, T, U)] + log_probs[idx4(n, t, u, 0, T, U, 2)];
    float a = alphas[mem_loc + t * (u + 1) + u] + log_probs[mem_beg + ((t * (u + 1) + u) << 1)];
    // float b = betas[idx3(n, 0, 0, T, U)];
    float b = betas[mem_loc];

    float ratio = fabsf(a - b) / fabsf(fmaxf(a, b));

    if (ratio > 0.001)
    {

        printf("\nWARNING: sample %d [%d, %d] has a forward/backward mismatch %f / %f\n",
               n, t + 1, u, a, b);

        // float *g = grads + idx4(n, 0, 0, 0, T, U, 2);
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
    // kernel_warp<<<blocks1, threads1, 0, stream>>>(counts, alphas, betas, log_probs, xn, yn, T, U);
    kernel_warp_compact<<<blocks1, threads1, 0, stream>>>(counts, alphas, betas, log_probs, xn, yn, memPref, labelPref);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    dim3 blocks2((T + G - 1) / G, U, N);
    // kernel_grads_blank<<<blocks2, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, T, U);
    kernel_grads_blank_compact<<<blocks2, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, memPref);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_GRADS_BLANK_FAILED;

    if (U > 1)
    {

        dim3 blocks3((T + G - 1) / G, U - 1, N);
        // kernel_grads_label<<<blocks3, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, T, U, fastemit_lambda);
        kernel_grads_label_compact<<<blocks3, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, memPref, labelPref, fastemit_lambda);
        if (cudaGetLastError() != cudaSuccess)
            return RNNT_STATUS_GRADS_LABEL_FAILED;
    }

    dim3 blocks4((N + B - 1) / B, 1, 1);
    // kernel_fill_costs<<<blocks4, B, 0, stream>>>(costs, grads, alphas, betas, log_probs, xn, yn, N, T, U);
    kernel_fill_costs_compact<<<blocks4, B, 0, stream>>>(costs, grads, alphas, betas, log_probs, xn, yn, memPref, N);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_COSTS_FAILED;

    return RNNT_STATUS_SUCCESS;
}
