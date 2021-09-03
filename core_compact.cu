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
    // log(exp(a)+exp(b)) = log(exp(a)(1+exp(b-a))) = a + log(1+exp(b-a)) = a + log1pf(expf(b-a))
    return maximum;
}

__device__ void kernel_warp_alphas_compact(unsigned int *counts, volatile float *alphas,
                                           const int *labels, const float *log_probs,
                                           const int *xn, const int *yn, const int *memPref, const int *labelPref,
                                           int V, int blank)
{

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    assert(d < W);
    // assert(u <= Um);
    // assert(gridDim.y == Um);
    assert(blockDim.x == W);

    int actual_t = xn[n];
    int actual_u = yn[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    unsigned int mem_beg = memPref[n];
    unsigned int mem_loc = mem_beg / V;

    // unsigned int *lock = counts + n * Um * 2 + blockIdx.y;
    unsigned int *lock = counts + (labelPref[n] + n) * 2 + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        // initialize the state as log(p) = 0.
        // alphas[idx3(n, 0, 0, Tm, Um)] = 0;
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

        // unsigned int l = labels[idx2(n, u - 1, Um - 1)];
        unsigned int l = labels[labelPref[n] + u - 1];

        // float a = alphas[idx3(n, 0, u - 1, Tm, Um)];
        float a = alphas[mem_loc + u - 1];
        // float b = log_probs[idx4(n, 0, u - 1, l, Tm, Um, V)];
        float b = log_probs[mem_beg + (u - 1) * V + l];

        // alphas[idx3(n, 0, u, Tm, Um)] = a + b;
        alphas[mem_loc + u] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t)
    {

        // Compute initial column with local scan algorithm

        float a;
        // float b = log_probs[idx4(n, t - 1, 0, blank, Tm, Um, V)];
        float b = log_probs[mem_beg + (t - 1) * actual_u * V + blank];

#pragma unroll
        for (unsigned int i = 1; i < W; i *= 2)
        {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d)
            {
                b += a;
            }
        }

        // a = alphas[idx3(n, p, 0, Tm, Um)];
        a = alphas[mem_loc + p * actual_u];

        // alphas[idx3(n, t, 0, Tm, Um)] = a + b;
        alphas[mem_loc + t * actual_u] = a + b;
    }

    if (t < actual_t && u < actual_u)
    {

        // Ready to compute alphas[t, u]

        // unsigned int l = labels[idx2(n, u - 1, Um - 1)];
        unsigned int l = labels[labelPref[n] + u - 1];

        // float bias = log_probs[idx4(n, t - 1, u, blank, Tm, Um, V)];
        float bias = log_probs[mem_beg + (t - 1) * actual_u * V + u * V + blank];
        // float skip = alphas[idx3(n, p, u, Tm, Um)] + bias;
        float skip = alphas[mem_loc + p * actual_u + u] + bias;
        // float emit = alphas[idx3(n, t, u - 1, Tm, Um)] + log_probs[idx4(n, t, u - 1, l, Tm, Um, V)];
        float emit = alphas[mem_loc + t * actual_u + u - 1] + log_probs[mem_beg + t * actual_u * V + (u - 1) * V + l];

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

        // alphas[idx3(n, t, u, Tm, Um)] = output;
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
                                          const int *labels, const float *log_probs,
                                          const int *xn, const int *yn, const int *memPref, const int *labelPref,
                                          int V, int blank)
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

    int actual_t = xn[n];
    int actual_u = yn[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    int T1 = actual_t - 1;
    int U1 = actual_u - 1;
    unsigned int mem_beg = memPref[n];
    unsigned int mem_loc = mem_beg / V;

    // unsigned int *lock = counts + n * U * 2 + U + blockIdx.y;
    unsigned int *lock = counts + (labelPref[n] + n) * 2 + actual_u + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        // betas[idx3(n, T1, U1, T, U)] = log_probs[idx4(n, T1, U1, blank, T, U, V)];
        betas[mem_loc + T1 * actual_u + U1] = log_probs[mem_beg + T1 * actual_u * V + U1 * V + blank];
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

        // unsigned int l = labels[idx2(n, U1 - u, U - 1)];
        unsigned int l = labels[labelPref[n] + U1 - u];

        // float a = betas[idx3(n, T1, U1 - u + 1, T, U)];
        float a = betas[mem_loc + T1 * actual_u + U1 - u + 1];
        // float b = log_probs[idx4(n, T1, U1 - u, l, T, U, V)];
        float b = log_probs[mem_beg + T1 * actual_u * V + (U1 - u) * V + l];

        // betas[idx3(n, T1, U1 - u, T, U)] = a + b;
        betas[mem_loc + T1 * actual_u + U1 - u] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t)
    {

        // Compute last column with local scan algorithm

        float a;
        // float b = log_probs[idx4(n, T1 - t, U1, blank, T, U, V)];
        float b = log_probs[mem_beg + (T1 - t) * actual_u * V + U1 * V + blank];

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
        a = betas[mem_loc + (T1 - p) * actual_u + U1];

        // betas[idx3(n, T1 - t, U1, T, U)] = a + b;
        betas[mem_loc + (T1 - t) * actual_u + U1] = a + b;
    }

    if (t < actual_t && u < actual_u)
    {

        // Ready to compute betas[T1-t, U1-u]

        // unsigned int l = labels[idx2(n, U1 - u, U - 1)];
        unsigned int l = labels[labelPref[n] + U1 - u];

        // float bias = log_probs[idx4(n, T1 - t, U1 - u, blank, T, U, V)];
        float bias = log_probs[mem_beg + (T1 - t) * actual_u * V + (U1 - u) * V + blank];
        // float skip = betas[idx3(n, T1 - p, U1 - u, T, U)] + bias;
        float skip = betas[mem_loc + (T1 - p) * actual_u + U1 - u] + bias;
        // float emit = betas[idx3(n, T1 - t, U1 - u + 1, T, U)] + log_probs[idx4(n, T1 - t, U1 - u, l, T, U, V)];
        float emit = betas[mem_loc + (T1 - t) * actual_u + U1 - u + 1] + log_probs[mem_beg + (T1 - t) * actual_u * V + (U1 - u) * V + l];

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
        betas[mem_loc + (T1 - t) * actual_u + U1 - u] = output;
    }

    if (d == 0)
    {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__global__ void kernel_warp_compact(unsigned int *counts, volatile float *alphas, volatile float *betas,
                                    const int *labels, const float *log_probs,
                                    const int *xn, const int *yn, const int *memPref, const int *labelPref,
                                    int V, int blank)
{
    if (threadIdx.y == 0)
    {
        kernel_warp_alphas_compact(counts, alphas, labels, log_probs, xn, yn, memPref, labelPref, V, blank);
        // kernel_warp_alphas_compact(counts, alphas, labels, log_probs, xn, yn, Tm, Um, V, blank);
    }
    else if (threadIdx.y == 1)
    {
        kernel_warp_betas_compact(counts, betas, labels, log_probs, xn, yn, memPref, labelPref, V, blank);
        // kernel_warp_betas_compact(counts, betas, labels, log_probs, xn, yn, Tm, Um, V, blank);
    }
}

__global__ void kernel_grads_blank_compact(float *grads, const float *alphas, const float *betas, const float *log_probs,
                                           const int *xn, const int *yn, const int *memPref, int V, int blank)
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

    int actual_t = xn[n];
    int actual_u = yn[n] + 1;

    if (t >= actual_t || u >= actual_u)
        return;

    if (t == actual_t - 1 && u < actual_u - 1)
        return;

    int mem_beg = memPref[n];
    int mem_loc = mem_beg / V;
    // float a = alphas[idx3(n, t, u, T, U)];
    float a = alphas[mem_loc + t * actual_u + u];

    if (t < actual_t - 1)
    {
        // a += betas[idx3(n, t + 1, u, T, U)];
        a += betas[mem_loc + (t + 1) * actual_u + u];
    }

    // unsigned int index = idx4(n, t, u, blank, T, U, V);
    unsigned int index = mem_beg + t * actual_u * V + u * V + blank;

    // a = expf(a + log_probs[index] - betas[idx3(n, 0, 0, T, U)]);
    a = expf(a + log_probs[index] - betas[mem_loc]);

    grads[index] = -a;
}

__global__ void kernel_grads_label_compact(float *grads, const float *alphas, const float *betas,
                                           const int *labels, const float *log_probs,
                                           const int *xn, const int *yn, const int *memPref, const int *labelPref,
                                           int V, float fastemit_lambda)
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

    int actual_t = xn[n];
    int actual_u = yn[n];

    if (t >= actual_t || u >= actual_u)
        return;

    unsigned int mem_beg = memPref[n];
    unsigned int mem_loc = mem_beg / V;
    unsigned int _index = t * (actual_u + 1) + u;

    // unsigned int l = labels[idx2(n, u, U - 1)];
    unsigned int l = labels[labelPref[n] + u];

    // float a = alphas[idx3(n, t, u, T, U)] + betas[idx3(n, t, u + 1, T, U)];
    float a = alphas[mem_loc + _index] + betas[mem_loc + _index + 1];

    // unsigned int index = idx4(n, t, u, l, T, U, V);
    unsigned int index = mem_beg + _index * V + l;

    // a = expf(a + log_probs[index] - betas[idx3(n, 0, 0, T, U)]);
    a = expf(a + log_probs[index] - betas[mem_loc]);

    // apply FastEmit regularization
    // https://arxiv.org/abs/2010.11148
    a = (1. + fastemit_lambda) * a;

    grads[index] = -a;
}

__global__ void kernel_fill_costs_compact(float *costs, float *grads, const float *alphas, const float *betas, const float *log_probs,
                                          const int *xn, const int *yn, const int *memPref,
                                          int N, int V, int blank)
{

    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N)
        return;

    int t = xn[n] - 1;
    int u = yn[n];
    unsigned int mem_beg = memPref[n];
    unsigned int mem_loc = mem_beg / V;

    /*
     * According to transducer forward/backward algorithm, 
     * ... the cost (i.e. the negative log likelihood) is equal to
     *     \alpha(T-1, U-1) * \phi(T-1,U-1), also equal to \beta(0,0)
     * ... where the \phi(t, u) indicates the probability of emittitng blank
     * ... at the position (t, u). So here is a assertion whether these two variables
     * ... are equal, if not, empty all the gradients.
     */
    // float a = alphas[idx3(n, t, u, T, U)] + log_probs[idx4(n, t, u, blank, T, U, V)];
    float a = alphas[mem_loc + t * (u + 1) + u] + log_probs[mem_beg + t * (u + 1) * V + u * V + blank];
    // float b = betas[idx3(n, 0, 0, T, U)];
    float b = betas[mem_loc];

    float ratio = fabsf(a - b) / fabsf(fmaxf(a, b));

    // FIXME (maxwellzh): I think the ratio in logarithm is too strict.
    if (ratio > 0.001)
    {

        printf("\nWARNING: sample %d [%d, %d] has a forward/backward mismatch %f / %f\n",
               n, t + 1, u, a, b);

        // float *g = grads + idx4(n, 0, 0, 0, T, U, V);
        float *g = grads + mem_beg;

        for (int i = 0; i < t + 1; ++i)
        {
            for (int j = 0; j < u + 1; ++j)
            {
                for (int v = 0; v < V; ++v, ++g)
                {
                    *g = 0.0f;
                }
            }
        }
        b = (a + b) / 2.0f;
    }

    costs[n] = -b;
}

rnntStatus_t run_warp_rnnt_compact(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                                   const int *labels, const float *log_probs, float *grads, float *costs,
                                   const int *xn, const int *yn, const int *memPref, const int *labelPref,
                                   int N, int Tm, int Um, int V, int blank, float fastemit_lambda)
{

    // counts: (2*\sum_{U_i+1},)
    // alphas/betas: (\sum_{T_i*(U_i+1)}, )
    // log_probs/grads: (\sum_{T_i*(U_i+1)}, V)
    // costs: (N, )
    // labelPref: (N, )

    dim3 threads1(W, 2);
    dim3 blocks1((Tm + W - 1) / W, Um, N);
    kernel_warp_compact<<<blocks1, threads1, 0, stream>>>(counts, alphas, betas, labels, log_probs, xn, yn, memPref, labelPref, V, blank);
    // kernel_warp_compact<<<blocks1, threads1, 0, stream>>>(counts, alphas, betas, labels, log_probs, xn, yn, Tm, Um, V, blank);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    dim3 blocks2((Tm + G - 1) / G, Um, N);
    kernel_grads_blank_compact<<<blocks2, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, memPref, V, blank);
    // kernel_grads_blank_compact<<<blocks2, G, 0, stream>>>(grads, alphas, betas, log_probs, xn, yn, Tm, Um, V, blank);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_GRADS_BLANK_FAILED;

    if (Um > 1)
    {

        dim3 blocks3((Tm + G - 1) / G, Um - 1, N);
        kernel_grads_label_compact<<<blocks3, G, 0, stream>>>(grads, alphas, betas, labels, log_probs, xn, yn, memPref, labelPref, V, fastemit_lambda);
        // kernel_grads_label_compact<<<blocks3, G, 0, stream>>>(grads, alphas, betas, labels, log_probs, xn, yn, Tm, Um, V, fastemit_lambda);
        if (cudaGetLastError() != cudaSuccess)
            return RNNT_STATUS_GRADS_LABEL_FAILED;
    }

    dim3 blocks4((N + B - 1) / B, 1, 1);
    kernel_fill_costs_compact<<<blocks4, B, 0, stream>>>(costs, grads, alphas, betas, log_probs, xn, yn, memPref, N, V, blank);
    // kernel_fill_costs_compact<<<blocks4, B, 0, stream>>>(costs, grads, alphas, betas, log_probs, xn, yn, N, Tm, Um, V, blank);
    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_COSTS_FAILED;

    return RNNT_STATUS_SUCCESS;
}
