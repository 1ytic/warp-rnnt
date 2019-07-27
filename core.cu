#include "core.h"

#define W 32
#define G 1024
#define B 256

__forceinline__ __device__ static int idx2(int n, int u, int U1) {
    return n * U1 + u;
}

__forceinline__ __device__ static int idx3(int n, int t, int u, int T, int U) {
    return n * (T * U) + t * U + u;
}

__forceinline__ __device__ static int idx4(int n, int t, int u, int v, int T, int U, int V) {
    return n * (T * U * V) + t * (U * V) + u * V + v;
}

__forceinline__ __device__ static float log_sum_exp(float a, float b) {
    float maximum, diff;
    if (a > b) {
        maximum = a;
        diff = b-a;
    } else {
        maximum = b;
        diff = a-b;
    }
    //if (diff > -42) {
        maximum += log1pf(expf(diff));
    //}
    return maximum;
}

__device__
void kernel_warp_alphas(unsigned int *counts, float *alphas, const int *labels, const float *log_probs,
                        const int *xn, const int *yn, int T, int U, int V, int blank) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    assert (u < U);
    assert (d < W);
    assert (blockDim.x == W);
    assert (gridDim.y == U - 1);

    int actual_t = xn[n];
    int actual_u = yn[n] + 1;

    if (t >= T || t >= actual_t || u >= actual_u)
        return;

    unsigned int l = labels[idx2(n, u-1, U-1)];

    unsigned int *lock_col = counts + n * U * 2 + u - 1;
    unsigned int *lock_row = counts + n * U * 2 + u;

    if (u > 1) {
        // Wait previous column
        do {} while (atomicAdd(lock_col, 0) <= g);
    }

    if (g == 0) {

        alphas[idx3(n, 0, 0, T, U)] = 0;

        // Compute initial row value

        float a = alphas[idx3(n, 0, u-1, T, U)];
        float b = log_probs[idx4(n, 0, u-1, l, T, U, V)];

        alphas[idx3(n, 0, u, T, U)] = a + b;
    }
    else {
        // Wait previous row
        do {} while (atomicAdd(lock_row, 0) < g);
    }

    unsigned int mask = __activemask();
    int w = __popc(mask);

    if (blockIdx.y == 0) {

        // Compute initial column with local scan algorithm

        float a;
        float b = log_probs[idx4(n, t-1, 0, blank, T, U, V)];

#pragma unroll
        for(unsigned int i = 1; i < W; i *= 2) {
            a = __shfl_up_sync(mask, b, i);
            if (i <= d) {
                b += a;
            }
        }

        a = alphas[idx3(n, p, 0, T, U)];

        alphas[idx3(n, t, 0, T, U)] = a + b;
    }

    // Ready to compute alphas[t, u]

    float bias = log_probs[idx4(n, t-1, u, blank, T, U, V)];
    float emit = alphas[idx3(n, t, u-1, T, U)] + log_probs[idx4(n, t, u-1, l, T, U, V)];

    float r = log_sum_exp(alphas[idx3(n, p, u, T, U)] + bias, emit);
    float output = r;

    for(unsigned int i = 1; i < W; i++) {
        r = __shfl_up_sync(mask, r, 1);
        if (i == d) {
            r = log_sum_exp(r + bias, emit);
            output = r;
        }
    }

    alphas[idx3(n, t, u, T, U)] = output;

    if (d == w - 1) {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock_row, 1);
    }
}

__device__
void kernel_warp_betas(unsigned int *counts, float *betas, const int *labels, const float *log_probs,
                       const int *xn, const int *yn, int T, int U, int V, int blank) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    assert (u < U);
    assert (d < W);
    assert (blockDim.x == W);
    assert (gridDim.y == U - 1);

    int actual_t = xn[n];
    int actual_u = yn[n] + 1;

    if (t >= T || t >= actual_t || u >= actual_u)
        return;

    unsigned int *lock_col = counts + n * U * 2 + U + u - 1;
    unsigned int *lock_row = counts + n * U * 2 + U + u;

    if (u > 1) {
        // Wait previous column
        do {} while (atomicAdd(lock_col, 0) <= g);
    }

    int T1 = actual_t - 1;
    int U1 = actual_u - 1;

    u = U1 - u;
    t = T1 - t;
    p = T1 - p;

    unsigned int l = labels[idx2(n, u, U-1)];

    if (g == 0) {

        betas[idx3(n, T1, U1, T, U)] = log_probs[idx4(n, T1, U1, blank, T, U, V)];

        // Compute last row value

        float a = betas[idx3(n, T1, u+1, T, U)];
        float b = log_probs[idx4(n, T1, u, l, T, U, V)];

        betas[idx3(n, T1, u, T, U)] = a + b;
    }
    else {
        // Wait previous row
        do {} while (atomicAdd(lock_row, 0) < g);
    }

    unsigned int mask = __activemask();
    int w = __popc(mask);

    if (blockIdx.y == 0) {

        // Compute last column with local scan algorithm

        float a;
        float b = log_probs[idx4(n, t, U1, blank, T, U, V)];

#pragma unroll
        for(unsigned int i = 1; i < W; i *= 2) {
            a = __shfl_up_sync(mask, b, i);
            if (i <= d) {
                b += a;
            }
        }

        a = betas[idx3(n, p, U1, T, U)];

        betas[idx3(n, t, U1, T, U)] = a + b;
    }

    // Ready to compute betas[t, u]

    float bias = log_probs[idx4(n, t, u, blank, T, U, V)];
    float emit = betas[idx3(n, t, u+1, T, U)] + log_probs[idx4(n, t, u, l, T, U, V)];

    float r = log_sum_exp(betas[idx3(n, p, u, T, U)] + bias, emit);
    float output = r;

    for(unsigned int i = 1; i < W; i++) {
        r = __shfl_up_sync(mask, r, 1);
        if (i == d) {
            r = log_sum_exp(r + bias, emit);
            output = r;
        }
    }

    betas[idx3(n, t, u, T, U)] = output;

    if (d == w - 1) {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock_row, 1);
    }
}

__global__
void kernel_warp(unsigned int *counts, float *alphas, float *betas, const int *labels, const float *log_probs,
                       const int *xn, const int *yn, int T, int U, int V, int blank) {
    if (threadIdx.y == 0) {
        kernel_warp_alphas(counts, alphas, labels, log_probs, xn, yn, T, U, V, blank);
    }
    else if (threadIdx.y == 1) {
        kernel_warp_betas(counts, betas, labels, log_probs, xn, yn, T, U, V, blank);
    }
}

__global__
void kernel_grads_blank(float *grads, const float *alphas, const float *betas, const float *log_probs,
                        const int *xn, const int *yn, int T, int U, int V, int blank) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = g * G + d;

    assert (u < U);
    assert (d < G);
    assert (blockDim.x == G);
    assert (gridDim.y == U);

    int actual_t = xn[n];
    int actual_u = yn[n] + 1;

    if (t >= T || t >= actual_t || u >= actual_u)
        return;

    if (t == actual_t-1 && u < actual_u-1)
        return;

    float a = alphas[idx3(n, t, u, T, U)];

    if (t < actual_t-1) {
        a += betas[idx3(n, t+1, u, T, U)];
    }

    unsigned int index = idx4(n, t, u, blank, T, U, V);

    a = expf(a + log_probs[index] - betas[idx3(n, 0, 0, T, U)]);

    grads[index] = -a;
}

__global__
void kernel_grads_label(float *grads, const float *alphas, const float *betas,
                        const int *labels, const float *log_probs,
                        const int *xn, const int *yn, int T, int U, int V) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = g * G + d;

    assert (u < U - 1);
    assert (d < G);
    assert (blockDim.x == G);
    assert (gridDim.y == U - 1);

    int actual_t = xn[n];
    int actual_u = yn[n];

    if (t >= T || t >= actual_t || u >= actual_u)
        return;

    unsigned int l = labels[idx2(n, u, U-1)];

    float a = alphas[idx3(n, t, u, T, U)] + betas[idx3(n, t, u+1, T, U)];

    unsigned int index = idx4(n, t, u, l, T, U, V);

    a = expf(a + log_probs[index] - betas[idx3(n, 0, 0, T, U)]);

    grads[index] = -a;
}

__global__
void kernel_fill_costs(float *costs, float *grads, const float *alphas, const float *betas, const float *log_probs,
                       const int *xn, const int *yn, int N, int T, int U, int V, int blank) {

    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N)
        return;

    int t = xn[n] - 1;
    int u = yn[n];

    float a = alphas[idx3(n, t, u, T, U)] + log_probs[idx4(n, t, u, blank, T, U, V)];
    float b = betas[idx3(n, 0, 0, T, U)];

    float ratio = fabsf(a - b) / fabsf(fmaxf(a, b));

    if (ratio > 0.001) {

        printf("\nWARNING: sample %d [%d, %d] has a forward/backward mismatch %f / %f\n",
                n, t + 1, u, a, b);

        float *g = grads + idx4(n, 0, 0, 0, T, U, V);

        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < U; ++j) {
                for (int v = 0; v < V; ++v, ++g) {
                    *g = 0;
                }
            }
        }

        b = (a + b) / 2.0f;
    }

    costs[n] = -b;
}

rnntStatus_t run_warp_rnnt(cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
                           const int *labels, const float *log_probs, float *grads, float *costs,
                           const int *xn, const int *yn, int N, int T, int U, int V, int blank) {

    dim3 threads1(W, 2);

    dim3 blocks1((T - 1 + W - 1) / W, U - 1, N);
    dim3 blocks2((T + G - 1) / G, U, N);
    dim3 blocks3((T + G - 1) / G, U - 1, N);
    dim3 blocks4((N + B - 1) / B, 1, 1);

    kernel_warp <<<blocks1, threads1, 0, stream>>> (counts, alphas, betas, labels, log_probs, xn, yn, T, U, V, blank);

    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_WARP_FAILED;

    kernel_grads_blank <<<blocks2, G, 0, stream>>> (grads, alphas, betas, log_probs, xn, yn, T, U, V, blank);

    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_GRADS_BLANK_FAILED;

    kernel_grads_label <<<blocks3, G, 0, stream>>> (grads, alphas, betas, labels, log_probs, xn, yn, T, U, V);

    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_GRADS_LABEL_FAILED;

    kernel_fill_costs <<<blocks4, B, 0, stream>>> (costs, grads, alphas, betas, log_probs, xn, yn, N, T, U, V, blank);

    if (cudaGetLastError() != cudaSuccess)
        return RNNT_STATUS_COSTS_FAILED;

    return RNNT_STATUS_SUCCESS;
}
