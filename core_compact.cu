#include "core.h"

#include <algorithm>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define G 1024
#define WL 512
#define B 256
#define W 32
#define H 16

__forceinline__ __device__ static float logaddexpf(float a, float b) {
    float const tmp = a - b;

    if (a == b)
        return (float)(a + M_LN2);

    if (tmp > 0)
        return a + log1pf(expf(-tmp));
    else if (tmp <= 0)
        return b + log1pf(expf(tmp));
    // in case of overflow
    return tmp;
}

__global__ void
kernel_warp_alphas_compact(unsigned int *counts, volatile float *alphas,
                           const float *log_probs, const unsigned int *xn,
                           const unsigned int *yn, const unsigned int *memPref,
                           const unsigned int *labelPref) {

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int n = blockIdx.z;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;

    unsigned int *lock = counts + ((labelPref[n] + n) << 1) + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        // initialize the state as log(p) = 0.
        // alphas[n, 0, 0] = 0;
        alphas[mem_loc] = 0.0f;
    }

    if (blockIdx.x > 0) {
        // Wait previous row
        do {
        } while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0) {
        // Wait previous column
        do {
        } while (atomicAdd(lock - 1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u) {

        // Compute initial row value

        // a = alphas[n, 0, u-1]
        float a = alphas[mem_loc + u - 1];

        // b = log_probs[n, 0, u-1, 1]
        float b = log_probs[mem_beg + (u << 1) -
                            1]; // should be [mem_beg + 2 * (u-1) +
                                // 1] in a more readable manner.

        // alphas[n, 0, u] = a + b
        alphas[mem_loc + u] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t) {

        // Compute initial column with local scan algorithm

        float a;

        // b = log_probs[n, t-1, 0, 0]
        float b = log_probs[mem_beg + ((t - 1) * actual_u << 1)];

#pragma unroll
        for (unsigned int i = 1; i < W; i *= 2) {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d) {
                b += a;
            }
        }

        // a = alphas[n, p, 0]
        a = alphas[mem_loc + p * actual_u];

        // alphas[n, t, 0] = a + b;
        alphas[mem_loc + t * actual_u] = a + b;
    }

    if (t < actual_t && u < actual_u) {

        // Ready to compute alphas[t, u]

        // bias = log_probs[n, t-1, u, 0]
        float bias = log_probs[mem_beg + (((t - 1) * actual_u + u) << 1)];

        // skip = alphas[n, p, u] + bias
        float skip = alphas[mem_loc + p * actual_u + u] + bias;

        // emit = alphas[n, t, u-1] + log_probs[n, t, u-1, 1]
        float emit = alphas[mem_loc + t * actual_u + u - 1] +
                     log_probs[mem_beg + ((t * actual_u + u) << 1) - 1];

        float r = logaddexpf(skip, emit);
        float output = r;

        for (unsigned int i = 1; i < W; i++) {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d) {
                r = logaddexpf(r + bias, emit);
                output = r;
            }
        }

        // alphas[n, t, u] = output
        alphas[mem_loc + t * actual_u + u] = output;
    }

    if (d == 0) {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__global__ void
kernel_warp_betas_compact(unsigned int *counts, volatile float *betas,
                          const float *log_probs, const unsigned int *xn,
                          const unsigned int *yn, const unsigned int *memPref,
                          const unsigned int *labelPref) {

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

    unsigned int *lock =
        counts + ((labelPref[n] + n) << 1) + actual_u + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        // betas[n, T1, U1] = log_probs[n, T1, U1, 0]
        betas[mem_loc + _valm1 + u] = log_probs[mem_beg + ((_valm1 + u) << 1)];
    }

    if (blockIdx.x > 0) {
        // Wait previous row
        do {
        } while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0) {
        // Wait previous column
        do {
        } while (atomicAdd(lock - 1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u) {

        // Compute last row value

        // a = betas[n, T1, U1-u+1]
        float a = betas[mem_loc + _val];

        // b = log_probs[n, T1, U1-u, 1]
        float b = log_probs[mem_beg + (_val << 1) - 1];

        // betas[n, T1, U1-u] = a + b
        betas[mem_loc + _valm1] = a + b;
    }

    if (blockIdx.y == 0 && t < actual_t) {

        // Compute last column with local scan algorithm

        float a;

        // b = log_probs[n, T1-t, U1, 0]
        float b = log_probs[mem_beg + ((_valm1 + u - t * actual_u) << 1)];

#pragma unroll
        for (unsigned int i = 1; i < W; i *= 2) {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d) {
                b += a;
            }
        }

        // a = betas[n, T1-p, U1]
        a = betas[mem_loc + _valm1 + u - p * actual_u];

        // betas[n, T1 - t, U1] = a + b;
        betas[mem_loc + _valm1 + u - t * actual_u] = a + b;
    }

    if (t < actual_t && u < actual_u) {

        // Ready to compute betas[T1-t, U1-u]

        // bias = log_probs[n, T1 - t, U1 - u, 0];
        float bias = log_probs[mem_beg + ((_valm1 - t * actual_u) << 1)];

        // skip = betas[n, T1 - p, U1 - u] + bias;
        float skip = betas[mem_loc + _valm1 - p * actual_u] + bias;

        // emit = betas[n, T1 - t, U1 - u + 1] + log_probs[n, T1 - t, U1 - u,
        // 1];
        float emit = betas[mem_loc + _val - t * actual_u] +
                     log_probs[mem_beg + ((_val - t * actual_u) << 1) - 1];

        float r = logaddexpf(skip, emit);
        float output = r;

        for (unsigned int i = 1; i < W; i++) {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d) {
                r = logaddexpf(r + bias, emit);
                output = r;
            }
        }

        // betas[n, T1 - t, U1 - u] = output;
        betas[mem_loc + _valm1 - t * actual_u] = output;
    }

    if (d == 0) {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__global__ void kernel_grads_blank_compact(float *grads, const float *alphas,
                                           const float *betas,
                                           const float *log_probs,
                                           const unsigned int *xn,
                                           const unsigned int *yn,
                                           const unsigned int *memPref) {

    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int tmax = xn[n] - 1;
    unsigned int actual_u = yn[n] + 1;

    if (t > tmax || u >= actual_u)
        return;

    if (t == tmax && u < actual_u - 1) {
        grads[(memPref[n] + t * actual_u + u) << 1] = 0.0f;
        return;
    }

    unsigned int mem_loc = memPref[n];
    // a = alphas[n, t, u];
    float a = alphas[mem_loc + t * actual_u + u];

    if (t < tmax) {
        // a += betas[n, t + 1, u];
        a += betas[mem_loc + (t + 1) * actual_u + u];
    }

    // index = (n, t, u, 0);
    unsigned int index = (mem_loc + (t * actual_u + u)) << 1;

    // -expf(a + log_probs[index] - betas[n, 0, 0]);
    grads[index] = -expf(a + log_probs[index] - betas[mem_loc]);
}

__global__ void kernel_grads_label_compact(
    float *grads, const float *alphas, const float *betas,
    const float *log_probs, const unsigned int *xn, const unsigned int *yn,
    const unsigned int *memPref, const unsigned int *labelPref,
    float fastemit_lambda) {

    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= xn[n] || u > yn[n])
        return;

    if (u == yn[n]) {
        grads[((memPref[n] + t * (u + 1) + u) << 1) + 1] = 0.0f;
        return;
    }

    unsigned int mem_loc = memPref[n];
    unsigned int mem_beg = mem_loc << 1;
    unsigned int _index = t * (yn[n] + 1) + u;

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

__global__ void kernel_fill_costs_compact(float *costs, const float *betas,
                                          const unsigned int *memPref,
                                          unsigned int N) {

    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N)
        return;

    // b = betas[n, 0, 0]
    costs[n] = -betas[memPref[n]];
}

void run_warp_rnnt_compact(unsigned int *counts, float *alphas, float *betas,
                           const float *log_probs, float *grads, float *costs,
                           const unsigned int *xn, const unsigned int *yn,
                           const unsigned int *memPref,
                           const unsigned int *labelPref, unsigned int N,
                           unsigned int T, unsigned int U,
                           float fastemit_lambda, bool required_grad) {

    dim3 threads1(W);
    dim3 blocks1((T + W - 1) / W, U, N);

    // if not require grad, cal beta only, useful in inference mode
    kernel_warp_betas_compact<<<blocks1, threads1>>>(
        counts, betas, log_probs, xn, yn, memPref, labelPref);
    CHECK_KERNEL_STAT("rnnt loss compact betas");

    dim3 blocks4((N + B - 1) / B, 1, 1);
    kernel_fill_costs_compact<<<blocks4, B>>>(costs, betas, memPref, N);
    CHECK_KERNEL_STAT("rnnt loss compact filling costs")

    if (required_grad) {
        kernel_warp_alphas_compact<<<blocks1, threads1>>>(
            counts, alphas, log_probs, xn, yn, memPref, labelPref);
        CHECK_KERNEL_STAT("rnnt loss compact alphas")

        dim3 blocks2((T + G - 1) / G, U, N);
        kernel_grads_blank_compact<<<blocks2, G>>>(grads, alphas, betas,
                                                   log_probs, xn, yn, memPref);
        CHECK_KERNEL_STAT("rnnt loss compact computing gradients for blank")

        if (U > 1) {
            dim3 blocks3((T + G - 1) / G, U - 1, N);
            kernel_grads_label_compact<<<blocks3, G>>>(
                grads, alphas, betas, log_probs, xn, yn, memPref, labelPref,
                fastemit_lambda);
            CHECK_KERNEL_STAT(
                "rnnt loss compact computing gradients for labels")
        }
    }

    return;
}

__global__ void kernel_fill_gather(const float *xs, const int *ys,
                                   const unsigned int *xn,
                                   const unsigned int *yn, float *gather_xs,
                                   long *loc, const unsigned int *memPref,
                                   const unsigned int *labelPref,
                                   unsigned int V, unsigned int blank) {
    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int u = blockIdx.y * H + threadIdx.y;
    unsigned int n = blockIdx.z;

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n] + 1;

    if (t >= actual_t || u >= actual_u)
        return;

    unsigned int mem_loc = memPref[n];

    // l = ys(n, u)
    unsigned int _index = mem_loc + t * actual_u + u;
    float *ptr_gather = gather_xs + (_index << 1);
    // gather_xs(n, t, u, 0) = xs(n, t, u, blank)
    *(ptr_gather++) = xs[_index * V + blank];
    unsigned int l;
    if (u == yn[n]) {
        // last row
        l = blank;
    } else {
        l = ys[labelPref[n] + u];
    }
    loc[_index] = l;
    // gather_xs(n, t, u, 1) = xs(n, t, u, l)
    *ptr_gather = xs[_index * V + l];
}

void run_gather_for_compact(const float *xs, const int *ys,
                            const unsigned int *xn, const unsigned int *yn,
                            float *gather_xs, long *loc,
                            const unsigned int *memPref,
                            const unsigned int *labelPref, unsigned int N,
                            unsigned int T, unsigned int U, unsigned int V,
                            unsigned int blank) {

    dim3 threads(W, H);
    dim3 blocks((T + W - 1) / W, (U + H - 1) / H, N);

    kernel_fill_gather<<<blocks, threads>>>(xs, ys, xn, yn, gather_xs, loc,
                                            memPref, labelPref, V, blank);
    CHECK_KERNEL_STAT("rnnt loss gather for compact")

    return;
}

__global__ void kernel_fill_scatter_grad(const float *grad_cost,
                                         const float *gather_grad,
                                         const long *loc, const int *cum_lens,
                                         float *scatter_grad, unsigned int STU,
                                         unsigned int V, unsigned int N,
                                         unsigned int blank) {
    unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * W + threadIdx.x;
    if (i >= STU)
        return;

    // must be signed int
    int l = 0;
    int r = N - 1;
    // we need to clarify which batch the thread-i belongs to.
    unsigned int n;
    while (l <= r) {
        n = l + (r - l) / 2;
        if (i >= cum_lens[n]) {
            l = n + 1;
        } else {
            r = n - 1;
        }
    }
    n = l;

    scatter_grad[i * V + blank] = gather_grad[i << 1] * grad_cost[n];
    if (loc[i] != blank)
        scatter_grad[i * V + loc[i]] = gather_grad[(i << 1) + 1] * grad_cost[n];
}

void run_scatter_grad_for_compact(const float *grad_cost,
                                  const float *gather_grad, const long *loc,
                                  const int *cum_lens, float *scatter_grad,
                                  unsigned int STU, unsigned int N,
                                  unsigned int V, unsigned int blank) {
    dim3 threads(W);
    // avoid dim-x to be too large
    dim3 blocks(((W + STU - 1) / W + W - 1) / W, W);

    kernel_fill_scatter_grad<<<blocks, threads>>>(
        grad_cost, gather_grad, loc, cum_lens, scatter_grad, STU, V, N, blank);
    CHECK_KERNEL_STAT("rnnt loss filling scatter grad")

    return;
}
