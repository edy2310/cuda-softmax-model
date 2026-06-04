#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction for (max, sum) using the online combine rule.
// This keeps numerical stability while reducing within a warp.
__device__ inline void warp_reduce_max_sum(float& max_val, float& sum_val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);

        float m = (max_val > other_max) ? max_val : other_max;
        float s;
        if (m == -INFINITY) {
            s = 0.0f;
        } else {
            float term_self = (max_val == -INFINITY) ? 0.0f : sum_val * expf(max_val - m);
            float term_other = (other_max == -INFINITY) ? 0.0f : other_sum * expf(other_max - m);
            s = term_self + term_other;
        }

        max_val = m;
        sum_val = s;
    }
}

// Softmax with online pass + warp primitives (simple, not fully optimized).
// Assumes input is a 2D matrix in row-major order: [rows, cols].
// One block per row, threads cooperate using warp-level reductions.
__global__ void softmax_warp_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Each thread processes a strided slice with an online (max, sum).
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float x = in_row[i];
        if (x > local_max) {
            local_sum = local_sum * expf(local_max - x) + 1.0f;
            local_max = x;
        } else {
            local_sum += expf(x - local_max);
        }
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    // Reduce within each warp using warp shuffle primitives.
    // Skip empty warps to avoid (-inf - -inf) -> NaN during reductions.
    bool warp_has_data = (warp_id * 32) < cols;
    if (warp_has_data) {
        warp_reduce_max_sum(local_max, local_sum);
    } else {
        local_max = -INFINITY;
        local_sum = 0.0f;
    }

    // Store one (max, sum) per warp in shared memory.
    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp results.
    float block_max = -INFINITY;
    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        if (lane_id < num_warps) {
            block_max = warp_max[lane_id];
            block_sum = warp_sum[lane_id];
        }
        warp_reduce_max_sum(block_max, block_sum);
    }

    // Broadcast the final (max, sum) to all threads in the block.
    __shared__ float final_max;
    __shared__ float final_sum;
    if (threadIdx.x == 0) {
        final_max = block_max;
        final_sum = block_sum;
    }
    __syncthreads();

    // Write normalized probabilities.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - final_max) / final_sum;
    }
}

// Simple C-style launcher for the warp-primitive kernel.
extern "C" void softmax_warp_cuda(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    dim3 grid(rows);
    dim3 block(256);
    softmax_warp_kernel<<<grid, block>>>(input, output, rows, cols);
}