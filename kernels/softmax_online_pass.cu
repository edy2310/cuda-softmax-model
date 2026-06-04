#include <cuda_runtime.h>
#include <cmath>

// Online-pass softmax kernel.
// Assumes input is a 2D matrix in row-major order: [rows, cols].
// One block per row, threads cooperate to compute max and sum online.
__global__ void softmax_online_kernel(
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

    // Each thread processes a strided slice and keeps a local (max, sum).
    // The sum is maintained in a numerically stable "online" way.
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

    // Shared memory holds per-thread max and sum for reduction.
    extern __shared__ float shmem[];
    float* max_shared = shmem;
    float* sum_shared = shmem + blockDim.x;

    max_shared[threadIdx.x] = local_max;
    sum_shared[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce (max, sum) pairs across the block using the online combine rule.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float m1 = max_shared[threadIdx.x];
            float s1 = sum_shared[threadIdx.x];
            float m2 = max_shared[threadIdx.x + offset];
            float s2 = sum_shared[threadIdx.x + offset];

            float m = (m1 > m2) ? m1 : m2;
            float s = s1 * expf(m1 - m) + s2 * expf(m2 - m);

            max_shared[threadIdx.x] = m;
            sum_shared[threadIdx.x] = s;
        }
        __syncthreads();
    }

    float max_val = max_shared[0];
    float sum_val = sum_shared[0];

    // Second pass to write normalized probabilities.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - max_val) / sum_val;
    }
}

// Simple C-style launcher for the online-pass kernel.
extern "C" void softmax_online_cuda(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    dim3 grid(rows);
    dim3 block(256);
    size_t shmem_bytes = static_cast<size_t>(block.x) * 2 * sizeof(float);
    softmax_online_kernel<<<grid, block, shmem_bytes>>>(input, output, rows, cols);
}