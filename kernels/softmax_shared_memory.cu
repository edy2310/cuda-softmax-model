#include <cuda_runtime.h>
#include <cmath>

// Shared-memory softmax kernel
// Assumes input is a 2D matrix in row-major order: [rows, cols].
// One block per row, multiple threads cooperate to compute the softmax.
__global__ void softmax_shared_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    // Shared memory layout:
    // [exp_values_with_padding | partial_sums]
    extern __shared__ float shmem[];
    int padding = cols / 32;           // simple padding to reduce bank conflicts
    int stride = cols + padding;
    float* exp_shared = shmem;
    float* sum_shared = exp_shared + stride;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Each thread computes exp for its elements and accumulates a partial sum.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        int idx = i + i / 32;          // padded index in shared memory
        float v = expf(in_row[i]);
        exp_shared[idx] = v;
        local_sum += v;
    }

    // Reduce partial sums in shared memory.
    sum_shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sum_shared[threadIdx.x] += sum_shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    float sum = sum_shared[0];

    // Normalize to get probabilities.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        int idx = i + i / 32;
        out_row[i] = exp_shared[idx] / sum;
    }
}

// Simple C-style launcher for the shared-memory kernel.
extern "C" void softmax_shared_cuda(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    dim3 grid(rows);
    dim3 block(256);
    int padding = cols / 32;
    int stride = cols + padding;
    size_t shmem_bytes = static_cast<size_t>(stride + block.x) * sizeof(float);
    softmax_shared_kernel<<<grid, block, shmem_bytes>>>(input, output, rows, cols);
}
