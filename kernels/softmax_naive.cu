#include <cuda_runtime.h>
#include <cmath>

// Naive softmax kernel.
// Assumes input is a 2D matrix in row-major order: [rows, cols].
// Each block handles one row, and a single thread computes the entire row.
__global__ void softmax_naive_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    // Only one thread does all the work (slow but very simple).
    if (threadIdx.x == 0) {
        const float* in_row = input + row * cols;
        float* out_row = output + row * cols;

        // Compute sum of exp for this row.
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            sum += expf(in_row[i]);
        }

        // Normalize to get probabilities.
        for (int i = 0; i < cols; ++i) {
            out_row[i] = expf(in_row[i]) / sum;
        }
    }
}

// Simple C-style launcher for the naive kernel.
extern "C" void softmax_naive_cuda(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    dim3 grid(rows);
    dim3 block(256);
    softmax_naive_kernel<<<grid, block>>>(input, output, rows, cols);
}