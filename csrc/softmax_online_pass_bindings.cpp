#include <torch/extension.h>

// CUDA launcher implemented in kernels/softmax_online_pass.cu
extern "C" void softmax_online_cuda(
    const float* input,
    float* output,
    int rows,
    int cols
);

// Simple PyTorch binding: input [rows, cols] -> output [rows, cols]
torch::Tensor softmax_online_forward(torch::Tensor input) {
    // Basic checks for a minimal, safe binding
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [rows, cols]");

    auto input_contig = input.contiguous();
    auto output = torch::zeros_like(input_contig);

    int rows = static_cast<int>(input_contig.size(0));
    int cols = static_cast<int>(input_contig.size(1));

    softmax_online_cuda(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}

// PyBind module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_online_forward, "Online-pass softmax (CUDA)");
}
