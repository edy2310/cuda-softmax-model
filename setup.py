from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Build all CUDA kernels as separate PyTorch extensions
ext_modules = [
    CUDAExtension(
        name="softmax_naive_ext",
        sources=[
            "csrc/softmax_naive_bindings.cpp",
            "kernels/softmax_naive.cu",
        ],
    ),
    CUDAExtension(
        name="softmax_shared_ext",
        sources=[
            "csrc/softmax_shared_memory_bindings.cpp",
            "kernels/softmax_shared_memory.cu",
        ],
    ),
    CUDAExtension(
        name="softmax_online_ext",
        sources=[
            "csrc/softmax_online_pass_bindings.cpp",
            "kernels/softmax_online_pass.cu",
        ],
    ),
    CUDAExtension(
        name="softmax_warp_ext",
        sources=[
            "csrc/softmax_warp_primitives_bindings.cpp",
            "kernels/softmax_warp_primitives.cu",
        ],
    ),
]


setup(
    name="softmax_kernels",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
