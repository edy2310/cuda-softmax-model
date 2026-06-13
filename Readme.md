# Softmax Kernels: CUDA softmax for NVIDIA GPUs

Softmax Kernels is a CUDA project focused on building, validating, and deploying optimized softmax implementations for NVIDIA GPUs. It demonstrates how a core GPU primitive can be developed as a standalone kernel, exposed through a clean Python API, and exercised inside a realistic inference workflow.

## Demonstrated Capabilities

- **Kernel Design:** four softmax variants, from a clear baseline to shared-memory, online-pass, and warp-primitive implementations.
- **Systems Integration:** PyTorch C++ extensions that expose CUDA kernels through a stable Python API.
- **Performance Analysis:** latency, throughput, and roofline benchmarking with interpretation instead of raw numbers only.
- **Production Deployment:** a Triton Python backend that runs the same custom kernel inside an inference server.
- **Correctness Engineering:** equivalence checks against `torch.softmax` plus input validation in the bindings.
- **Inference Workflow Design:** a real model harness (`HARMLP`) that proves the kernel works inside a downstream application.

## Architecture & Design Decisions

### Execution Flow

The flow is intentionally simple and inspectable:

```text
Python tests / benchmarks / models
               ↓
       PyTorch C++ bindings
               ↓
           CUDA kernel
```

In practice, the Python layer is responsible for loading the extension and driving validation, performance measurements, or model execution. The C++ binding is the contract boundary: it validates the input shape and dtype, ensures the tensor is CUDA-backed, prepares the output tensor, and dispatches the request to the chosen kernel implementation. The CUDA kernel then performs the row-wise softmax reduction on the GPU, returning results back through the binding so they can be compared against `torch.softmax`, benchmarked, or consumed by `HARMLP` during inference.

### Memory Layout

The kernels operate on row-wise softmax over 2D `[rows, cols]` tensors. Each row is treated as one reduction domain, which matches the last dimension softmax used in classification and inference workloads. The binding materializes a contiguous input tensor before launching the kernel, and the output keeps the same shape as the input.

### Engineering Trade-offs

- The naive kernel exists as a baseline for correctness and debugging.
- The shared-memory version reduces repeated global memory traffic.
- The online-pass version favors numerical stability and avoids extra passes where possible.
- The warp-primitive version leans on warp-level collectives to reduce synchronization overhead.

These variants show deliberate engineering choices, not just alternate code paths. Softmax is memory-bound, so the main optimization lever is reducing reads, writes, and synchronization rather than chasing compute intensity.

## Performance Results & Interpretation

### Throughput

**Plot**  
![Softmax throughput](benchmarks/results/softmax_throughput.png)

**Interpretation**  
Small shapes are launch-overhead bound, so throughput is low when the row width is tiny. As `cols` grows, the kernels get more useful work per launch and throughput improves before flattening out, which is the expected shape for a memory-bound kernel. The custom kernels improve in the order you would expect from the design: naive < warp primitives < online pass < shared memory. PyTorch still leads because it is heavily tuned and benefits from production-grade scheduling and fusion.

### Latency

**Plot**  
![Softmax latency](benchmarks/results/softmax_latency.png)

**Interpretation**  
Latency follows the same pattern as throughput. Small inputs cluster together because fixed launch cost dominates. Larger inputs separate the kernels more clearly, and the shared-memory and online-pass variants reduce the per-element cost relative to the naive baseline. The result is what a reviewer should expect from a kernel that is improving memory behavior rather than changing the algorithm itself.

### Roofline

**Plot**  
![Softmax roofline](benchmarks/results/softmax_roofline.png)

**Interpretation**  
The roofline stays in the memory-bound region because softmax has low arithmetic intensity. On the current benchmark set, AI stays roughly in the 0.35-0.42 FLOP/byte range, so the points remain near the bandwidth ceiling and far from the peak FP32 line. That is not a weakness in the analysis; it is the right conclusion for this workload. The important signal is that higher GFLOP/s at similar AI means better effective bandwidth use.

## Correctness & Validation

Validation is built around two checks:

1. Compare the custom-kernel model against an equivalent PyTorch model with identical weights.
2. Assert that invalid inputs fail early in the C++ bindings (`CUDA`, `float32`, and 2D shape checks).

The equivalence test runs multiple batch sizes and checks numerical closeness with `torch.allclose`, which gives confidence that the kernel is functionally correct before performance is considered.

```bash
python tests/softmax_equivalence_test.py
```

## Quick Start

### Requirements

- NVIDIA CUDA-capable GPU
- CUDA toolkit and a PyTorch build with CUDA support
- Python dependencies available in the environment

### Build

```bash
python setup.py build_ext --inplace
```

### Example Usage

```python
import torch
import softmax_warp_ext

x = torch.randn(1024, 512, device="cuda", dtype=torch.float32)
y = softmax_warp_ext.forward(x)
print(y.shape)
```

## Optimization Roadmap

- Add FP16 and BF16 paths with correctness checks and explicit accumulation strategy.
- Fuse softmax with adjacent ops to reduce memory traffic in inference pipelines.
- Expand warp-specialized reductions and vectorized loads for better bandwidth efficiency.
- Explore persistent kernels and better block sizing for large, irregular batches.
- Profile with Nsight Compute to confirm bottlenecks and guide tuning decisions.
- Keep the long-term goal in view: reduce data movement first, then improve occupancy and synchronization costs.
