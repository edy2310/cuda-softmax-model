import os
import sys

import matplotlib.pyplot as plt
import torch

# Ensure the project root (where the .so is built) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import softmax_naive_ext
import softmax_shared_ext
import softmax_online_ext
import softmax_warp_ext


# ---- Benchmark configuration ----
WARMUP_ITERS = 10
BENCH_ITERS = 200

# (rows, cols) shapes to sweep (same as throughput benchmark)
SHAPES = [
    (256, 6),
    (1024, 6),
    (4096, 6),
    (16384, 6),
    (4096, 128),
    (4096, 512),
    (4096, 1024),
]


def time_kernel(fn, x):
    # Warm-up to stabilize GPU clocks and caches
    for _ in range(WARMUP_ITERS):
        _ = fn(x)
    torch.cuda.synchronize()

    # Measure average latency using CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(BENCH_ITERS):
        _ = fn(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms / BENCH_ITERS


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    kernels = [
        ("naive_cuda", softmax_naive_ext.forward),
        ("shared_memory", softmax_shared_ext.forward),
        ("online_pass", softmax_online_ext.forward),
        ("warp_primitives", softmax_warp_ext.forward),
        ("pytorch_softmax", lambda t: torch.softmax(t, dim=-1)),
    ]

    # Collect results: {kernel_name: [(shape, ms), ...]}
    results = {name: [] for name, _ in kernels}

    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        for name, fn in kernels:
            ms = time_kernel(fn, x)
            results[name].append(((rows, cols), ms))

    # Print Markdown table
    print("\n| Kernel | Shape (rows, cols) | Latency (ms) |")
    print("|---|---:|---:|")
    for name in results:
        for (rows, cols), ms in results[name]:
            print(f"| {name} | ({rows}, {cols}) | {ms:.4f} |")

    # Plot latency lines per kernel
    plt.figure(figsize=(9, 6))
    for name in results:
        x_labels = [f"{r}x{c}" for (r, c), _ in results[name]]
        y_vals = [t for _, t in results[name]]
        plt.plot(range(len(x_labels)), y_vals, marker="o", label=name)

    plt.xticks(range(len(SHAPES)), [f"{r}x{c}" for r, c in SHAPES], rotation=45)
    plt.xlabel("Input shape (rows x cols)")
    plt.ylabel("Latency (ms)")
    plt.title("Softmax Latency Benchmark")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("softmax_latency.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
