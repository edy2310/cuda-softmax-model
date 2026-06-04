import math
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


# ---- Simple configuration ----
ROWS = 4096           # batch size
COLS = 6              # number of classes (HAR has 6 activities)
WARMUP_ITERS = 10
BENCH_ITERS = 100

# NVIDIA T4 (Google Colab) reference values
PEAK_BW_BYTES = 320e9     # ~320 GB/s
PEAK_FLOPS = 8.1e12       # ~8.1 TFLOPS (FP32)


def time_kernel(fn, x):
    # Warm-up to stabilize GPU clocks and caches
    for _ in range(WARMUP_ITERS):
        _ = fn(x)
    torch.cuda.synchronize()

    # Measure average time using CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(BENCH_ITERS):
        _ = fn(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return (ms / 1000.0) / BENCH_ITERS


def estimate_bytes_moved(rows, cols):
    # Simple approximation: 2 reads + 1 write of float32 data
    elements = rows * cols
    return elements * 4 * 3


def estimate_flops(rows, cols):
    # Simple approximation: exp + add + exp + div per element ~ 4 FLOPs
    elements = rows * cols
    return elements * 4


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    # Input tensor for benchmarking
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.float32)

    # Define all kernels to test (including PyTorch softmax)
    kernels = [
        ("naive_cuda", softmax_naive_ext.forward),
        ("shared_memory", softmax_shared_ext.forward),
        ("online_pass", softmax_online_ext.forward),
        ("warp_primitives", softmax_warp_ext.forward),
        ("pytorch_softmax", lambda t: torch.softmax(t, dim=-1)),
    ]

    # Collect results
    results = []
    bytes_moved = estimate_bytes_moved(ROWS, COLS)
    flops = estimate_flops(ROWS, COLS)
    oi = flops / bytes_moved

    for name, fn in kernels:
        seconds = time_kernel(fn, x)
        bandwidth = bytes_moved / seconds
        perf = flops / seconds
        pct_peak = (bandwidth / PEAK_BW_BYTES) * 100.0
        results.append((name, bandwidth, pct_peak, perf))

    # Print a Markdown table with bandwidth and % peak
    print("\n| Kernel | Bandwidth (GB/s) | % Peak BW |")
    print("|---|---:|---:|")
    for name, bandwidth, pct_peak, _ in results:
        print(f"| {name} | {bandwidth/1e9:.2f} | {pct_peak:.2f}% |")

    # ---- Roofline plot ----
    # Roofline: performance = min(peak_flops, peak_bw * OI)
    oi_vals = [1e-3, 1e2]
    roofline = [min(PEAK_FLOPS, PEAK_BW_BYTES * x) for x in oi_vals]

    plt.figure(figsize=(8, 6))
    plt.loglog(oi_vals, roofline, label="Roofline (T4)")

    # Plot each kernel as a point at the same OI but different performance
    for name, _, _, perf in results:
        plt.loglog([oi], [perf], marker="o", label=name)

    plt.xlabel("Operational Intensity (FLOPs/byte)")
    plt.ylabel("Performance (FLOPs/s)")
    plt.title("Softmax Roofline (UCI HAR output size)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("softmax_roofline.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
