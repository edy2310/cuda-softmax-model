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
BENCH_ITERS = 120

# (rows, cols) shapes to sweep
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


def estimate_flops(rows, cols):
    # Rough softmax cost per row:
    # 1) max reduction: (cols - 1) comparisons
    # 2) exp + sum: cols exp + (cols - 1) adds
    # 3) normalize: cols divs
    # 4) subtract max before exp: cols subs
    # Treat comparisons as 1 FLOP for roofline simplicity.
    flops_per_row = (cols - 1) + cols + (cols - 1) + cols + cols
    return rows * flops_per_row


def estimate_bytes(rows, cols):
    # Simple traffic model that varies with columns:
    # - Two full reads of input (max pass + exp/sum pass)
    # - One full write of output
    # - Two scalars per row (max + sum) as reduction overhead
    bytes_per_row = (3 * cols + 2) * 4
    return rows * bytes_per_row


def fp32_cores_per_sm(major, minor):
    # Minimal mapping for common architectures (enough for T4 in Colab)
    if (major, minor) == (7, 5):  # T4
        return 64
    if (major, minor) == (7, 0):  # V100
        return 64
    if (major, minor) == (8, 0):  # A100
        return 64
    if (major, minor) == (8, 6):  # RTX 30xx / A10
        return 128
    if (major, minor) == (9, 0):  # H100
        return 128
    # Reasonable fallback for unknown GPUs
    return 64


def device_roofline_limits():
    # Compute theoretical roofline limits from device properties
    props = torch.cuda.get_device_properties(0)

    # Peak FP32 throughput (GFLOP/s)
    sm_count = props.multi_processor_count
    core_count = fp32_cores_per_sm(props.major, props.minor)
    clock_hz = props.clock_rate * 1e3  # kHz -> Hz
    peak_gflops = (sm_count * core_count * 2 * clock_hz) / 1e9

    # Memory bandwidth (GB/s)
    mem_clock = getattr(props, "memory_clock_rate", None)
    bus_width = getattr(props, "memory_bus_width", None)
    if mem_clock is None or bus_width is None:
        if "T4" in props.name:
            bandwidth_gb_s = 320.0
            print("Using default T4 memory bandwidth: 320 GB/s.")
        else:
            raise RuntimeError("Cannot infer memory bandwidth from device properties.")
    else:
        mem_clock_hz = mem_clock * 1e3  # kHz -> Hz
        bandwidth_gb_s = (mem_clock_hz * (bus_width / 8) * 2) / 1e9

    return peak_gflops, bandwidth_gb_s, props.name


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

    peak_gflops, bandwidth_gb_s, device_name = device_roofline_limits()

    # Collect results: {kernel_name: [(shape, ai, gflops), ...]}
    results = {name: [] for name, _ in kernels}

    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        flops = estimate_flops(rows, cols)
        bytes_moved = estimate_bytes(rows, cols)
        ai = flops / bytes_moved  # FLOPs per byte

        for name, fn in kernels:
            seconds = time_kernel(fn, x)
            if seconds <= 0:
                raise RuntimeError(f"Non-positive timing for {name} at {rows}x{cols}.")
            gflops = (flops / seconds) / 1e9
            results[name].append(((rows, cols), ai, gflops))

    # Print Markdown table
    print("\n| Kernel | Shape (rows, cols) | Arithmetic Intensity (FLOP/byte) | Performance (GFLOP/s) |")
    print("|---|---:|---:|---:|")
    for name in results:
        for (rows, cols), ai, gflops in results[name]:
            print(f"| {name} | ({rows}, {cols}) | {ai:.4f} | {gflops:.2f} |")

    # Build roofline curve
    all_ai = [ai for vals in results.values() for _, ai, _ in vals if ai > 0]
    min_ai = min(all_ai)
    max_ai = max(all_ai)
    if min_ai == max_ai:
        ai_left = max(min_ai * 0.7, 1e-4)
        ai_right = max(min_ai * 1.3, ai_left * 1.1)
    else:
        ai_left = max(min_ai * 0.7, 1e-4)
        ai_right = max_ai * 1.3

    ai_values = [
        ai_left * (ai_right / ai_left) ** (i / 99)
        for i in range(100)
    ]
    bandwidth_line = [bandwidth_gb_s * ai for ai in ai_values]
    # Plot ceilings and kernel points
    plt.figure(figsize=(9, 6))
    plt.plot(
        ai_values,
        bandwidth_line,
        color="magenta",
        linestyle="--",
        linewidth=2.5,
        label="Bandwidth limit",
    )
    plt.axhline(
        peak_gflops,
        color="gold",
        linestyle=":",
        linewidth=2.5,
        label="Peak FP32",
    )

    for name in results:
        x_vals = [ai for _, ai, _ in results[name]]
        y_vals = [g for _, _, g in results[name]]
        plt.plot(x_vals, y_vals, marker="o", linewidth=1, label=name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=ai_left, right=ai_right)
    plt.ylim(bottom=1)
    plt.xlabel("Arithmetic Intensity (FLOP/byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(f"Softmax Roofline (GPU: {device_name})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("softmax_roofline.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
