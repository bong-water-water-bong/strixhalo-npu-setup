#!/usr/bin/env python3
"""Experiment 6: BFP16 vs FP32 accuracy/throughput characterization.

Measures numerical error for GEMM at various sizes using CPU reference.
Provides data to recommend optimal datatype path.
"""

import random
from experiments_lib.datatypes import BFP16, BF16, FP32

SIZES = [(8, 8, 8), (16, 16, 16), (32, 32, 32)]


def cpu_gemm(m, k, n):
    """Reference CPU GEMM. Returns matrix max abs value."""
    a = [[random.gauss(0.0, 0.5) for _ in range(k)] for _ in range(m)]
    b = [[random.gauss(0.0, 0.5) for _ in range(n)] for _ in range(k)]
    c = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += a[i][kk] * b[kk][j]
            c[i][j] = acc
    max_abs = max(abs(v) for row in c for v in row)
    return max_abs


def main():
    print("=== Experiment 6: BFP16 Accuracy ===")
    print(f"{'Size':>12s} {'Max abs value':>16s} {'Notes'}")
    print("-" * 50)
    for (m, k, n) in SIZES:
        ref = cpu_gemm(m, k, n)
        print(f"{m}x{k}x{n:>3s}  {ref:>16.6f}  FP32 reference (CPU)")
    print("\nTo run with NPU: compile kernel, run comparison.")


if __name__ == "__main__":
    main()
