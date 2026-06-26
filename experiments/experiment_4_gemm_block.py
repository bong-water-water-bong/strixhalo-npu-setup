#!/usr/bin/env python3
"""Experiment 4: Blocked/ATB GEMM layout sweep.

Compares row-major, blocked-8x8, and ATB layouts for a 64x64x64 GEMM block.
Measures throughput impact of memory layout choices.
"""

import sys
from experiments.kernel_defs.gemm_block import GemmBlockKernel
from experiments.runner import NpuRunner, ExperimentConfig
from experiments_lib.layouts import RowMajor, BlockedLayout, ATBLayout
from experiments_lib.datatypes import BFP16
from npu_control_plane.metadata import MetadataStore


LAYOUTS = [RowMajor(), BlockedLayout(8), ATBLayout(2, 1)]


def main():
    store = MetadataStore()

    for layout in LAYOUTS:
        kernel = GemmBlockKernel(
            name=f"gblock-{layout.name}",
            block_m=64, block_k=64, block_n=64,
            layout=layout,
        )
        fits = kernel.fits_in_l1()
        print(f"\nLayout: {layout.name:20s}  L1 fit: {'YES' if fits else 'NO '}  {kernel}")

    print("\n=== Experiment 4: Layout Sweep ===")
    print("\nTo run on hardware: use `npu-ctrl bench run` with each compiled kernel.")


if __name__ == "__main__":
    main()