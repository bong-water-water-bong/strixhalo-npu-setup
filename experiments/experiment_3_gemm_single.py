#!/usr/bin/env python3
"""Experiment 3: Single-tile GEMM microkernel.

Benchmarks 8x8x8 BFP16 matmul on one AIE2P tile.
Measures:
- Raw microkernel throughput (GOPS)
- DMA overhead
- Warmup convergence
"""

import sys

from .kernel_defs.gemm_8x8x8 import Gemm8x8x8Kernel
from .runner import NpuRunner, ExperimentConfig
from experiments_lib.tile_shapes import TileShape, TileShapeConfig
from experiments_lib.datatypes import BFP16
from npu_control_plane.metadata import MetadataStore


def main():
    store = MetadataStore()
    cfg = ExperimentConfig(
        label="exp3-gemm-single",
        kernel_name="gemm_8x8x8",
        shape_str="M=8,K=8,N=8",
        warmup=5,
        iters=100,
        toolchain="iron+peano",
        store=store,
    )
    kernel = Gemm8x8x8Kernel(name="gemm8")
    runner = NpuRunner(cfg, dry_run=True)

    tile_shapes = TileShapeConfig(m=64, k=64, n=64)
    print("=== Experiment 3: Single-Tile GEMM ===")
    print(f"Microkernel: {kernel}")
    print(f"Dry run — no NPU execution.")
    print(f"Tile shapes that fit L1 ({32*1024} bytes):")
    for ts in tile_shapes.plausible_shapes():
        fits = ts.fits_in_l1(dtype=BFP16)
        print(f"  {ts.m}x{ts.k}x{ts.n}: {ts.byte_count(BFP16)} bytes {'✅' if fits else '❌'}")

    source = kernel.generate_iron_source()
    print(f"\nKernel source generated ({len(source)} bytes)")

    print("\nTo run on hardware:")
    print("  NPU_CTRL_STORE=$STORE python3 experiments/experiment_3_gemm_single.py --no-dry-run")


if __name__ == "__main__":
    main()