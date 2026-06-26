#!/usr/bin/env python3
"""Experiment 5: Multi-tile GEMM scaling.

Scales from 1 tile to 32 tiles to measure:
- Strong scaling (fixed problem size)
- Weak scaling (fixed per-tile size)
- Communication/routing overhead
"""

import sys
from experiments.kernel_defs.gemm_block import GemmBlockKernel
from experiments.runner import NpuRunner, ExperimentConfig
from experiments_lib.tile_shapes import MultiTileConfig
from npu_control_plane.metadata import MetadataStore


CLUSTERS = [
    MultiTileConfig(1, 1),
    MultiTileConfig(2, 2),
    MultiTileConfig(4, 4),
    MultiTileConfig(8, 4),
]


def main():
    store = MetadataStore()
    print("=== Experiment 5: Multi-Tile Scaling ===")
    for cluster in CLUSTERS:
        kernel = GemmBlockKernel(
            name=f"gemm-multi-{cluster.rows}x{cluster.cols}",
            multi_tile=cluster,
        )
        print(f"  {cluster.rows}x{cluster.cols} = {cluster.tile_count:2d} tiles  {kernel}")

    print("\nTo run on hardware:")
    print("  NPU_CTRL_STORE=$STORE python3 experiments/experiment_5_gemm_multi.py --no-dry-run")


if __name__ == "__main__":
    main()