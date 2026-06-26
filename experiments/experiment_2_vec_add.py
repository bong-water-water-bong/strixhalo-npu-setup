#!/usr/bin/env python3
"""Experiment 2: Single-tile vector add benchmark.

Sweeps vector sizes, records throughput in GB/s and MOPS.
"""

from .kernel_defs.vec_add import VecAddKernel
from .runner import NpuRunner, ExperimentConfig
from npu_control_plane.metadata import MetadataStore

SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]


def main():
    store = MetadataStore()
    cfg = ExperimentConfig(
        label="exp2-vecadd",
        kernel_name="vec_add",
        shape_str="N=1024",
        warmup=5,
        iters=30,
        toolchain="iron+peano",
        store=store,
    )
    kernel = VecAddKernel(name="vadd")
    runner = NpuRunner(cfg, dry_run=True)

    print("=== Experiment 2: Vector Add Sweep ===")
    print(f"Dry run — no NPU execution. Sizes: {SIZES}")
    print(f"Kernel: {kernel}")
    print(f"Store: {store.root}")

    for n in SIZES:
        source = kernel.generate_iron_source(n)
        print(f"\nN={n}: source ({len(source)} bytes)")

    print("\nTo run on hardware: set dry_run=False.")


if __name__ == "__main__":
    main()