#!/usr/bin/env python3
"""Experiment 1: Single-tile passthrough kernel baseline.

Measures raw data movement bandwidth for one AIE2P tile using
the simplest possible kernel. Establishes floor for all later experiments.

Usage:
    NPU_CTRL_STORE=/tmp/npu-ctrl-exp1 python3 experiments/experiment_1_passthrough.py
"""

from experiments.kernel_defs.passthrough import PassthroughKernel
from experiments.runner import NpuRunner, ExperimentConfig
from npu_control_plane.metadata import MetadataStore

SIZES = [64, 256, 1024, 4096, 16384]


def main():
    store = MetadataStore()
    cfg = ExperimentConfig(
        label="exp1-passthrough",
        kernel_name="passthrough",
        shape_str="N=1024",
        warmup=5,
        iters=30,
        toolchain="iron+peano",
        store=store,
    )
    kernel = PassthroughKernel(name="pt")
    runner = NpuRunner(cfg, dry_run=True)

    print("=== Experiment 1: Passthrough Baseline ===")
    print(f"Dry run mode — no NPU execution. Sizes: {SIZES}")
    print(f"Kernel: {kernel}")
    print(f"Store: {store.root}")

    for n in SIZES:
        source = kernel.generate_iron_source(n)
        print(f"\nN={n}: source generated ({len(source)} bytes)")
        for line in source.splitlines()[:6]:
            print(f"  {line}")
        print("  ...")

    print("\nTo run on hardware: set dry_run=False and provide NPU access.")


if __name__ == "__main__":
    main()
