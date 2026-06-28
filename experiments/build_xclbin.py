#!/usr/bin/env python3
"""Build xclbin from experiment_10 packed kernel and save to disk.

Once we have the xclbin, we can use it with XRT + GTT dma-buf buffers
for zero-copy, pre-packed GEMM submission — bypassing IRON ObjectFifo DMA.

Usage:
    source /home/bcloud/mlir-aie/.venv/bin/activate
    export PYTHONPATH=/home/bcloud/mlir-aie/install/python:$PYTHONPATH
    export PATH=/home/bcloud/mlir-aie/build/bin:$PATH
    python3 experiments/build_xclbin.py
"""

import os, sys, tempfile

# Add experiment dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_10_phase5 import _build_phase5_design

# Import IRON
import aie.iron as iron
from aie.iron.device import NPU2, from_name
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.utils import compile_mlir_module, compile_external_kernel
from aie.utils.compile import resolve_target_arch
from aie.iron import ExternalFunction

OUTPUT_XCLBIN = os.path.join(os.path.dirname(__file__), "phase5_packed.xclbin")

def main():
    # Use same config as our best experiment
    M, K, N = 4096, 8192, 1024
    m, k, n = 32, 128, 32
    n_aie_cols = 8
    dev = from_name("npu2", n_cols=n_aie_cols)
    target_arch = resolve_target_arch(dev)

    print(f"Building xclbin for {m}×{k}×{n} packed GEMM...")
    print(f"  Problem: M={M} K={K} N={N}")
    print(f"  Columns: {n_aie_cols}")
    print(f"  Device:  {dev}")

    # Build the design
    program = _build_phase5_design(
        dev, M, K, N, m, k, n, n_aie_cols,
        dtype_in_str="bf16",
        dtype_out_str="f32",
        b_col_maj=1,
        c_col_maj=0,
        emulate_bf16_mmul_with_bfp16=True,
        use_chess=False,
        scalar=False,
        pre_pack=False,  # Use B_COL_MAJ path (transpose in kernel)
        no_ii1=True,     # II=1 pragma is no-op on Peano
        unroll_2x=False,
        swp=False,
        acc8=False,
    )

    # Compile external kernels
    external_kernels = list(ExternalFunction._instances)
    ExternalFunction._instances.clear()

    kernel_dir = tempfile.mkdtemp(prefix="xclbin_build_")
    print(f"  Kernel dir: {kernel_dir}")

    for func in external_kernels:
        if not func._compiled:
            compile_external_kernel(func, kernel_dir, target_arch)

    # Compile MLIR to xclbin
    compile_mlir_module(
        mlir_module=program.mlir_module,
        xclbin_path=OUTPUT_XCLBIN,
        work_dir=kernel_dir,
        use_chess=False,
    )

    # Verify
    if os.path.exists(OUTPUT_XCLBIN):
        size_mb = os.path.getsize(OUTPUT_XCLBIN) / 1e6
        print(f"\n✅ xclbin saved: {OUTPUT_XCLBIN} ({size_mb:.1f} MB)")
    else:
        print(f"\n❌ xclbin NOT created at {OUTPUT_XCLBIN}")
        sys.exit(1)


if __name__ == "__main__":
    main()
