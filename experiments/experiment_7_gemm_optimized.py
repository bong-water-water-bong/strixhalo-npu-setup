#!/usr/bin/env python3
"""Experiment 7: Optimized BF16 GEMM via IRON ``kernels.mm()`` + Peano.

Replaces the scalar-Python matmul with IRON's built-in ``kernels.mm()`` which
lowers to ``aie::mmul<8,8,8,bf16,bf16,accfloat>`` vector intrinsics via Peano.

Key improvements over experiments 1-6:
- ``kernels.mm(..., emulate_bf16_mmul_with_bfp16=True)`` → 8×8×8 BFP16 (512 MACs/insn)
- ObjectFifo double buffering (depth=2) → DMA/compute overlap
- K-tiling with accumulator zero-init → arbitrary K depth
- Peano compilation only — zero Chess involvement

Usage:
    source /home/bcloud/mlir-aie/.venv/bin/activate
    export PYTHONPATH=/home/bcloud/mlir-aie/build/python:$PYTHONPATH
    python3 experiments/experiment_7_gemm_optimized.py --dev npu2 -w 2 -i 50
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

try:
    import aie.iron as iron
    from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, kernels
    from aie.helpers.taplib import TensorTiler2D
    from aie.utils.hostruntime.argparse import add_compile_args, add_benchmark_args
    from aie.utils.hostruntime.cli import run_design_cli
    from aie.utils.verify import assert_close_with_benchmark
    from aie.utils.benchmark import run_iters
except ImportError as e:
    print(f"IRON not available: {e}")
    print("  source /home/bcloud/mlir-aie/.venv/bin/activate")
    print("  export PYTHONPATH=/home/bcloud/mlir-aie/build/python:$PYTHONPATH")
    sys.exit(1)


# =============================================================================
# Configuration — tune these for your benchmark
# =============================================================================
M_GLOBAL = 64    # rows
K_GLOBAL = 64    # inner dim (single K-step for max-L1 test)
N_GLOBAL = 64    # cols — larger N for more MACs per K-step
M_TILE   = 64    # tile rows (divisible by r=8)
K_TILE   = 64    # K-step (divisible by s=8)
N_TILE   = 64    # tile cols (divisible by t=8)

# Single-core for now
N_AIE_COLS = 1
N_AIE_ROWS = 1

FIFO_DEPTH = 2

_L1_BYTES = M_TILE * K_TILE * 2 + K_TILE * N_TILE * 2 + M_TILE * N_TILE * 4
assert _L1_BYTES <= 32768, f"L1 overflow: {_L1_BYTES} > 32768"
print(f"[config] M={M_GLOBAL} K={K_GLOBAL} N={N_GLOBAL}  tile={M_TILE}x{K_TILE}x{N_TILE}  L1={_L1_BYTES}B  K-steps={K_GLOBAL//K_TILE}")


# =============================================================================
# Single-core BF16 GEMM with K-tiling (IRON design)
# =============================================================================

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemm_peano_optimized(A: In, B: In, C: Out):
    matmul_kernel = kernels.mm(
        dim_m=M_TILE, dim_k=K_TILE, dim_n=N_TILE,
        input_dtype=bfloat16, output_dtype=np.float32,
        use_chess=False, emulate_bf16_mmul_with_bfp16=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert M_TILE % r == 0 and K_TILE % s == 0 and N_TILE % t == 0

    # Buffer types (generic alias form required by IRON internals)
    A_ty   = np.ndarray[(M_GLOBAL * K_GLOBAL,), np.dtype[bfloat16]]
    B_ty   = np.ndarray[(K_GLOBAL * N_GLOBAL,), np.dtype[bfloat16]]
    C_ty   = np.ndarray[(M_GLOBAL * N_GLOBAL,), np.dtype[np.float32]]
    A_l2_ty = np.ndarray[(M_TILE * K_TILE,), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(K_TILE * N_TILE,), np.dtype[bfloat16]]
    C_l2_ty = np.ndarray[(M_TILE * N_TILE,), np.dtype[np.float32]]
    A_l1_ty = np.ndarray[(M_TILE, K_TILE), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(K_TILE, N_TILE), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(M_TILE, N_TILE), np.dtype[np.float32]]

    # Streaming dims for bf16→fp32 with BFP16 mmul (r,s,t) = (8,8,8)
    a_dims = [(M_TILE // r, r * K_TILE), (K_TILE // s, s), (r, K_TILE), (s, 1)]
    b_dims = [(K_TILE // s, s * N_TILE), (N_TILE // t, t), (s, N_TILE), (t, 1)]
    c_dims = [(M_TILE // r, r * N_TILE), (r, t), (N_TILE // t, r * t), (t, 1)]

    # A: L3 → L2 → L1
    inA = ObjectFifo(A_l2_ty, name="inA", depth=FIFO_DEPTH)
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # B: L3 → L2 → L1
    inB = ObjectFifo(B_l2_ty, name="inB", depth=FIFO_DEPTH)
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # C: L1 → L2 → L3
    memC = ObjectFifo(C_l2_ty, name="memC", depth=FIFO_DEPTH)
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    K_steps = K_GLOBAL // K_TILE

    def core_fn(of_a, of_b, of_c, zero, matmul):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range(K_steps):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matmul(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xD00,
    )

    # DMA tiling: single tile means trivial tiling
    A_tiles = TensorTiler2D.group_tiler(
        (M_GLOBAL, K_GLOBAL), (M_TILE, K_TILE), (1, K_steps), prune_step=False,
    )
    B_tiles = TensorTiler2D.group_tiler(
        (K_GLOBAL, N_GLOBAL), (K_TILE, N_TILE), (K_steps, 1), prune_step=False,
    )
    C_tile = TensorTiler2D.group_tiler(
        (M_GLOBAL, N_GLOBAL), (M_TILE, N_TILE), (1, 1), prune_step=False,
    )[0]

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(inA.prod(), A, tap=A_tiles[0])
        rt.fill(inB.prod(), B, tap=B_tiles[0])
        rt.drain(outC.cons(), C, tap=C_tile, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


# =============================================================================
# CLI + verification
# =============================================================================

def _make_argparser():
    p = argparse.ArgumentParser(
        prog="IRON Optimized GEMM (Peano, bf16→fp32, BFP16 mmul)",
        description=f"Single-core BF16 GEMM: {M_GLOBAL}x{K_GLOBAL}x{N_GLOBAL}, "
                    f"tile={M_TILE}x{K_TILE}x{N_TILE}, L1={_L1_BYTES}/32768 B, "
                    f"K-steps={K_GLOBAL//K_TILE}",
    )
    add_compile_args(p, short_dev=None)
    add_benchmark_args(p)
    return p


def _numpy_reference(A_np, B_np):
    return (A_np.astype(np.float32) @ B_np.astype(np.float32)).astype(np.float32)


def _run_and_verify(opts):
    rng = np.random.default_rng(1726250518)
    A_np = (rng.random((M_GLOBAL, K_GLOBAL)) * 2.0 - 1.0).astype(bfloat16)
    B_np = (rng.random((K_GLOBAL, N_GLOBAL)) * 2.0 - 1.0).astype(bfloat16)
    A_t = iron.tensor(A_np.reshape(-1), dtype=bfloat16, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=bfloat16, device="npu")
    C_t = iron.zeros(M_GLOBAL * N_GLOBAL, dtype=np.float32, device="npu")

    bench = run_iters(
        gemm_peano_optimized, A_t, B_t, C_t,
        warmup=opts.warmup, iters=opts.iters,
    )
    expected = _numpy_reference(A_np, B_np)
    actual = C_t.numpy().reshape(M_GLOBAL, N_GLOBAL)
    assert_close_with_benchmark(
        actual, expected, bench=bench,
        ops=2.0 * M_GLOBAL * K_GLOBAL * N_GLOBAL,
        fail_msg="output does not match A @ B",
    )

    total_ops = 2.0 * M_GLOBAL * K_GLOBAL * N_GLOBAL
    # bench has median_s or median_ms — check which
    if hasattr(bench, 'median_s'):
        median_s = bench.median_s
    elif hasattr(bench, 'median_ms'):
        median_s = bench.median_ms / 1000.0
    else:
        median_s = bench.median_us / 1e6
    gflops = total_ops / (median_s * 1e9)
    n_cores = N_AIE_ROWS * max(N_AIE_COLS, 1)
    pts = "µs"
    print()
    print("=" * 70)
    print(f"GEMM: M={M_GLOBAL}, K={K_GLOBAL}, N={N_GLOBAL}  |  {n_cores} core(s)  |  bf16→fp32")
    print(f"Tile: m={M_TILE}, k={K_TILE}, n={N_TILE}  |  L1: {_L1_BYTES}/32768 B")
    print(f"mmul: BFP16 8{chr(0xd7)}8{chr(0xd7)}8  |  K-steps: {K_GLOBAL//K_TILE}  |  Peano (no Chess)")
    print(f"Median: {median_s*1e6:.1f} µs  |  {gflops:.1f} GFLOPS  |  per-core: {gflops/n_cores:.1f} GFLOPS")
    print("=" * 70)
    print()


def _compile_kwargs(opts):
    return {}


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gemm_peano_optimized, opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
