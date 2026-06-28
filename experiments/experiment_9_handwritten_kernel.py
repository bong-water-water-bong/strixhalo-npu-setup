#!/usr/bin/env python3
"""Experiment 9: Hand-written BFP16 GEMM kernel via ExternalFunction.

Uses the whole_array.py proven multi-tile design pattern but replaces
``kernels.mm()`` with our custom ``ExternalFunction`` pointing to
``kernel_gemm_bfp16_optimized.cc``.

Compares against the built-in ``kernels.mm()`` baseline.

Usage:
    source /home/bcloud/mlir-aie/.venv/bin/activate
    export PYTHONPATH=/home/bcloud/mlir-aie/install/python:$PYTHONPATH
    python3 experiments/experiment_9_handwritten_kernel.py --dev npu2 -w 2 -i 10 \\
        -M 256 -K 8192 -N 512 -m 64 -k 32 -n 64 --n-aie-cols 8
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

try:
    import aie.iron as iron
    from aie.iron import (
        In, ObjectFifo, Out, Program, Runtime, Worker,
        ExternalFunction, CompileTime,
    )
    from aie.iron.controlflow import range_
    from aie.helpers.taplib import TensorTiler2D
    from aie.utils.hostruntime.argparse import add_compile_args, add_benchmark_args
    from aie.utils.hostruntime.cli import run_design_cli
    from aie.utils.verify import assert_close_with_benchmark
    from aie.utils.benchmark import run_iters
except ImportError as e:
    print(f"IRON not available: {e}")
    print("  source /home/bcloud/mlir-aie/.venv/bin/activate")
    print("  export PYTHONPATH=/home/bcloud/mlir-aie/install/python:$PYTHONPATH")
    sys.exit(1)

_KERNEL_CC = os.path.join(os.path.dirname(__file__),
                          "kernel_gemm_bfp16_optimized.cc")


@iron.jit(aiecc_flags=["--alloc-scheme=bank-aware", "--dynamic-objFifos"])
def gemm_handwritten(
    A: In, B: In, C: Out,
    *,
    M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    m: CompileTime[int], k: CompileTime[int], n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str], dtype_out_str: CompileTime[str],
    emulate_bf16_mmul_with_bfp16: CompileTime[bool],
    use_chess: CompileTime[bool], scalar: CompileTime[bool],
):
    """Multi-tile BFP16 GEMM with hand-written 2×2 mmul kernel."""
    n_aie_rows = 4
    r, s, t = (8, 8, 8)
    K_STEPS = K // k

    # -- Hand-written external functions --
    matmul_fn = ExternalFunction(
        name="gemm_bf16_f32_bfp16",
        source_file=_KERNEL_CC,
        arg_types=[
            np.ndarray[(m * k,), np.dtype[bfloat16]],
            np.ndarray[(k * n,), np.dtype[bfloat16]],
            np.ndarray[(m * n,), np.dtype[np.float32]],
        ],
        compile_flags=[
            f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )
    zero_fn = ExternalFunction(
        name="zero_f32",
        source_file=_KERNEL_CC,
        arg_types=[np.ndarray[(m * n,), np.dtype[np.float32]]],
        compile_flags=[
            f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )

    # -- Buffer types --
    A_ty    = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty    = np.ndarray[(K * N,), np.dtype[bfloat16]]
    C_ty    = np.ndarray[(M * N,), np.dtype[np.float32]]
    A_l2_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[bfloat16]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[np.float32]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[np.float32]]

    # -- Streaming dims for BFP16 (r,s,t) = (8,8,8) --
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

    # -- A: L3 → L2 → L1 (fan-out by row) --
    A_l3l2 = [ObjectFifo(A_l2_ty, name=f"A_L3L2_{r}", depth=2)
              for r in range(n_aie_rows)]
    A_l2l1 = [f.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{i}",
              dims_to_stream=a_dims) for i, f in enumerate(A_l3l2)]

    # -- B: L3 → L2 → L1 (fan-out by col) --
    B_l3l2 = [ObjectFifo(B_l2_ty, name=f"B_L3L2_{c}", depth=2)
              for c in range(n_aie_cols)]
    B_l2l1 = [f.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{i}",
              dims_to_stream=b_dims) for i, f in enumerate(B_l3l2)]

    # -- C: L1 → L2 → L3 (join rows into column stream) --
    C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3 = []
    for col in range(n_aie_cols):
        of_ = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=2)
        parts = of_.prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[2] * n_aie_rows,
        )
        for row in range(n_aie_rows):
            C_l1l2[row][col] = parts[row]
        C_l2l3.append(of_)

    # -- Core compute function --
    def core_fn(in_a, in_b, out_c, zero, matmul):
        elem_out = out_c.acquire(1)
        zero(elem_out)
        for _ in range_(K_STEPS):
            elem_a = in_a.acquire(1)
            elem_b = in_b.acquire(1)
            matmul(elem_a, elem_b, elem_out)
            in_a.release(1)
            in_b.release(1)
        out_c.release(1)

    # -- Worker grid --
    workers = Worker.grid(
        n_aie_rows, n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [A_l2l1[row].cons(), B_l2l1[col].cons(),
             C_l1l2[row][col].prod(), zero_fn, matmul_fn],
            stack_size=0xD00,
        ),
    )

    # -- Runtime DMA --
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A_t, B_t, C_t):
        rt.start(*[w for row in workers for w in row])
        tg = rt.task_group()
        for k_step in range(K_STEPS):
            for row in range(n_aie_rows):
                rt.fill(A_l3l2[row].prod(), A_t, task_group=tg)
            for col in range(n_aie_cols):
                rt.fill(B_l3l2[col].prod(), B_t, task_group=tg)
                rt.drain(C_l2l3[col].cons(), C_t, wait=True, task_group=tg)
            rt.finish_task_group(tg)
            tg = rt.task_group()
        rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()


# =============================================================================
# CLI + verification
# =============================================================================

def _make_argparser():
    p = argparse.ArgumentParser(
        prog="Hand-Written BFP16 GEMM",
        description="Multi-tile BFP16 GEMM with custom 2×2 mmul ExternalFunction",
    )
    add_compile_args(p, short_dev=None)
    add_benchmark_args(p)
    p.add_argument("-M", type=int, default=256)
    p.add_argument("-K", type=int, default=8192)
    p.add_argument("-N", type=int, default=256)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=64)
    p.add_argument("--n-aie-cols", type=int, default=4)
    p.add_argument("--dtype_in", type=str, default="bf16")
    p.add_argument("--dtype_out", type=str, default="f32")
    p.add_argument("--emulate-bf16-mmul-with-bfp16", type=int, default=1)
    p.add_argument("--use_chess", type=int, default=0)
    p.add_argument("--scalar", type=int, default=0)
    p.add_argument("--b_col_maj", type=int, default=0)
    p.add_argument("--c_col_maj", type=int, default=0)
    return p


def _compile_kwargs(opts):
    return dict(
        M=opts.M, K=opts.K, N=opts.N,
        m=opts.m, k=opts.k, n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        scalar=bool(opts.scalar),
    )


def _run_and_verify(opts):
    rng = np.random.default_rng(1726250518)
    A_np = (rng.random((opts.M, opts.K)) * 2.0 - 1.0).astype(bfloat16)
    B_np = (rng.random((opts.K, opts.N)) * 2.0 - 1.0).astype(bfloat16)
    A_t = iron.tensor(A_np.reshape(-1), dtype=bfloat16, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=bfloat16, device="npu")
    C_t = iron.zeros(opts.M * opts.N, dtype=np.float32, device="npu")

    bench = run_iters(
        gemm_handwritten, A_t, B_t, C_t,
        M=opts.M, K=opts.K, N=opts.N,
        m=opts.m, k=opts.k, n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        scalar=bool(opts.scalar),
        warmup=opts.warmup, iters=opts.iters,
    )
    expected = (A_np.astype(np.float32) @ B_np.astype(np.float32)).astype(np.float32)
    actual = C_t.numpy().reshape(opts.M, opts.N)
    assert_close_with_benchmark(
        actual, expected, bench=bench,
        ops=2.0 * opts.M * opts.K * opts.N,
        fail_msg="output does not match A @ B",
    )

    total_ops = 2.0 * opts.M * opts.K * opts.N
    median_s = getattr(bench, 'median_us', bench.median_s) / 1e6
    gflops = total_ops / (median_s * 1e9)
    n_tiles = 4 * opts.n_aie_cols

    print()
    print("=" * 70)
    print(f"Hand-Written Kernel: M={opts.M}, K={opts.K}, N={opts.N}")
    print(f"Tiles: {n_tiles} (4×{opts.n_aie_cols})  |  "
          f"tile: {opts.m}×{opts.k}×{opts.n}")
    print(f"μkernel: 8×8×8 BFP16 (512 MACs/insn)  |  2×2 mmul expansion")
    print(f"Peano pragmas: _Pragma(clang loop min_iteration_count(8))")
    print(f"               _Pragma(clang loop pipeline_initiation_interval(1))")
    print(f"Median: {median_s*1e6:.1f} µs  |  {gflops:.1f} GFLOPS  |  "
          f"{gflops/1000:.3f} TFLOPS")
    print(f"Per-tile: {gflops/n_tiles:.1f} GFLOPS")
    print("=" * 70)
    print()


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gemm_handwritten, opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
