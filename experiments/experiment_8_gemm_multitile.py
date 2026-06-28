#!/usr/bin/env python3
"""Experiment 8: Multi-tile BF16 GEMM via Worker.grid + Peano.

Scales the optimized single-core GEMM from experiment 7 to multiple AIE2P tiles
using ``Worker.grid``. Each tile computes an (M_TILE, N_TILE) sub-block of C,
with K-tiling over the shared dimension.

Target: 32 tiles → >50 GFLOPS (step toward 1 TFLOP)

Usage:
    source /home/bcloud/mlir-aie/.venv/bin/activate
    export PYTHONPATH=/home/bcloud/mlir-aie/install/python:$PYTHONPATH
    export PATH=.../mlir/bin:.../llvm-aie/bin:.../build/bin:$PATH
    python3 experiments/experiment_8_gemm_multitile.py --dev npu2 -w 2 -i 20
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

try:
    import aie.iron as iron
    from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, kernels
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


# =============================================================================
# Multi-tile GEMM configuration
# =============================================================================
# Global dimensions — scale up for meaningful TFLOPS measurement
M_GLOBAL = 512     # rows (must be divisible by M_TILE * N_AIE_ROWS)
K_GLOBAL = 2048    # inner dim (must be divisible by K_TILE)
N_GLOBAL = 512     # cols (must be divisible by N_TILE * N_AIE_COLS)

# Per-tile dimensions — fits 32KB L1 with double buffering (depth=2)
M_TILE   = 64
K_TILE   = 64
N_TILE   = 32

# Tile grid — scale up from (1,1) to (4,8) for 32 tiles
N_AIE_ROWS = 2    # rows of AIE cores (max 4 on Strix Halo)
N_AIE_COLS = 4    # cols of AIE cores (max 8 on Strix Halo)

N_TILES = N_AIE_ROWS * N_AIE_COLS
FIFO_DEPTH = 2

# Validation
_L1_BYTES = 2 * (M_TILE * K_TILE * 2 + K_TILE * N_TILE * 2 + M_TILE * N_TILE * 4)
assert M_GLOBAL % (M_TILE * N_AIE_ROWS) == 0, \
    f"M_GLOBAL ({M_GLOBAL}) must be divisible by M_TILE*N_AIE_ROWS ({M_TILE}*{N_AIE_ROWS})"
assert K_GLOBAL % K_TILE == 0
assert N_GLOBAL % (N_TILE * N_AIE_COLS) == 0, \
    f"N_GLOBAL ({N_GLOBAL}) must be divisible by N_TILE*N_AIE_COLS ({N_TILE}*{N_AIE_COLS})"

K_STEPS = K_GLOBAL // K_TILE
M_TILES_PER_ROW = M_GLOBAL // (M_TILE * N_AIE_ROWS)
N_TILES_PER_COL = N_GLOBAL // (N_TILE * N_AIE_COLS)
TILES_PER_CORE = M_TILES_PER_ROW * N_TILES_PER_COL

print(f"[config] Global: M={M_GLOBAL} K={K_GLOBAL} N={N_GLOBAL}  "
      f"Tile: {M_TILE}×{K_TILE}×{N_TILE}  "
      f"Grid: {N_AIE_ROWS}×{N_AIE_COLS}={N_TILES} cores  "
      f"L1/2buf: {_L1_BYTES}B  K-steps: {K_STEPS}  "
      f"Tiles/core: {TILES_PER_CORE}")


# =============================================================================
# Multi-tile GEMM design
# =============================================================================

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential", "--dynamic-objFifos"])
def gemm_multitile(A: In, B: In, C: Out):
    matmul_kernel = kernels.mm(
        dim_m=M_TILE, dim_k=K_TILE, dim_n=N_TILE,
        input_dtype=bfloat16, output_dtype=np.float32,
        use_chess=False, emulate_bf16_mmul_with_bfp16=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert M_TILE % r == 0 and K_TILE % s == 0 and N_TILE % t == 0

    # -- Buffer types --
    A_ty    = np.ndarray[(M_GLOBAL * K_GLOBAL,), np.dtype[bfloat16]]
    B_ty    = np.ndarray[(K_GLOBAL * N_GLOBAL,), np.dtype[bfloat16]]
    C_ty    = np.ndarray[(M_GLOBAL * N_GLOBAL,), np.dtype[np.float32]]
    A_l2_ty = np.ndarray[(M_TILE * K_TILE,), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(K_TILE * N_TILE,), np.dtype[bfloat16]]
    C_l2_ty = np.ndarray[(M_TILE * N_TILE * N_AIE_ROWS,), np.dtype[np.float32]]
    A_l1_ty = np.ndarray[(M_TILE, K_TILE), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(K_TILE, N_TILE), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(M_TILE, N_TILE), np.dtype[np.float32]]

    # -- Streaming dims for BFP16 (r,s,t) = (8,8,8) --
    a_dims = [(M_TILE // r, r * K_TILE), (K_TILE // s, s), (r, K_TILE), (s, 1)]
    b_dims = [(K_TILE // s, s * N_TILE), (N_TILE // t, t), (s, N_TILE), (t, 1)]
    c_dims = [(M_TILE // r, r * N_TILE), (r, t), (N_TILE // t, r * t), (t, 1)]

    # -- A: L3 → L2 → L1 (one stream per row of cores) --
    A_l3l2_fifos = [None] * N_AIE_ROWS
    A_l2l1_fifos = [None] * N_AIE_ROWS
    for row in range(N_AIE_ROWS):
        A_l3l2_fifos[row] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=FIFO_DEPTH)
        A_l2l1_fifos[row] = A_l3l2_fifos[row].cons().forward(
            obj_type=A_l1_ty, name=f"A_L2L1_{row}", dims_to_stream=a_dims,
        )

    # -- B: L3 → L2 → L1 (one stream per column of cores, broadcast) --
    B_l3l2_fifos = [None] * N_AIE_COLS
    B_l2l1_fifos = [None] * N_AIE_COLS
    for col in range(N_AIE_COLS):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=FIFO_DEPTH)
        B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
            obj_type=B_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims,
        )

    # -- C: L1 → L2 → L3 (per-tile output, join rows into column stream) --
    C_l1l2_fifos = [[None] * N_AIE_COLS for _ in range(N_AIE_ROWS)]
    C_l2l3_fifos = [None] * N_AIE_COLS
    for col in range(N_AIE_COLS):
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=FIFO_DEPTH,
        )
        of_offsets = [M_TILE * N_TILE * i for i in range(N_AIE_ROWS)]
        c_tmp = C_l2l3_fifos[col].prod().join(
            of_offsets,
            obj_types=[C_l1_ty] * N_AIE_ROWS,
            names=[f"C_L1L2_{col}_{row}" for row in range(N_AIE_ROWS)],
            depths=[FIFO_DEPTH] * N_AIE_ROWS,
        )
        for row in range(N_AIE_ROWS):
            C_l1l2_fifos[row][col] = c_tmp[row]
        C_l2l3_fifos[col] = C_l2l3_fifos[col]

    # -- Core compute function with K-tiling --
    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range_(TILES_PER_CORE) if TILES_PER_CORE > 1 else range(1)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_STEPS) if K_STEPS > 1 else range(1):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    # -- Worker grid --
    workers = Worker.grid(
        N_AIE_ROWS, N_AIE_COLS,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )

    # -- DMA tiling --
    N_SHIM_MEM_A = min(N_AIE_ROWS, N_AIE_COLS)
    N_A_PER_SHIM = N_AIE_ROWS // N_AIE_COLS if N_AIE_COLS < N_AIE_ROWS else 1

    A_tiles = TensorTiler2D.group_tiler(
        (M_GLOBAL, K_GLOBAL), (M_TILE * N_AIE_ROWS, K_TILE),
        (1, K_STEPS),
        pattern_repeat=N_TILES_PER_COL,
        prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K_GLOBAL, N_GLOBAL), (K_TILE, N_TILE),
        tile_group_repeats=(K_STEPS, N_TILES_PER_COL),
        tile_group_steps=(1, N_AIE_COLS),
        tile_group_col_major=True,
        prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M_GLOBAL, N_GLOBAL), (M_TILE * N_AIE_ROWS, N_TILE),
        tile_group_repeats=(2, N_TILES_PER_COL),
        tile_group_steps=(1, N_AIE_COLS),
        prune_step=False,
    )

    # -- Runtime orchestration (ping-pong DMA) --
    c_idx = 0
    rt = Runtime()

    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*[w for row in workers for w in row])

        for tb in range(iron.ceildiv(M_TILES_PER_ROW, 4)):
            for pp in [0, 1]:
                if c_idx >= len(C_tiles):
                    break

                tg = rt.task_group()
                for col in range(N_AIE_COLS):
                    rt.drain(
                        C_l2l3_fifos[col].cons(), C, tap=C_tiles[c_idx],
                        wait=True, task_group=tg,
                    )
                    c_idx += 1
                    for row in range(N_AIE_ROWS):
                        a_idx = (row + tb * N_AIE_ROWS) % len(A_tiles)
                        rt.fill(
                            A_l3l2_fifos[row].prod(), A, tap=A_tiles[a_idx],
                            task_group=tg,
                        )
                    rt.fill(
                        B_l3l2_fifos[col].prod(), B, tap=B_tiles[col],
                        task_group=tg,
                    )

                if tb > 0 or pp > 0:
                    rt.finish_task_group(tg)
                    tg = rt.task_group()

    return Program(iron.get_current_device(), rt).resolve_program()


# =============================================================================
# CLI + verification
# =============================================================================

def _make_argparser():
    p = argparse.ArgumentParser(
        prog="IRON Multi-Tile GEMM (Peano, bf16→fp32, BFP16)",
        description=f"{N_TILES}-tile BF16 GEMM: {M_GLOBAL}×{K_GLOBAL}×{N_GLOBAL}, "
                    f"tile={M_TILE}×{K_TILE}×{N_TILE}, K-steps={K_STEPS}",
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
        gemm_multitile, A_t, B_t, C_t,
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
    if hasattr(bench, 'median_us'):
        median_s = bench.median_us / 1e6
    elif hasattr(bench, 'median_ms'):
        median_s = bench.median_ms / 1000.0
    else:
        median_s = bench.median_s
    gflops = total_ops / (median_s * 1e9)
    tflops = gflops / 1000.0

    print()
    print("=" * 70)
    print(f"GEMM: M={M_GLOBAL}, K={K_GLOBAL}, N={N_GLOBAL}")
    print(f"Tiles: {N_TILES} ({N_AIE_ROWS}×{N_AIE_COLS})  |  per-tile: {M_TILE}×{K_TILE}×{N_TILE}")
    print(f"mmul: BFP16 8×8×8  |  K-steps: {K_STEPS}  |  Peano (no Chess)")
    print(f"Median: {median_s*1e6:.1f} µs  |  {gflops:.1f} GFLOPS  |  {tflops:.3f} TFLOPS")
    print(f"Per-tile: {gflops/N_TILES:.1f} GFLOPS")
    print("=" * 70)
    print()


def _compile_kwargs(opts):
    return {}


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gemm_multitile, opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
