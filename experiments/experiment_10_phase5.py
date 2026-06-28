#!/usr/bin/env python3
"""Phase 5: Pre-packed BFP16 operand layout for AIE2P mmul-ready format.

Two-stage optimization:
  1. b_col_maj=1 — B stored column-major, streamed in col-major tile order.
     The inner K loop gets SEQUENTIAL B loads (instead of strided).
  2. --pre-pack — CPU-side pre-packing eliminates runtime transpose.

Uses the whole_array.py proven multi-tile design pattern but with
the packed C++ kernel that performs all-sequential inner-loop loads.

Usage:
    source /home/bcloud/mlir-aie/.venv/bin/activate
    export PYTHONPATH=/home/bcloud/mlir-aie/install/python:$PYTHONPATH

    # Stage 1: b_col_maj sequential B loads
    python3 experiments/experiment_10_phase5.py --dev npu2 -w 2 -i 10 \\
        -M 512 -K 8192 -N 512 -m 64 -k 32 -n 64 --n-aie-cols 8 --b-col-maj 1

    # Stage 2: CPU pre-packed + b_col_maj (full Phase 5)
    python3 experiments/experiment_10_phase5.py --dev npu2 -w 2 -i 10 \\
        -M 512 -K 8192 -N 512 -m 64 -k 32 -n 64 --n-aie-cols 8 \\
        --b-col-maj 1 --pre-pack
"""

import argparse
import sys
import os

import numpy as np
from ml_dtypes import bfloat16

# Packing utilities (local, no IRON dependency)
from pack_bfp16 import pack_A_bfp16, pack_B_bfp16_colmaj, unpack_C_f32

try:
    import aie.iron as iron
    from aie.iron import (
        In, ObjectFifo, Out, Program, Runtime, Worker,
        ExternalFunction, CompileTime, kernels, str_to_dtype,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2, from_name
    from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D, TensorAccessPattern
    from aie.utils.hostruntime.argparse import add_compile_args, add_benchmark_args
    from aie.utils.hostruntime.cli import run_design_cli
    from aie.utils.verify import assert_close_with_benchmark
    from aie.utils.benchmark import run_iters
except ImportError as e:
    print(f"IRON not available: {e}")
    print("  source /home/bcloud/mlir-aie/.venv/bin/activate")
    print("  export PYTHONPATH=/home/bcloud/mlir-aie/install/python:$PYTHONPATH")
    sys.exit(1)

_KERNEL_PACKED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kernel_gemm_bfp16_packed.cc",
)
_KERNEL_UNROLL2X = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kernel_gemm_bfp16_unroll2x.cc",
)
_KERNEL_SWP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kernel_gemm_bfp16_swp.cc",
)
_KERNEL_8ACC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kernel_gemm_bfp16_8acc.cc",
)
_KERNEL_VLIW = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kernel_gemm_vliw.cc",
)


def _device_for(dev_str, n_aie_cols):
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def _packed_B_taps(K, N, n, k, n_aie_cols):
    """Generate TensorAccessSequence with one tap per L1 tile.

    Each L1 tile (n×k elements) is a contiguous chunk of the packed array.
    We create one 4D tap per tile so each fill() gets a unique offset — the
    same mechanism TensorTiler2D uses for auto-advancing through data.
    """
    r, s, t = 8, 8, 8
    tile_elems = n * k
    L1_tiles_per_col = (N // n // n_aie_cols) * (K // k)

    seq = TensorAccessSequence(tensor_dims=[K * N], num_steps=0)
    for col in range(n_aie_cols):
        col_start = col * L1_tiles_per_col * tile_elems
        for tile_idx in range(L1_tiles_per_col):
            offset = col_start + tile_idx * tile_elems
            seq.insert(len(seq), TensorAccessPattern(
                tensor_dims=[K * N],
                offset=offset,
                sizes=[n // t, k // s, t, s],
                strides=[t * k, s * t, t, 1],
            ))

    return seq


def _build_phase5_design(
    dev, M, K, N, m, k, n, n_aie_cols,
    dtype_in_str, dtype_out_str,
    b_col_maj, c_col_maj,
    emulate_bf16_mmul_with_bfp16,
    use_chess, scalar,
    pre_pack,  # Phase 5: CPU pre-packing
    no_ii1,    # Disable II=1 pragma to measure its impact
    unroll_2x, # 2× unrolled inner K-loop for better VLIW scheduling
    swp,       # Phase 6: manual software pipelining
    acc8,      # Phase 6: 8-accumulator 2×4 mmul expansion
    vliw,      # Explicit VLIW-scheduled kernel (Chess+Peano compatible)
):
    """Build the packed GEMM IRON design and resolve to MLIR."""
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    r, s, t = (8, 8, 8)

    # Validation
    assert M % (m * n_aie_rows) == 0
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0
    assert m % r == 0 and k % s == 0 and n % t == 0

    # ---- ExternalFunctions: packed kernel ----
    # The packed kernel always uses column-major tile order for B
    # (sequential inner-loop access). When b_col_maj=1, the IRON
    # streaming delivers B in exactly this order.
    is_int8 = (dtype_in_str == 'i8')
    is_bf16 = (dtype_in_str == 'bf16')

    compile_flags_base = [
        f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}",
    ]
    if is_bf16:
        compile_flags_base.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
    if b_col_maj and not pre_pack:
        # BFP16 mmul uses mac_8x8_8x8T (transposed B input)
        # b_col_maj=1 DMA delivers row-major micro-tiles → matches T-input
        # Only add -DB_COL_MAJ for int8 (non-T mmul needs column-major)
        if is_int8:
            compile_flags_base.append("-DB_COL_MAJ")

    if no_ii1:
        compile_flags_base.append("-DNO_II1")
    if vliw:
        kernel_file = _KERNEL_VLIW
        matmul_name = "gemm_bf16_f32_vliw" if is_bf16 else "gemm_i8_i32_vliw"
        zero_name = "zero_f32_vliw" if is_bf16 else "zero_i32_vliw"
    elif acc8:
        kernel_file = _KERNEL_8ACC
        matmul_name = "gemm_bf16_f32_bfp16_8acc" if is_bf16 else "gemm_i8_i32_8acc"
        zero_name = "zero_f32_8acc" if is_bf16 else "zero_i32_8acc"
    elif swp:
        kernel_file = _KERNEL_SWP
        matmul_name = "gemm_bf16_f32_bfp16_swp" if is_bf16 else "gemm_i8_i32_swp"
        zero_name = "zero_f32_swp" if is_bf16 else "zero_i32_swp"
    elif unroll_2x:
        kernel_file = _KERNEL_UNROLL2X
        matmul_name = "gemm_bf16_f32_bfp16_packed_unroll2x" if is_bf16 else "gemm_i8_i32_packed_unroll2x"
        zero_name = "zero_f32_packed_unroll2x" if is_bf16 else "zero_i32_packed_unroll2x"
    else:
        kernel_file = _KERNEL_PACKED
        matmul_name = "gemm_bf16_f32_bfp16_packed" if is_bf16 else "gemm_i8_i32_packed"
        zero_name = "zero_f32_packed" if is_bf16 else "zero_i32_packed"

    matmul_kernel = ExternalFunction(
        name=matmul_name,
        source_file=kernel_file,
        arg_types=[
            np.ndarray[(m * k,), np.dtype[dtype_in]],
            np.ndarray[(k * n,), np.dtype[dtype_in]],
            np.ndarray[(m * n,), np.dtype[dtype_out]],
        ],
        compile_flags=compile_flags_base + ["-DMATMUL_ONLY"],
    )
    zero_kernel = ExternalFunction(
        name=zero_name,
        source_file=kernel_file,
        arg_types=[np.ndarray[(m * n,), np.dtype[dtype_out]]],
        compile_flags=compile_flags_base + ["-DZERO_ONLY"],
    )

    # ---- Buffer types ----
    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    if b_col_maj:
        B_l1_ty = np.ndarray[(n, k), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(n * k,), np.dtype[dtype_in]]
    if pre_pack:
        # Pre-packed B: flat 1D L1 tile, no transpose needed in kernel
        B_l1_ty = np.ndarray[(n * k,), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(n * k,), np.dtype[dtype_in]]

    # ---- ObjectFifos ----
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # A: standard row-major streaming (already sequential in K dimension)
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [[
            (m // r, r * k), (k // s, s), (r, k), (s, 1),
        ]] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i].cons().split(
                of_offsets, obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # B: column-major tile order (sequential K-dimension access!)
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if pre_pack:
            # Pre-packed: data already in kernel-ready order → linear L2→L1 DMA
            B_l2l1_fifos[col] = (
                B_l3l2_fifos[col].cons().forward(
                    obj_type=B_l1_ty, name=f"B_L2L1_{col}",
                )
            )
        else:
            if b_col_maj:
                # Column-major tile order: N tiles outer, K tiles inner
                dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
            else:
                # Row-major tile order: K tiles outer, N tiles inner (strided B loads)
                dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
            B_l2l1_fifos[col] = (
                B_l3l2_fifos[col].cons().forward(
                    obj_type=B_l1_ty, name=f"B_L2L1_{col}",
                    dims_to_stream=dims_to_stream,
                )
            )

        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
            dims_to_stream=(
                [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
                if not c_col_maj
                else [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
            ),
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = (
            C_l2l3_fifos[col].prod().join(
                of_offsets, obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # ---- Core compute function ----
    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_aie_rows, n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(),
             C_l1l2_fifos[row][col].prod(), zero_kernel, matmul_kernel],
            stack_size=0xD00,
        ),
    )

    # ---- Tensor tiling ----
    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_tiles_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_aie_cols, prune_step=False,
    )
    if pre_pack:
        B_tiles = _packed_B_taps(K, N, n, k, n_aie_cols)
    elif b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K), (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1), prune_step=False,
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N), (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True, prune_step=False,
        )
    if c_col_maj:
        C_tiles = TensorTiler2D.step_tiler(
            (N, M), (n, m),
            tile_group_repeats=(N // n // n_aie_cols, n_aie_rows),
            tile_group_steps=(n_aie_cols, 1), iter_col_major=True, prune_step=False,
        )
    else:
        C_tiles = TensorTiler2D.step_tiler(
            (M, N), (m * n_aie_rows, n),
            tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols), prune_step=False,
        )

    # ---- Runtime DMA sequence ----
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*[w for row in workers for w in row])

        tg = rt.task_group()
        c_index = 0
        # Per-column tap indices for pre-packed B (one tap per tile)
        L1_tiles_per_col_b = (N // n // n_aie_cols) * (K // k)
        b_tap_idx = [col * L1_tiles_per_col_b for col in range(n_aie_cols)]
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min([
                    tb_max_n_rows // 2, M // m // n_aie_rows - row_base,
                ])

                for col in range(n_aie_cols):
                    rt.drain(C_l2l3_fifos[col].cons(), C, tap=C_tiles[c_index],
                             wait=True, task_group=tg)
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            rt.fill(A_l3l2_fifos[col].prod(), A,
                                    tap=A_tiles[tile_offset], task_group=tg)
                        b_tap = B_tiles[b_tap_idx[col]] if pre_pack else B_tiles[col]
                        rt.fill(B_l3l2_fifos[col].prod(), B,
                                tap=b_tap, task_group=tg)
                        if pre_pack:
                            b_tap_idx[col] += 1

                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def phase5_packed(
    A: In, B: In, C: Out, *,
    M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    m: CompileTime[int], k: CompileTime[int], n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str], dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0, c_col_maj: CompileTime[int] = 0,
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    use_chess: CompileTime[bool] = False, scalar: CompileTime[bool] = False,
    pre_pack: CompileTime[bool] = False,
    no_ii1: CompileTime[bool] = False,
    unroll_2x: CompileTime[bool] = False,
    swp: CompileTime[bool] = False,
    acc8: CompileTime[bool] = False,
    vliw: CompileTime[bool] = False,
):
    return _build_phase5_design(
        iron.get_current_device(), M, K, N, m, k, n, n_aie_cols,
        dtype_in_str, dtype_out_str, b_col_maj, c_col_maj,
        emulate_bf16_mmul_with_bfp16, use_chess, scalar, pre_pack, no_ii1, unroll_2x, swp, acc8, vliw,
    )


# =============================================================================
# CLI + verification
# =============================================================================

def _make_argparser():
    p = argparse.ArgumentParser(
        prog="Phase 5: Pre-Packed BFP16 GEMM",
        description="BFP16 GEMM with column-major B + optional CPU pre-packing",
    )
    add_compile_args(p, short_dev=None)
    add_benchmark_args(p)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=8192)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=64)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=1,
                   help="1 = column-major B (sequential inner-loop loads)")
    p.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--dtype_in", type=str, default="bf16")
    p.add_argument("--dtype_out", type=str, default="f32")
    p.add_argument("--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=1)
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument("--scalar", type=int, choices=[0, 1], default=0)
    p.add_argument("--pre-pack", action="store_true",
                   help="Pre-pack A and B on CPU for mmul-ready format")
    p.add_argument("--no-ii1", action="store_true",
                   help="Remove II=1 pragma from inner K-loop (measure impact)")
    p.add_argument("--unroll-2x", action="store_true",
                   help="2× unrolled inner K-loop for better VLIW scheduling")
    p.add_argument("--swp", action="store_true",
                   help="Phase 6: manual prologue/kernel/epilogue software pipelining")
    p.add_argument("--8acc", action="store_true", dest="acc8",
                   help="Phase 6: 8-accumulator 2x4 mmul expansion (II=1 capable)")
    p.add_argument("--vliw", action="store_true",
                   help="Explicit VLIW-scheduled kernel (Chess+Peano compatible)")
    p.add_argument("--save-xclbin", action="store_true",
                   help="Save xclbin from JIT cache to saved_xclbins/")
    return p


def _compile_kwargs(opts):
    return dict(
        M=opts.M, K=opts.K, N=opts.N,
        m=opts.m, k=opts.k, n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in, dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj, c_col_maj=opts.c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess), scalar=bool(opts.scalar),
        pre_pack=opts.pre_pack,
        no_ii1=opts.no_ii1,
        unroll_2x=opts.unroll_2x,
        swp=opts.swp,
        acc8=opts.acc8,
        vliw=opts.vliw,
    )


def _run_and_verify(opts):
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)

    rng = np.random.default_rng(1726250518)
    if np.issubdtype(dtype_in, np.integer):
        info = np.iinfo(dtype_in)
        A_np = rng.integers(info.min // 4, info.max // 4,
                            size=(opts.M, opts.K), dtype=dtype_in)
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = rng.integers(info.min // 4, info.max // 4,
                            size=B_shape, dtype=dtype_in)
    else:
        A_np = (rng.random((opts.M, opts.K)) * 4.0).astype(dtype_in)
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = (rng.random(B_shape) * 4.0).astype(dtype_in)

    # Pre-pack B on CPU (Phase 5/6: eliminate transpose + bank conflicts)
    if opts.pre_pack:
        assert opts.b_col_maj == 1, "--pre-pack requires --b-col-maj 1"
    if opts.pre_pack:
        # B_logical is always row-major K×N for packing
        B_logical = B_np if not opts.b_col_maj else B_np.T
        B_packed = pack_B_bfp16_colmaj(B_logical.ravel(), opts.K, opts.N)
        A_dev = iron.tensor(A_np, dtype=dtype_in, device="npu")
        B_dev = iron.tensor(B_packed, dtype=dtype_in, device="npu")
    else:
        A_dev = iron.tensor(A_np, dtype=dtype_in, device="npu")
        B_dev = iron.tensor(B_np, dtype=dtype_in, device="npu")

    C_dev = iron.zeros((opts.M, opts.N), dtype=dtype_out, device="npu")

    # ---- Save xclbin from JIT cache (for XRT + GTT dma-buf path) ----
    if opts.save_xclbin:
        import json, shutil
        from pathlib import Path
        from aie.utils.compile import NPU_CACHE_HOME
        try:
            spec = phase5_packed.specialize(**_compile_kwargs(opts))
            cache_hash = json.loads(spec.compilable.to_json())["cache_hash"]
            cache_dir = Path(NPU_CACHE_HOME) / cache_hash
            xclbin_src = cache_dir / "final.xclbin"
            inst_src = cache_dir / "insts.bin"
            out_dir = Path("/home/bcloud/strixhalo-npu-setup/saved_xclbins")
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"phase5_{opts.dtype_in}_{opts.M}x{opts.K}x{opts.N}_m{opts.m}k{opts.k}n{opts.n}"
            if opts.pre_pack: tag += "_prepack"
            if opts.no_ii1: tag += "_noii1"
            if opts.unroll_2x: tag += "_unroll2x"
            shutil.copy(xclbin_src, out_dir / f"{tag}.xclbin")
            shutil.copy(inst_src, out_dir / f"{tag}_insts.bin")
            print(f"\n💾 xclbin saved: {out_dir / f'{tag}.xclbin'}")
            print(f"💾 insts saved: {out_dir / f'{tag}_insts.bin'}")
        except Exception as e:
            print(f"\n⚠️  xclbin save skipped: {e}")

    bench = run_iters(
        phase5_packed, A_dev, B_dev, C_dev,
        M=opts.M, K=opts.K, N=opts.N,
        m=opts.m, k=opts.k, n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in, dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj, c_col_maj=opts.c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess), scalar=bool(opts.scalar),
        pre_pack=opts.pre_pack,
        no_ii1=opts.no_ii1,
        unroll_2x=opts.unroll_2x,
        swp=opts.swp,
        acc8=opts.acc8,
        vliw=opts.vliw,
        warmup=opts.warmup, iters=opts.iters,
    )

    # Reference: compute expected result
    B_logical = B_np.T if opts.b_col_maj else B_np
    if np.issubdtype(A_np.dtype, np.integer):
        expected = (A_np.astype(np.int64) @ B_logical.astype(np.int64)).astype(dtype_out)
    else:
        expected = (A_np.astype(np.float32) @ B_logical.astype(np.float32)).astype(dtype_out)

    if opts.c_col_maj:
        actual = C_dev.numpy().reshape(opts.N, opts.M)
        expected = expected.T
    else:
        actual = C_dev.numpy().reshape(opts.M, opts.N)

    assert_close_with_benchmark(
        actual, expected, bench=bench,
        ops=2.0 * opts.M * opts.K * opts.N,
        fail_msg="output does not match A @ B",
        mismatch_indices=True,
    )

    total_ops = 2.0 * opts.M * opts.K * opts.N
    npu_stats = bench.npu if bench.npu else bench.e2e
    avg_us = npu_stats.avg_us
    gflops = total_ops / (avg_us * 1e3)  # total_ops / (avg_us * 1e-6 * 1e9) = total_ops / (avg_us * 1e3)
    n_tiles = 4 * opts.n_aie_cols

    print()
    print("=" * 70)
    print(f"Phase 5: M={opts.M}, K={opts.K}, N={opts.N}")
    print(f"Tiles: {n_tiles} (4×{opts.n_aie_cols})  |  "
          f"tile: {opts.m}×{opts.k}×{opts.n}")
    print(f"b_col_maj={opts.b_col_maj}  |  pre_pack={opts.pre_pack}")
    print(f"μkernel: 8×8×8 BFP16 (512 MACs/insn)")
    print(f"B access: {'SEQUENTIAL' if opts.b_col_maj else 'STRIDED'} inner loop")
    print(f"NPU avg/min/max: {npu_stats.avg_us:.1f} / {npu_stats.min_us:.1f} / {npu_stats.max_us:.1f} µs")
    print(f"Performance: {gflops:.1f} GFLOPS  |  {gflops/1000:.3f} TFLOPS")
    print(f"Per-tile: {gflops/n_tiles:.1f} GFLOPS")
    print("=" * 70)
    print()


def _validate_shape_args(opts):
    n_aie_rows = 4
    if opts.M % (opts.m * n_aie_rows) != 0:
        sys.exit(f"-M must be a multiple of -m * n_aie_rows")
    if opts.K % opts.k != 0:
        sys.exit(f"-K must be a multiple of -k")
    if opts.N % (opts.n * opts.n_aie_cols) != 0:
        sys.exit(f"-N must be a multiple of -n * n_aie_cols")
    if opts.dev == "npu" and opts.n_aie_cols > 4:
        sys.exit(f"NPU1 max 4 cols")
    if opts.dev == "npu2" and opts.n_aie_cols > 8:
        sys.exit(f"NPU2 max 8 cols")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        phase5_packed, opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
        validate=_validate_shape_args,
    )


if __name__ == "__main__":
    main()
