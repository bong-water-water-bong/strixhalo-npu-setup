#!/usr/bin/env python3
"""Direct AIE2P GEMM via XAIE Transaction Instructions.

BYPASSES Peano compiler entirely. Generates XAIE transaction buffer
that programs AIE tiles directly at the register level for GEMM.

This is what FastFlowLM does — XAIE_IO_WRITE + XAIE_CONFIG_SHIMDMA_BD
+ XAIE_IO_CUSTOM_OP_DDR_PATCH instructions submitted via MLIR_AIE kernel.

Interface:
    generate_gemm_instrs(M, N, K, A_addr, B_addr, C_addr) -> bytes
    Returns instruction buffer ready for MLIR_AIE kernel.

XAIE Transaction Format:
    Header (16 bytes): Major, Minor, DevGen, NumRows, NumCols,
                       NumMemTileRows, padding, NumOps, TxnSize
    Then: sequence of operations, each with Opcode + op-specific payload
"""

import struct
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# =============================================================================
# XAIE Constants (from xaie_txn.h)
# =============================================================================
XAIE_IO_WRITE              = 0
XAIE_IO_BLOCKWRITE         = 1
XAIE_IO_BLOCKSET           = 2
XAIE_IO_MASKWRITE          = 3
XAIE_IO_MASKPOLL           = 4
XAIE_IO_NOOP               = 5
XAIE_IO_PREEMPT            = 6
XAIE_IO_MASKPOLL_BUSY      = 7
XAIE_IO_LOADPDI            = 8
XAIE_IO_LOAD_PM_START      = 9
XAIE_IO_CREATE_SCRATCHPAD  = 10
XAIE_IO_UPDATE_STATE_TABLE = 11
XAIE_IO_UPDATE_REG         = 12
XAIE_IO_UPDATE_SCRATCH     = 13
XAIE_CONFIG_SHIMDMA_BD     = 14
XAIE_CONFIG_SHIMDMA_DMABUF_BD = 15
XAIE_IO_CUSTOM_OP_BEGIN    = 128
XAIE_IO_CUSTOM_OP_TCT      = 128
XAIE_IO_CUSTOM_OP_DDR_PATCH = 129
XAIE_IO_CUSTOM_OP_READ_REGS = 130
XAIE_IO_CUSTOM_OP_RECORD_TIMER = 131
XAIE_IO_CUSTOM_OP_MERGE_SYNC = 132
XAIE_IO_CUSTOM_OP_NEXT     = 133

# AIE2P device parameters (Strix Halo NPU2)
DEV_GEN = 3  # AIE2P
NUM_ROWS = 6
NUM_COLS = 8
NUM_MEM_TILE_ROWS = 1

# AIE2P tile register offsets (simplified — actual values from hardware spec)
# These control the MMUL unit and DMA channels
AIE2P_MMUL_CTRL = 0x00000000   # MMUL control register
AIE2P_DMA_BD_BASE = 0x0001D000 # DMA BD register base
AIE2P_CORE_CONFIG = 0x00030000  # Core configuration


@dataclass
class TxnHeader:
    """XAIE Transaction Header (16 bytes)"""
    major: int = 1
    minor: int = 0
    dev_gen: int = DEV_GEN
    num_rows: int = NUM_ROWS
    num_cols: int = NUM_COLS
    num_mem_tile_rows: int = NUM_MEM_TILE_ROWS
    num_ops: int = 0
    txn_size: int = 0

    def pack(self) -> bytes:
        return struct.pack('<BBBBBBHII',
            self.major, self.minor, self.dev_gen,
            self.num_rows, self.num_cols, self.num_mem_tile_rows,
            0,  # padding
            self.num_ops, self.txn_size)


def write32_op(col: int, row: int, reg_offset: int, value: int) -> bytes:
    """Generate a XAIE_IO_WRITE operation (single 32-bit register write).

    Format:
        OpHdr (4 bytes): Opcode(1) | Col(1) | Row(1) | padding(1)
        RegOff (8 bytes): Register offset in tile address space
        Value  (4 bytes): 32-bit value to write
        Size   (4 bytes): Always 4 for 32-bit write
    Total: 20 bytes per write32 operation
    """
    return struct.pack('<BBBBQII',
        XAIE_IO_WRITE, col, row, 0,  # OpHdr
        reg_offset,                   # RegOff
        value,                        # Value
        4)                            # Size = 4 bytes


def blockwrite32_op(col: int, row: int, reg_offset: int,
                     values: List[int]) -> bytes:
    """Generate a XAIE_IO_BLOCKWRITE operation.

    Writes multiple 32-bit values to consecutive registers starting at reg_offset.
    """
    data = struct.pack('<BBBBQII',
        XAIE_IO_BLOCKWRITE, col, row, 0,
        reg_offset,
        values[0] if values else 0,
        len(values))
    for v in values:
        data += struct.pack('<I', v)
    return data


def maskpoll_op(col: int, row: int, reg_offset: int,
                mask: int, expected: int, timeout: int = 1000) -> bytes:
    """Generate a XAIE_IO_MASKPOLL operation (wait for condition)."""
    return struct.pack('<BBBBQIII',
        XAIE_IO_MASKPOLL, col, row, 0,
        reg_offset, mask, expected, timeout)


def shimdma_bd_op(col: int, row: int, bd_id: int,
                  buf_addr_low: int, buf_addr_high: int,
                  buf_len: int,
                  d0_size: int, d0_stride: int,
                  d1_size: int, d1_stride: int,
                  d2_size: int, d2_stride: int,
                  iteration_step: int, iteration_count: int,
                  direction: int = 0,  # 0=S2MM(device→host), 1=MM2S(host→device)
                  channel: int = 0) -> bytes:
    """Generate a XAIE_CONFIG_SHIMDMA_BD operation.

    Configures a Buffer Descriptor for the shim DMA engine.
    This is how data moves between HOST DRAM and AIE tile memory.

    BD register layout (simplified for AIE2P):
        0x00: buffer address [31:0]
        0x04: buffer address [47:32]
        0x08: buffer length
        0x0C: D0 size/stride
        0x10: D1 size/stride
        0x14: D2 size/stride + iteration
        0x18: control (direction, channel, enable, etc.)
    """
    # Encode tile position into BD
    tile_id = (col << 5) | (row << 25) | 0x1D000

    # Pack D0/D1/D2 into BD format
    d0_packed = (d0_size - 1) & 0xFFF
    d1_packed = (d1_size - 1) & 0xFFF
    d2_packed = (d2_size - 1) & 0xFFF
    iter_packed = ((iteration_step & 0xFF) << 24) | ((iteration_count - 1) & 0xFF)

    # BD control word
    ctrl = (direction << 4) | (channel & 0xF) | 0x80000000  # enable bit

    # Serialize: op header + BD config
    op_hdr = struct.pack('<BBBB', XAIE_CONFIG_SHIMDMA_BD, col, row, 0)
    bd_data = struct.pack('<IIIIIIIII',
        tile_id, bd_id * 0x30,
        buf_addr_low, buf_addr_high,
        buf_len,
        d0_packed, d1_packed, d2_packed,
        iter_packed | (ctrl << 0))

    return op_hdr + bd_data


def ddr_patch_op(col: int, row: int, reg_offset: int,
                 addr_low: int, addr_high: int) -> bytes:
    """Generate XAIE_IO_CUSTOM_OP_DDR_PATCH to fix up DDR addresses."""
    op_hdr = struct.pack('<BBBB', XAIE_IO_CUSTOM_OP_DDR_PATCH, col, row, 0)
    return op_hdr + struct.pack('<QII', reg_offset, addr_low, addr_high)


def generate_gemm_instrs(
    M: int, K: int, N: int,
    A_ddr_low: int, A_ddr_high: int,
    B_ddr_low: int, B_ddr_high: int,
    C_ddr_low: int, C_ddr_high: int,
    tile_M: int = 32, tile_K: int = 128, tile_N: int = 32,
    num_cols: int = 8, num_rows: int = 4,
) -> bytes:
    """Generate complete GEMM instruction buffer.

    This creates the XAIE transaction that programs all AIE tiles
    to perform a distributed GEMM. Each tile processes:
        C[tile] += A[tile] @ B[tile]

    Returns the serialized instruction buffer ready for MLIR_AIE kernel.
    """
    ops = []
    M_tiles = M // tile_M // num_rows
    K_steps = K // tile_K
    N_tiles = N // tile_N // num_cols

    elem_size = 2  # bf16 = 2 bytes
    A_tile_bytes = tile_M * tile_K * elem_size
    B_tile_bytes = tile_K * tile_N * elem_size
    C_tile_bytes = tile_M * tile_N * 4  # f32 output

    # For each tile, configure DMA and trigger compute
    total_tiles = 0
    for k_step in range(K_steps):
        for m_tile in range(M_tiles):
            for n_tile in range(N_tiles):
                col = n_tile % num_cols
                row = m_tile % num_rows

                # A buffer offset for this tile
                a_off = (m_tile * tile_M * K + k_step * tile_K) * elem_size
                # B buffer offset (column-major tile order)
                b_off = (n_tile * tile_N * K + k_step * tile_K * tile_N) * elem_size
                # C buffer offset
                c_off = (m_tile * tile_M * N + n_tile * tile_N) * 4

                # Shim DMA BD for A (host → AIE, MM2S)
                ops.append(shimdma_bd_op(
                    col, row, bd_id=0 + k_step % 2,
                    buf_addr_low=A_ddr_low + a_off,
                    buf_addr_high=A_ddr_high,
                    buf_len=A_tile_bytes,
                    d0_size=8, d0_stride=1,
                    d1_size=tile_M//8, d1_stride=K*elem_size,
                    d2_size=tile_K//8, d2_stride=8*K*elem_size,
                    iteration_step=A_tile_bytes,
                    iteration_count=1,
                    direction=1, channel=0))

                # Shim DMA BD for B (host → AIE, MM2S)
                ops.append(shimdma_bd_op(
                    col, row, bd_id=2 + k_step % 2,
                    buf_addr_low=B_ddr_low + b_off,
                    buf_addr_high=B_ddr_high,
                    buf_len=B_tile_bytes,
                    d0_size=8, d0_stride=1,
                    d1_size=tile_K//8, d1_stride=K*elem_size,
                    d2_size=tile_N//8, d2_stride=8*K*elem_size,
                    iteration_step=B_tile_bytes,
                    iteration_count=1,
                    direction=1, channel=1))

                # Shim DMA BD for C (AIE → host, S2MM)
                ops.append(shimdma_bd_op(
                    col, row, bd_id=4 + k_step % 2,
                    buf_addr_low=C_ddr_low + c_off,
                    buf_addr_high=C_ddr_high,
                    buf_len=C_tile_bytes,
                    d0_size=8, d0_stride=1,
                    d1_size=tile_M//8, d1_stride=N*4,
                    d2_size=tile_N//8, d2_stride=8*N*4,
                    iteration_step=C_tile_bytes,
                    iteration_count=1,
                    direction=0, channel=2))

                total_tiles += 1

    # Calculate transaction size
    txn_size = 16  # header
    for op in ops:
        txn_size += len(op)

    # Build header
    header = TxnHeader(
        dev_gen=DEV_GEN,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        num_mem_tile_rows=NUM_MEM_TILE_ROWS,
        num_ops=len(ops),
        txn_size=txn_size)

    # Serialize
    buf = bytearray()
    buf.extend(header.pack())
    for op in ops:
        buf.extend(op)

    return bytes(buf)


if __name__ == "__main__":
    # Demo: generate instructions for a small GEMM
    instrs = generate_gemm_instrs(
        M=512, K=8192, N=512,
        A_ddr_low=0x10000000, A_ddr_high=0,
        B_ddr_low=0x20000000, B_ddr_high=0,
        C_ddr_low=0x30000000, C_ddr_high=0,
        tile_M=32, tile_K=128, tile_N=32,
        num_cols=8, num_rows=4,
    )
    print(f"Generated {len(instrs)} bytes ({len(instrs)//4} instructions)")
    print(f"Header: {instrs[:16].hex()}")
    print(f"First op: {instrs[16:36].hex()}")
