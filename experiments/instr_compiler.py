#!/usr/bin/env python3
"""NPU Instruction Compiler — generates MLIR_AIE instruction buffers.

Parses, modifies, and regenerates instruction buffers in the exact format
used by both IRON and FastFlowLM for the MLIR_AIE DPU kernel.

Instruction format (reverse-engineered from IRON insts.bin + libgemm.so):

Header (4 words):
  [0] = (num_mem_tile_rows << 24) | (num_cols << 16) | (num_rows << 8) | version
  [1] = flags/config
  [2] = num_ops (total command count)
  [3] = total_size (bytes)

Commands:
  WRITE (0x00):    24 bytes — opcode(1) | pad(7) | addr_lo(4) | addr_hi(4) | value(4) | opSize(4)
  BLOCKWRITE (0x01): 48 bytes — opcode(1) | pad(7) | addr(4) | opSize(4) | bd_payload(32)
  MASKWRITE (0x03): 28 bytes — opcode(1) | pad(7) | addr_lo(4) | addr_hi(4) | value(4) | mask(4) | opSize(4)
  SYNC (0x80):      16 bytes — opcode(1) | pad(3) | opSize(4) | descriptor(4) | config(4)
  DDR_PATCH (0x81): 48 bytes — opcode(1) | pad(3) | opSize(4) | pad(12) | action(4) | patch_addr(4) | pad(4) | arg_idx(4) | pad(4) | arg_plus(4)

BD payload (32 bytes = 8 words within BLOCKWRITE):
  [0] = buffer_length (bytes)
  [1] = buffer_addr_lo (PATCHED by DDR_PATCH)
  [2] = buffer_addr_hi
  [3] = control (0x02000000=1D, 0x04000000=2D)
  [4] = dma_config_0
  [5] = dma_config_1
  [6] = dma_config_2
  [7] = lock_acq_rel

DDR_PATCH maps:
  arg_idx=0 → kernel arg 3 (A buffer)
  arg_idx=1 → kernel arg 4 (B buffer)
  arg_idx=2 → kernel arg 5 (C buffer)
  arg_plus = byte offset within the buffer
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import copy

# =============================================================================
# Command Types
# =============================================================================

@dataclass
class WriteCmd:
    addr: int
    value: int

    def pack(self) -> bytes:
        hdr = struct.pack('<BBBBxxII', 0x00, 0, 0, 0, self.addr & 0xFFFFFFFF, self.addr >> 32)
        return hdr + struct.pack('<II', self.value, 24)

@dataclass
class BlockWriteCmd:
    """DMA BD template. Written to NPU instruction memory."""
    addr: int
    bd: List[int]  # 8 words of BD descriptor

    def pack(self) -> bytes:
        hdr = struct.pack('<BBBBxxII', 0x01, 0, 0, 0, self.addr, 48)
        bd_data = b''.join(struct.pack('<I', w) for w in self.bd)
        assert len(bd_data) == 32, f"BD must be 8 words (32 bytes), got {len(bd_data)}"
        return hdr + bd_data

@dataclass
class SyncCmd:
    """DMA synchronization barrier."""
    col: int; row: int; direction: int

    def pack(self) -> bytes:
        desc = (self.col << 16) | (self.row << 8) | self.direction
        return struct.pack('<BBxxIIII', 0x80, 0, 0, 0, 16, desc, 0)

@dataclass
class DdrPatchCmd:
    """Patches a buffer address into a BD at runtime."""
    target_addr: int  # Instruction memory address to patch
    arg_idx: int      # 0=A, 1=B, 2=C
    arg_plus: int     # Byte offset within buffer

    def pack(self) -> bytes:
        # DDR_PATCH: 48 bytes total
        # [0:4] opcode,pad  [4:8] opSize=48  [8:20] reserved
        # [20:24] action  [24:28] patch_addr  [28:32] reserved
        # [32:36] arg_idx  [36:40] reserved  [40:44] arg_plus  [44:48] reserved
        return struct.pack('<II', 0x81, 48) + bytes(12) + \
               struct.pack('<III', 0, self.target_addr, 0) + \
               struct.pack('<III', self.arg_idx, 0, self.arg_plus)

# =============================================================================
# Instruction Buffer Builder
# =============================================================================

class InstrBuffer:
    """Builds MLIR_AIE instruction buffers for GEMM."""

    def __init__(self, num_rows=6, num_cols=8, num_mem_rows=1):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_mem_rows = num_mem_rows
        self.cmds: List = []

    def add_write(self, addr: int, value: int):
        self.cmds.append(WriteCmd(addr, value))

    def add_block_write(self, addr: int, bd: List[int]):
        self.cmds.append(BlockWriteCmd(addr, bd))

    def add_sync(self, col: int, row: int, direction: int):
        self.cmds.append(SyncCmd(col, row, direction))

    def add_ddr_patch(self, target_addr: int, arg_idx: int, arg_plus: int):
        self.cmds.append(DdrPatchCmd(target_addr, arg_idx, arg_plus))

    def build(self) -> bytes:
        """Serialize all commands into the instruction buffer."""
        # Header word 0
        hdr0 = (self.num_mem_rows << 24) | (self.num_cols << 16) | (self.num_rows << 8) | 0x00

        # Serialize commands
        body = b''
        for cmd in self.cmds:
            body += cmd.pack()

        total_size = 16 + len(body)
        buf = bytearray()
        buf.extend(struct.pack('<IIII', hdr0, 0x108, len(self.cmds), total_size))
        buf.extend(body)

        return bytes(buf)

# =============================================================================
# Parser: reads existing insts.bin and allows modification
# =============================================================================

def parse_insts(data: bytes):
    """Parse an existing instruction buffer into modifiable commands."""
    hdr = struct.unpack_from('<IIII', data, 0)
    num_rows = (hdr[0] >> 8) & 0xFF
    num_cols = (hdr[0] >> 16) & 0xFF
    num_mem = (hdr[0] >> 24) & 0xFF
    num_ops = hdr[2]

    buf = InstrBuffer(num_rows, num_cols, num_mem)
    offset = 16

    for _ in range(num_ops):
        if offset >= len(data):
            break
        opcode = data[offset]

        if opcode == 0x00:  # WRITE
            addr_lo = struct.unpack_from('<I', data, offset+8)[0]
            addr_hi = struct.unpack_from('<I', data, offset+12)[0]
            value = struct.unpack_from('<I', data, offset+16)[0]
            buf.cmds.append(WriteCmd(addr_lo | (addr_hi << 32), value))
            offset += 24

        elif opcode == 0x01:  # BLOCKWRITE
            addr = struct.unpack_from('<I', data, offset+8)[0]
            bd = [struct.unpack_from('<I', data, offset+16+i*4)[0] for i in range(8)]
            buf.cmds.append(BlockWriteCmd(addr, bd))
            offset += 48

        elif opcode == 0x03:  # MASKWRITE
            offset += 28

        elif opcode == 0x80:  # SYNC
            offset += 16

        elif opcode == 0x81:  # DDR_PATCH
            patch_addr = struct.unpack_from('<I', data, offset+24)[0]
            arg_idx = struct.unpack_from('<I', data, offset+32)[0]
            arg_plus = struct.unpack_from('<I', data, offset+40)[0]
            buf.cmds.append(DdrPatchCmd(patch_addr, arg_idx, arg_plus))
            offset += 48

        else:
            offset += 4  # skip unknown

    return buf

# =============================================================================
# GEMM Instruction Generator
# =============================================================================

def generate_gemm_instrs(M: int, K: int, N: int,
                          tile_m: int = 32, tile_k: int = 128, tile_n: int = 32,
                          num_rows: int = 4, num_cols: int = 8,
                          mem_rows: int = 1, mem_cols: int = 1) -> bytes:
    """Generate a complete GEMM instruction buffer.

    This is a WORK IN PROGRESS — currently generates a minimal valid buffer.
    The full implementation requires exact knowledge of AIE2P register addresses
    and DMA channel assignments for the target xclbin.
    """
    buf = InstrBuffer(num_rows=6, num_cols=num_cols, num_mem_rows=mem_rows)
    return buf.build()

# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    # Test: parse existing IRON insts.bin
    data = np.fromfile(sys.argv[1] if len(sys.argv) > 1 else
                       "/home/bcloud/strixhalo-npu-setup/saved_xclbins/fresh_insts.bin",
                       dtype=np.uint8)

    parsed = parse_insts(bytes(data))
    print(f"Parsed {len(parsed.cmds)} commands from {len(data)} bytes")

    # Count command types
    from collections import Counter
    types = Counter(type(c).__name__ for c in parsed.cmds)
    for t, n in sorted(types.items()):
        print(f"  {t}: {n}")

    # Show DDR patch mapping
    patches = [c for c in parsed.cmds if isinstance(c, DdrPatchCmd)]
    print(f"\nDDR Patches: {len(patches)}")
    for p in patches[:5]:
        print(f"  addr=0x{p.target_addr:08x} arg={p.arg_idx} plus=0x{p.arg_plus:08x}")

    # Verify round-trip
    rebuilt = parsed.build()
    print(f"\nRound-trip: {len(data)} → {len(rebuilt)} bytes")
    if bytes(data) == rebuilt[:len(data)]:
        print("✅ Exact match!")
    else:
        diff = sum(1 for i in range(min(len(data), len(rebuilt))) if data[i] != rebuilt[i])
        print(f"❌ {diff} bytes differ")
