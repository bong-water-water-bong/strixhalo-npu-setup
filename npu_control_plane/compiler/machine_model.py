"""AIE2P VLIW machine model for custom scheduling (corrected v2).

Updated with verified microarchitecture data from:
- mlir-aie AIETargetModel.cpp (TargetModel.cpp lines 1602-1614)
- mlir-aie XLLVMAIE2IntrOps.td (AIE2P intrinsics, lines 96-152)
- mlir-aie programming_guide/section-4/section-4c/README.md
- mlir-aie aie_kernels/aie_kernel_utils.h (pipeline pragmas)
- mlir-aie skills/aie-kernel-opt/SKILL.md (optimization catalog)
- AMD AIE-ML Architecture Manual (AM020)
- LLVM Discourse: Peano scheduling discussion
- AMD Adaptive Support: AIE loop pipelining measurements

Key corrections from v1:
- L1 data: 64 KB (not 32 KB) for AIE2P / Strix Halo
- VLIW: 7-way (not 3-way) with 2 load units + 1 store unit + 1 vector op
- Vector regs: 12 (not 16), Accumulator regs: 9 (not 8)
- BFP16 intrinsics: MacConfBFP576ACC2048 (64xI8 exp + 64xI8 data)
- Bank conflict penalty: 1 cycle (not 4)
- Logical register widths: 512/1024/2048 bits via concatenation
- Scalar ld latency: 7 cycles, mul: 5 cycles

All values independently derived from public sources.
No Chess/proprietary internals are referenced.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# AIE2P Architecture Constants — verified from public sources
# =============================================================================

# Clock frequency (Strix Halo NPU2 — Ryzen AI MAX+ 395)
AIE2P_CLOCK_MHZ = 1600  # 1.6 GHz

# ---- Memory (AIETargetModel.cpp lines 1605-1608) ----
AIE2P_L1_DATA_BYTES = 64 * 1024      # 64 KB data memory per compute tile
AIE2P_L1_BANKS = 4                    # 4 banks
AIE2P_L1_BANK_BYTES = 16 * 1024      # 16 KB per bank (64 KB / 4 banks)
AIE2P_L1_BANK_CONFLICT_PENALTY = 1   # cycles (section-4c: "The bank conflict penalty is 1 cycle")
AIE2P_PROGRAM_MEMORY_BYTES = 32 * 1024  # 32 KB program memory

# ---- Vector unit (section-4c lines 175-179) ----
AIE2P_VECTOR_WIDTH_BITS = 512         # Physical vector register width
AIE2P_VECTOR_REGS = 12                # v0-v11 (verified: section-4c line 175)
AIE2P_ACCUMULATOR_REGS = 9            # a0-a8 (verified: section-4c line 179)

# AIE2P supports logical registers up to 2048 bits by concatenating physical regs:
#   I512 (512-bit):  1 vector reg, 1 acc reg
#   I1024 (1024-bit): 2 vector regs, 2 acc regs
#   ACC2048 (2048-bit): 4 acc regs (accumulator only)

# ---- Scalar unit ----
AIE2P_SCALAR_REGS = 32                # 32 general-purpose 32-bit scalar regs
AIE2P_SCALAR_WIDTH_BITS = 32

# ---- L1 Load/Store bandwidth (section-4c line 175) ----
# "2 parallel Load Units, each capable of loading 256 bits per clock cycle"
AIE2P_LOAD_BANDWIDTH_BITS_PER_CYCLE = 512   # 2 × 256-bit
AIE2P_STORE_BANDWIDTH_BITS_PER_CYCLE = 256  # 1 × 256-bit


# =============================================================================
# VLIW Issue Slots — 7-way VLIW (INFN Workshop March 2025 + section-4c)
# =============================================================================

class IssueSlot(Enum):
    """AIE2P VLIW issue slots (7 total per bundle).

    The AIE2P issues a single VLIW bundle per clock cycle with these slots:
    - SCALAR:         Scalar ALU / Branch / Address calculation (32-bit)
    - MOVE_0, MOVE_1: Up to 2 scalar register moves (32-bit each)
    - LOAD_0, LOAD_1: 2 Vector reads from L1 — VLDA + VLDB (256-bit each)
    - STORE:          1 Vector write to L1 — VST (256-bit)
    - VECTOR:         1 Vector instruction — VMAC/VMUL/VSHUFFLE/VADD
    """
    SCALAR = 0       # Slot 1: Scalar ALU, branch, address calc
    MOVE_0 = 1       # Slot 2: Scalar move
    MOVE_1 = 2       # Slot 3: Scalar move
    LOAD_0 = 3       # Slot 4: Vector load A (VLDA)
    LOAD_1 = 4       # Slot 5: Vector load B (VLDB)
    STORE = 5        # Slot 6: Vector store (VST)
    VECTOR = 6       # Slot 7: Vector ALU/MAC

    # Convenience groupings
    @property
    def is_load(self) -> bool:
        return self in (IssueSlot.LOAD_0, IssueSlot.LOAD_1)

    @property
    def is_move(self) -> bool:
        return self in (IssueSlot.MOVE_0, IssueSlot.MOVE_1)


# Slot aliases for code compatibility
LOAD_STORE_SLOTS = [IssueSlot.LOAD_0, IssueSlot.LOAD_1, IssueSlot.STORE]
COMPUTE_SLOTS = [IssueSlot.VECTOR]
SCALAR_SLOTS = [IssueSlot.SCALAR, IssueSlot.MOVE_0, IssueSlot.MOVE_1]


# =============================================================================
# Instruction latencies (cycles) — verified from LLVM Discourse + public docs
# =============================================================================

@dataclass(frozen=True)
class InstructionLatency:
    """Latency in cycles for key AIE2P operations.

    Sources:
    - Scalar ld/lda = 7 cycles, mul = 5 cycles: LLVM Discourse Peano discussion
    - Vector loads = 7 cycles (typical): section-4c analysis
    - mmul 8×8×8 bf16 = 8 cycles: AIE-ML v2 Intrinsics Guide
    - BFP16 mmul = same pipeline depth as BF16
    """
    # MAC operations
    mmul_8x8x8_bf16: int = 8            # aie::mmul<8,8,8,bf16,bf16,accfloat>
    mmul_4x8x8_bf16: int = 4            # aie::mmul<4,8,8> (smaller mac_dims)
    mmul_8x8x8_bfp16: int = 8           # BFP16: same pipeline as BF16 mmul

    # Vector memory
    vector_load: int = 7                # VLDA/VLDB from L1
    vector_store: int = 4               # VST to L1

    # Vector datapath
    vector_shuffle: int = 2             # vshuffle / permute
    bf16_to_accfloat: int = 2           # vconvert: bf16 → accfloat
    accfloat_to_bfp16: int = 2          # vpack: accfloat → bfp16ebs8
    vector_add: int = 2                 # vadd
    vector_mul: int = 2                 # vmul

    # Scalar
    scalar_load: int = 7                # ld/lda (LLVM Discourse)
    scalar_mul: int = 5                 # mul (LLVM Discourse)
    scalar_add: int = 1                 # add
    scalar_cmp: int = 1                 # compare
    scalar_branch: int = 2              # branch (taken)

    # System
    dma_startup: int = 20               # DMA engine startup overhead
    bank_conflict: int = 1              # Per-conflict penalty (section-4c)


# =============================================================================
# Register file model
# =============================================================================

class RegisterClass(Enum):
    VECTOR = auto()       # 512-bit vector (v0-v11)
    ACCUMULATOR = auto()  # Extended-precision accumulator (a0-a8)
    SCALAR = auto()       # 32-bit scalar (32 regs)
    POINTER = auto()      # Address pointer registers


@dataclass
class RegisterPressure:
    """Track register allocation pressure across all classes."""
    vector_used: int = 0
    acc_used: int = 0
    scalar_used: int = 0

    @property
    def vector_free(self) -> int:
        return AIE2P_VECTOR_REGS - self.vector_used

    @property
    def acc_free(self) -> int:
        return AIE2P_ACCUMULATOR_REGS - self.acc_used

    @property
    def scalar_free(self) -> int:
        return AIE2P_SCALAR_REGS - self.scalar_used

    @property
    def is_spilling(self) -> bool:
        """True if any register class is overcommitted."""
        return (self.vector_used > AIE2P_VECTOR_REGS or
                self.acc_used > AIE2P_ACCUMULATOR_REGS or
                self.scalar_used > AIE2P_SCALAR_REGS)

    def allocate(self, reg_class: RegisterClass, count: int = 1) -> bool:
        """Try to allocate registers. Returns True on success."""
        limits = {
            RegisterClass.VECTOR: AIE2P_VECTOR_REGS,
            RegisterClass.ACCUMULATOR: AIE2P_ACCUMULATOR_REGS,
            RegisterClass.SCALAR: AIE2P_SCALAR_REGS,
        }
        current = {
            RegisterClass.VECTOR: self.vector_used,
            RegisterClass.ACCUMULATOR: self.acc_used,
            RegisterClass.SCALAR: self.scalar_used,
        }
        if current[reg_class] + count <= limits[reg_class]:
            if reg_class == RegisterClass.VECTOR:
                self.vector_used += count
            elif reg_class == RegisterClass.ACCUMULATOR:
                self.acc_used += count
            elif reg_class == RegisterClass.SCALAR:
                self.scalar_used += count
            return True
        return False

    def free(self, reg_class: RegisterClass, count: int = 1) -> None:
        if reg_class == RegisterClass.VECTOR:
            self.vector_used = max(0, self.vector_used - count)
        elif reg_class == RegisterClass.ACCUMULATOR:
            self.acc_used = max(0, self.acc_used - count)
        elif reg_class == RegisterClass.SCALAR:
            self.scalar_used = max(0, self.scalar_used - count)

    def logical_regs_needed(self, width_bits: int) -> int:
        """Physical registers consumed by a logical register of given width.

        AIE2P supports logical registers up to 2048 bits by concatenating:
        - 512-bit → 1 physical vector reg
        - 1024-bit → 2 physical vector regs
        - 2048-bit → 4 physical acc regs (accumulator only)
        """
        if width_bits <= 512:
            return 1
        elif width_bits <= 1024:
            return 2
        else:
            return 4


# =============================================================================
# VLIW Bundle — 7-way
# =============================================================================

@dataclass
class AIE2PVLIWBundle:
    """A single VLIW instruction bundle for AIE2P.

    Can contain up to 1 operation per issue slot (7 slots total).
    Unused slots are NOPs.

    Co-issuable combinations:
    - VMAC + VST (different units, fully overlapped)
    - VLDA + VLDB (independent load units, when targeting different banks)
    - Scalar ALU + any vector operation
    - Scalar load/store + vector load + VMAC (all independent pipelines)
    """
    # Slot assignments
    scalar_op: Optional[str] = None       # Slot 0: Scalar ALU/branch
    move_0_op: Optional[str] = None       # Slot 1: Scalar move
    move_1_op: Optional[str] = None       # Slot 2: Scalar move
    load_0_op: Optional[str] = None       # Slot 3: Vector load A
    load_1_op: Optional[str] = None       # Slot 4: Vector load B
    store_op: Optional[str] = None        # Slot 5: Vector store
    vector_op: Optional[str] = None       # Slot 6: Vector ALU/MAC

    cycle_offset: int = 0
    comment: str = ""

    @property
    def nop_count(self) -> int:
        """Count of empty slots in this bundle."""
        ops = [self.scalar_op, self.move_0_op, self.move_1_op,
               self.load_0_op, self.load_1_op, self.store_op, self.vector_op]
        return sum(1 for op in ops if op is None)

    @property
    def utilization(self) -> float:
        """Slot utilization (0.0 = all NOPs, 1.0 = all slots filled)."""
        filled = 7 - self.nop_count
        return filled / 7.0

    @property
    def has_mmul(self) -> bool:
        """True if this bundle contains a matrix multiply."""
        return self.vector_op and 'mmul' in str(self.vector_op)

    @property
    def has_dual_load(self) -> bool:
        """True if both load units are active."""
        return self.load_0_op is not None and self.load_1_op is not None


# =============================================================================
# BFP16 micro-kernel configuration (AIE2P-specific)
# =============================================================================

@dataclass
class BFP16Config:
    """BFP16 (Block Floating Point 16) micro-kernel parameters.

    BFP16 uses 8-bit shared exponents for groups of 8 BF16 values.
    On AIE2P, the intrinsic `MacConfBFP576ACC2048` performs:
    - Input: 64×int8 data + 8×int8 shared exponents per operand = 576 bits each
    - Accumulator: 64×int32 = 2048 bits
    - Throughput: 64 MACs/cycle (vs 32 for native BF16 mmul)
    - Logical register width: I1024 for inputs, ACC2048 for acc

    This gives 2× throughput vs native BF16 for the same die area.
    """
    # Intrinsic name (from XLLVMAIE2IntrOps.td line 152)
    intrinsic: str = "MacConfBFP576ACC2048"

    # Data layout
    data_lanes: int = 64          # 64 elements per operand
    data_bits: int = 8            # int8 mantissa
    exponent_lanes: int = 8       # 8 shared exponents
    exponent_bits: int = 8        # 8-bit exponent per block
    exponent_block_size: int = 8  # 1 exponent shared per 8 data elements

    # Matrix dimensions for one intrinsic call
    mmul_m: int = 8   # M dimension (rows of C produced)
    mmul_k: int = 8   # K dimension (inner product width)
    mmul_n: int = 8   # N dimension (columns of C produced)

    # MACs per intrinsic: 8×8×8 = 512
    @property
    def macs(self) -> int:
        return self.mmul_m * self.mmul_k * self.mmul_n

    # Input bit-width: (64 data + 8 exp) × 8 bits = 576 bits each
    @property
    def input_bits(self) -> int:
        return (self.data_lanes + self.exponent_lanes) * self.data_bits

    # Accumulator bit-width: 64 × 32 = 2048 bits
    @property
    def acc_bits(self) -> int:
        return self.data_lanes * 32  # int32 accumulator


@dataclass
class BF16Config:
    """Native BF16 mmul configuration on AIE2P.

    AIE2P widens the N dimension to 8 (vs AIE2's 4):
    - MacConfBF16I1024ACC2048: 64×BF16 × 64×BF16 → 64×F32
    - Micro-kernel: 4×8×8 (vs AIE2's 4×8×4)

    Throughput: 32 MACs/cycle (half of BFP16)
    """
    mmul_m: int = 4
    mmul_k: int = 8
    mmul_n: int = 8   # AIE2P extension: 2× wider N vs AIE2

    @property
    def macs(self) -> int:
        return self.mmul_m * self.mmul_k * self.mmul_n  # 256

    @property
    def intrinsic(self) -> str:
        return "MacConfBF16I1024ACC2048"


@dataclass
class INT8Config:
    """INT8 mmul configuration on AIE2P.

    AIE2P widens M to 8 (vs AIE2's 4):
    - Micro-kernel: 8×8×8
    - Throughput: 64 MACs/cycle
    """
    mmul_m: int = 8
    mmul_k: int = 8
    mmul_n: int = 8

    @property
    def macs(self) -> int:
        return self.mmul_m * self.mmul_k * self.mmul_n  # 512


# =============================================================================
# GEMM micro-kernel template
# =============================================================================

@dataclass
class GemmMicroKernel:
    """Describes a GEMM micro-kernel schedule for AIE2P.

    The inner loop structure for BFP16 GEMM (C += A × B):
    ```
    // Load A tile [8×8 bf16 → 64×int8 data + 8×int8 exp] from L1
    // Load B tile [8×8 bf16 → 64×int8 data + 8×int8 exp] from L1
    // Convert to BFP16 format (insert exponents)
    // Transpose B (shuffle for mmul input layout)
    // mmul: C[8×8 acc] += A[8×8 bfp16] × B[8×8 bfp16]
    //       MacConfBFP576ACC2048: 512 MACs in 1 instruction
    // After all K iterations: pack C to bfp16ebs8, store to L1
    ```

    With 4-accumulator interleaving for II=1:
    ```
    acc0 += A[0]×B[0], acc1 += A[1]×B[1],
    acc2 += A[2]×B[2], acc3 += A[3]×B[3]
    ```
    This hides the 8-cycle mmul latency by interleaving 4 independent
    accumulator chains. Each chain starts a new mmul every cycle but
    doesn't need its result for 8 cycles → II=1.
    """
    # Micro-kernel dimensions (configurable per dtype)
    m_micro: int = 8   # rows per micro-kernel
    k_micro: int = 8   # inner dim per micro-kernel
    n_micro: int = 8   # cols per micro-kernel

    # Number of interleaved accumulators for II=1
    interleave_factor: int = 4  # 4 independent acc chains to hide 8-cycle mmul latency

    # Estimated cycle counts (from public latency tables)
    cycles_load_a: int = 7      # VLDA
    cycles_load_b: int = 7      # VLDB (can co-issue with VLDA if different banks)
    cycles_convert_a: int = 2   # vconvert: bf16 → bfp16
    cycles_convert_b: int = 2   # vconvert + vshuffle (transpose)
    cycles_mmul: int = 8        # MacConfBFP576ACC2048 pipeline depth
    cycles_pack: int = 2        # vpack: accfloat → bfp16ebs8
    cycles_store: int = 4       # VST

    @property
    def macs_per_ukernel(self) -> int:
        """MAC operations per micro-kernel invocation."""
        return self.m_micro * self.k_micro * self.n_micro  # 8×8×8 = 512

    @property
    def ops_per_ukernel(self) -> int:
        """Flop count (multiply + add = 2 ops per MAC)."""
        return self.macs_per_ukernel * 2  # 1024

    @property
    def compute_cycles_per_iter(self) -> int:
        """Cycles of compute per micro-kernel iteration (loads overlapped)."""
        # With double buffering, vector loads overlap with compute
        # The critical compute path: convert_A + convert_B + mmul = 2+2+8 = 12
        # But with 4-accumulator interleaving, mmuls overlap → effective 1/mmul
        return max(
            self.cycles_mmul // self.interleave_factor,  # Amortized mmul: 8/4 = 2
            self.cycles_convert_a + self.cycles_convert_b,  # Conversions: 2+2 = 4
        )  # = max(2, 4) = 4 (conversion-bound unless more convert units)

    @property
    def load_cycles_per_iter(self) -> int:
        """Cycles for loads per iteration (can overlap with compute)."""
        # 2 loads of 128 bytes each (8×8×bf16) at 512 bits/cycle = 2 cycles
        # But load latency is 7 cycles → need to issue loads 7 cycles ahead
        return 2  # 128 bytes × 2 / 64 bytes per cycle

    @property
    def ii_optimal(self) -> int:
        """Optimal Initiation Interval for this micro-kernel.

        With 4-accumulator interleaving:
        - ResMII: 1 VMAC/cycle = 1 (only 1 vector slot needed per iter)
        - RecMII: mmul latency / interleave = 8/4 = 2 (but pipelined to 1)
        - II=1 is achievable with perfect scheduling
        """
        return 1

    @property
    def peak_gflops_per_tile(self) -> float:
        """Theoretical peak GFLOPS per tile for this micro-kernel at II=1."""
        ops_per_iter = self.ops_per_ukernel
        return (ops_per_iter * AIE2P_CLOCK_MHZ * 1e6) / (self.ii_optimal * 1e9)

    def estimate_cycles(self, m_tile: int, k_tile: int, n_tile: int,
                        k_global: int, double_buffered: bool = True,
                        ii: int | None = None) -> int:
        """Estimate cycles for a (m_tile, k_global, n_tile) GEMM tile.

        Args:
            m_tile: Rows per tile
            k_tile: K dimension per step
            n_tile: Cols per tile
            k_global: Total K dimension
            double_buffered: Whether DMA overlaps with compute
            ii: Actual II (default: optimal II)
        """
        if ii is None:
            ii = self.ii_optimal

        m_iters = m_tile // self.m_micro
        n_iters = n_tile // self.n_micro
        k_steps = k_global // k_tile

        # Total micro-kernel iterations per tile
        total_ukernel_iters = m_iters * n_iters * k_steps

        # Steady state: total_ukernel_iters * II cycles
        # Prologue: (interleave_factor - 1) * II
        # Epilogue: (interleave_factor - 1) * II
        stage_count = max(self.interleave_factor, 1)
        steady_state = total_ukernel_iters * ii
        prologue = (stage_count - 1) * ii
        epilogue = (stage_count - 1) * ii

        # Pack/store overhead per C tile (once after all K steps)
        pack_store_cycles = (self.cycles_pack + self.cycles_store) * m_iters * n_iters

        return prologue + steady_state + epilogue + pack_store_cycles


# =============================================================================
# AIE2P Machine Model (corrected v2)
# =============================================================================

class AIE2PMachineModel:
    """Complete AIE2P tile model for scheduling and performance estimation.

    Tracks:
    - Register pressure (vector 12, accumulator 9, scalar 32)
    - Memory bank allocation (4 banks × 16 KB)
    - VLIW 7-way issue constraints
    - BFP16 micro-kernel performance
    - Software pipelining state
    """

    def __init__(self):
        self.regs = RegisterPressure()
        self.banks: dict[int, int] = {i: 0 for i in range(AIE2P_L1_BANKS)}
        self.latency = InstructionLatency()
        self.micro_kernel = GemmMicroKernel()
        self.bfp16 = BFP16Config()
        self.bf16 = BF16Config()
        self.int8_config = INT8Config()

    # -- Memory bank allocation (GAMA-style) --

    def allocate_buffer(self, name: str, size_bytes: int,
                        alignment: int = 64,
                        preferred_bank: int | None = None) -> tuple[int, int] | None:
        """Allocate a buffer in L1 memory with bank-aware placement.

        GAMA-style allocation: distributes A and B buffers to different banks
        to enable conflict-free dual-load (VLDA + VLDB same cycle).

        Args:
            name: Buffer identifier
            size_bytes: Size in bytes
            alignment: Address alignment (default 64-byte)
            preferred_bank: Preferred bank assignment

        Returns:
            (offset, bank_id) or None if allocation fails
        """
        # Try preferred bank first
        if preferred_bank is not None:
            if self.banks[preferred_bank] + size_bytes <= AIE2P_L1_BANK_BYTES:
                offset = self.banks[preferred_bank]
                self.banks[preferred_bank] += size_bytes
                return (offset, preferred_bank)

        # Greedy: assign to bank with most free space
        best_bank = min(
            range(AIE2P_L1_BANKS),
            key=lambda b: self.banks[b],
        )
        if self.banks[best_bank] + size_bytes <= AIE2P_L1_BANK_BYTES:
            offset = self.banks[best_bank]
            self.banks[best_bank] += size_bytes
            return (offset, best_bank)

        return None  # Out of L1 memory

    def allocate_gemm_buffers(self, m_tile: int, k_tile: int, n_tile: int,
                               dtype_in_bytes: int = 2,
                               dtype_out_bytes: int = 4) -> dict:
        """Allocate A, B, C buffers with conflict-free bank assignment.

        Places A and B in different banks to allow dual-load.
        """
        self.banks = {i: 0 for i in range(AIE2P_L1_BANKS)}  # Reset

        a_size = m_tile * k_tile * dtype_in_bytes
        b_size = k_tile * n_tile * dtype_in_bytes
        c_size = m_tile * n_tile * dtype_out_bytes

        result = {}
        result['A'] = self.allocate_buffer('A', a_size, preferred_bank=0)
        result['B'] = self.allocate_buffer('B', b_size, preferred_bank=1)
        result['C'] = self.allocate_buffer('C', c_size, preferred_bank=2)

        return result

    def check_bank_conflict(self, load_0_bank: int, load_1_bank: int) -> bool:
        """Check if two simultaneous loads conflict on the same bank."""
        return load_0_bank == load_1_bank

    # -- VLIW scheduling constraints --

    def can_co_issue_loads(self, bank_a: int, bank_b: int) -> bool:
        """Check if two vector loads (VLDA + VLDB) can co-issue.

        From section-4c: loads must come from different L1 memory banks
        or else a bank conflict will occur.
        """
        return bank_a != bank_b

    def check_hazard(self, producing_bundle: 'AIE2PVLIWBundle',
                     consuming_bundle: 'AIE2PVLIWBundle',
                     min_cycle_gap: int) -> list[str]:
        """Check for RAW/WAW/WAR hazards between two bundles."""
        hazards = []
        # Simplified — real implementation would track register def-use chains
        # and compare cycle distance to instruction latencies
        return hazards

    # -- Performance estimation --

    def estimate_gemm_gflops(self, m_tile: int, k_tile: int, n_tile: int,
                             k_global: int, n_tiles: int = 1,
                             double_buffered: bool = True,
                             bank_conflict_rate: float = 0.0,
                             ii: int | None = None) -> float:
        """Estimate GFLOPS for a GEMM configuration.

        Args:
            m_tile, k_tile, n_tile: Per-tile dimensions
            k_global: Total K dimension
            n_tiles: Number of AIE tiles
            double_buffered: Whether DMA overlaps compute
            bank_conflict_rate: Fraction of loads with bank conflicts (0-1)
            ii: Target initiation interval (None = optimal)

        Returns estimated GFLOPS.
        """
        total_macs = m_tile * k_global * n_tile * n_tiles
        total_ops = total_macs * 2

        cycles = self.micro_kernel.estimate_cycles(
            m_tile, k_tile, n_tile, k_global, double_buffered, ii,
        )

        # Bank conflict penalty: 1 cycle per conflict
        bank_conflict_cycles = int(
            cycles * bank_conflict_rate * self.latency.bank_conflict
        )
        total_cycles = cycles + bank_conflict_cycles

        time_s = total_cycles / (AIE2P_CLOCK_MHZ * 1e6)
        return total_ops / (time_s * 1e9)

    # -- Software pipelining state (from aie_kernel_utils.h pragmas) --

    def get_pipeline_pragmas(self, ii: int, min_iterations: int,
                              unroll_factor: int = 4) -> dict:
        """Generate software pipelining pragmas for Peano/Clang.

        Mirrors the macros in aie_kernel_utils.h:
        - AIE_PREPARE_FOR_PIPELINING
        - AIE_LOOP_MIN_ITERATION_COUNT(N)
        - AIE_TRY_INITIATION_INTERVAL(X)
        - AIE_LOOP_UNROLL(X)
        """
        return {
            "prepare_for_pipelining": "_Pragma(\"clang loop pipeline(enable)\")",
            "min_iteration_count": f"_Pragma(\"clang loop min_iteration_count({min_iterations})\")",
            "try_initiation_interval": f"_Pragma(\"clang loop pipeline_initiation_interval({ii})\")",
            "loop_unroll": f"_Pragma(\"clang loop unroll_count({unroll_factor})\")",
            "loop_unroll_full": "_Pragma(\"clang loop unroll(full)\")",
            "disable_pipelining": "_Pragma(\"clang loop pipeline(disable)\")",
        }

    def print_model_summary(self):
        """Print a summary of the AIE2P machine model."""
        mk = self.micro_kernel
        bfp = self.bfp16
        print("=" * 65)
        print("AIE2P Machine Model v2 (corrected — from public sources)")
        print("=" * 65)
        print(f"  Clock:                {AIE2P_CLOCK_MHZ} MHz")
        print(f"  L1 Data:              {AIE2P_L1_DATA_BYTES // 1024} KB "
              f"({AIE2P_L1_BANKS} banks × {AIE2P_L1_BANK_BYTES // 1024} KB)")
        print(f"  Program Memory:       {AIE2P_PROGRAM_MEMORY_BYTES // 1024} KB")
        print(f"  Vector Width:         {AIE2P_VECTOR_WIDTH_BITS} bits "
              f"(logical: 512/1024/2048)")
        print(f"  Vector Regs:          {AIE2P_VECTOR_REGS} (v0-v11)")
        print(f"  Accumulator Regs:     {AIE2P_ACCUMULATOR_REGS} (a0-a8)")
        print(f"  Scalar Regs:          {AIE2P_SCALAR_REGS} (32-bit)")
        print(f"  VLIW Slots:           7-way")
        print(f"  Load BW:              {AIE2P_LOAD_BANDWIDTH_BITS_PER_CYCLE} bits/cycle "
              f"(2×256b)")
        print(f"  Store BW:             {AIE2P_STORE_BANDWIDTH_BITS_PER_CYCLE} bits/cycle "
              f"(1×256b)")
        print()
        print(f"  BFP16 μkernel:        {bfp.mmul_m}×{bfp.mmul_k}×{bfp.mmul_n}")
        print(f"  BFP16 MACs/insn:      {bfp.macs} (2× vs BF16)")
        print(f"  BFP16 Intrinsic:      {bfp.intrinsic}")
        print(f"  BFP16 Input bits:     {bfp.input_bits} (data+exp)")
        print(f"  BFP16 Acc bits:        {bfp.acc_bits}")
        print(f"  BF16 μkernel:         {self.bf16.mmul_m}×{self.bf16.mmul_k}×{self.bf16.mmul_n}")
        print(f"  BF16 MACs/insn:       {self.bf16.macs}")
        print()
        print(f"  Peak GFLOPS/tile:     {mk.peak_gflops_per_tile:.1f} (II={mk.ii_optimal})")
        print(f"  Peak GFLOPS/32:       {mk.peak_gflops_per_tile * 32:.1f}")
        print(f"  Chess reference:      ~31200 GFLOPS (31.2 TFLOPS)")
        ratio = mk.peak_gflops_per_tile * 32 / 31200 * 100
        print(f"  Model vs Chess:       {ratio:.1f}% of Chess")
        print(f"  Interleave factor:    {mk.interleave_factor} (hides "
              f"{mk.cycles_mmul}-cycle mmul latency)")
        print("=" * 65)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    model = AIE2PMachineModel()
    model.print_model_summary()

    # Estimate for whole_array.py configuration: 64×64×32 tiles, 32 tiles, K=4096
    gflops_32tile = model.estimate_gemm_gflops(
        m_tile=64, k_tile=64, n_tile=32, k_global=4096, n_tiles=32,
        double_buffered=True, bank_conflict_rate=0.0, ii=1,
    )
    print(f"\nEstimated 32-tile (64×4096×32 each, 1024×4096×1024 total, II=1):")
    print(f"  {gflops_32tile:.1f} GFLOPS ({gflops_32tile/1000:.2f} TFLOPS)")

    # With realistic II=16 (Peano generated)
    gflops_32tile_ii16 = model.estimate_gemm_gflops(
        m_tile=64, k_tile=64, n_tile=32, k_global=4096, n_tiles=32,
        double_buffered=True, bank_conflict_rate=0.1, ii=16,
    )
    print(f"\nEstimated 32-tile (II=16, 10% bank conflicts):")
    print(f"  {gflops_32tile_ii16:.1f} GFLOPS ({gflops_32tile_ii16/1000:.2f} TFLOPS)")
    print(f"\nActual (whole_array.py, 32 tiles): 3011 GFLOPS (3.01 TFLOPS)")

    # Buffer allocation test
    print(f"\n--- L1 Buffer Allocation (64×64×32, BF16→FP32) ---")
    alloc = model.allocate_gemm_buffers(64, 64, 32)
    for name, (offset, bank) in alloc.items():
        buf_size = { 'A': 64*64*2, 'B': 64*32*2, 'C': 64*32*4 }[name]
        print(f"  {name}: offset={offset:5d} bank={bank} size={buf_size}B")
