"""GAMA-style bank-conflict-aware L1 buffer allocator for AIE2P.

Implements bank-aware buffer allocation that guarantees conflict-free
dual-vector-load (VLDA + VLDB) per cycle. Based on the GAMA layout rules
referenced in the mlir-aie bank-aware allocation pass.

Key principles:
1. A and B operand buffers MUST live in different memory banks for
   dual-load (VLDA + VLDB) in the same cycle
2. 2D access strides that are multiples of 4 banks cause pathological
   conflicts — pad dimensions by +1 element to break alignment
3. Double-buffered tiles need even/odd buffer pairs in separate banks
4. Accumulator C benefits from a dedicated bank to avoid store contention

Sources:
- mlir-aie: AIEPasses.td (bank-aware allocation scheme)
- programming_guide/section-4/section-4c/README.md
- AMD AIE-ML Architecture Manual (bank structure)
- Llosa et al., "Lifetime-sensitive modulo scheduling" (PLDI 96)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    from .machine_model import (
        AIE2P_L1_DATA_BYTES,
        AIE2P_L1_BANKS,
        AIE2P_L1_BANK_BYTES,
        AIE2P_LOAD_BANDWIDTH_BITS_PER_CYCLE,
    )
except ImportError:
    from machine_model import (
        AIE2P_L1_DATA_BYTES,
        AIE2P_L1_BANKS,
        AIE2P_L1_BANK_BYTES,
        AIE2P_LOAD_BANDWIDTH_BITS_PER_CYCLE,
    )


# =============================================================================
# Bank allocation primitives
# =============================================================================

class BufferClass(Enum):
    """Buffer classification for bank assignment."""
    A_OPERAND = "A"       # Input A (M×K) — needs dedicated bank for VLDA
    B_OPERAND = "B"       # Input B (K×N) — needs dedicated bank for VLDB
    C_ACCUMULATOR = "C"   # Output C (M×N) — separate bank preferred
    DOUBLE_BUFFER_A = "A2"  # Second A buffer (double buffering)
    DOUBLE_BUFFER_B = "B2"  # Second B buffer
    STACK = "stack"       # Stack / scratch


@dataclass
class BufferPlacement:
    """Placement result for a single buffer."""
    name: str
    buffer_class: BufferClass
    size_bytes: int
    bank: int
    offset: int                       # Byte offset within bank
    alignment: int = 64               # Address alignment

    @property
    def end_offset(self) -> int:
        return self.offset + self.size_bytes

    @property
    def fits(self) -> bool:
        return self.end_offset <= AIE2P_L1_BANK_BYTES


@dataclass
class BankLayout:
    """Complete L1 memory layout across 4 banks."""
    placements: list[BufferPlacement] = field(default_factory=list)
    bank_usage: dict[int, int] = field(default_factory=dict)  # bank → bytes used

    @property
    def total_bytes(self) -> int:
        return sum(self.bank_usage.values())

    @property
    def fits_in_l1(self) -> bool:
        return (self.total_bytes <= AIE2P_L1_DATA_BYTES and
                all(u <= AIE2P_L1_BANK_BYTES for u in self.bank_usage.values()))

    def check_dual_load_conflict(self) -> bool:
        """True if A and B are on different banks (no conflict for dual load)."""
        a_bank = None
        b_bank = None
        for p in self.placements:
            if p.buffer_class == BufferClass.A_OPERAND:
                a_bank = p.bank
            elif p.buffer_class == BufferClass.B_OPERAND:
                b_bank = p.bank
        if a_bank is not None and b_bank is not None:
            return a_bank == b_bank  # True = conflict
        return False  # No conflict if we can't find both


class BankAllocator:
    """GAMA-style bank-aware L1 buffer allocator for AIE2P GEMM.

    Guarantees conflict-free dual-vector-load by placing A and B
    operand buffers in different memory banks.

    Algorithm:
    1. Assign A to bank 0 (lowest address)
    2. Assign B to bank 1 (different from A)
    3. Assign C to bank 2 (different from A, B)
    4. For double buffering, A2 goes to bank 3, B2 goes to bank 0
       (rotating assignment with offset)
    5. Check 2D stride alignment and suggest padding if needed
    """

    def __init__(self, l1_bytes: int = AIE2P_L1_DATA_BYTES,
                 num_banks: int = AIE2P_L1_BANKS,
                 bank_bytes: int = AIE2P_L1_BANK_BYTES):
        self.l1_bytes = l1_bytes
        self.num_banks = num_banks
        self.bank_bytes = bank_bytes
        # Track next free offset in each bank
        self._bank_next: dict[int, int] = {b: 0 for b in range(num_banks)}

    def reset(self):
        """Reset bank allocation state."""
        self._bank_next = {b: 0 for b in range(self.num_banks)}

    def allocate(self, name: str, size_bytes: int,
                 buffer_class: BufferClass,
                 alignment: int = 64) -> BufferPlacement | None:
        """Allocate a buffer with bank-aware placement.

        Args:
            name: Buffer identifier
            size_bytes: Size in bytes
            buffer_class: Type of buffer (determines bank preference)
            alignment: Address alignment

        Returns:
            BufferPlacement or None if allocation fails
        """
        # Determine preferred bank based on buffer class
        preferred_bank = self._preferred_bank(buffer_class)

        # Align size
        aligned_size = ((size_bytes + alignment - 1) // alignment) * alignment

        # Try preferred bank first
        if preferred_bank is not None:
            offset = self._bank_next[preferred_bank]
            if offset + aligned_size <= self.bank_bytes:
                self._bank_next[preferred_bank] = offset + aligned_size
                return BufferPlacement(
                    name=name,
                    buffer_class=buffer_class,
                    size_bytes=aligned_size,
                    bank=preferred_bank,
                    offset=offset,
                    alignment=alignment,
                )

        # Find any bank with enough space
        for bank in sorted(range(self.num_banks),
                          key=lambda b: self._bank_next[b]):
            offset = self._bank_next[bank]
            if offset + aligned_size <= self.bank_bytes:
                self._bank_next[bank] = offset + aligned_size
                return BufferPlacement(
                    name=name,
                    buffer_class=buffer_class,
                    size_bytes=aligned_size,
                    bank=bank,
                    offset=offset,
                    alignment=alignment,
                )

        return None  # Out of memory

    def _preferred_bank(self, buffer_class: BufferClass) -> int | None:
        """Preferred bank for each buffer type (GAMA layout rules).

        A → bank 0: Lowest address, first load unit (VLDA)
        B → bank 1: Different from A, second load unit (VLDB)
        C → bank 2: Separate from both A and B (avoid store contention)
        Double buffers → interleave across remaining bank
        """
        if buffer_class == BufferClass.A_OPERAND:
            return 0
        elif buffer_class == BufferClass.B_OPERAND:
            return 1
        elif buffer_class == BufferClass.C_ACCUMULATOR:
            return 2
        elif buffer_class == BufferClass.DOUBLE_BUFFER_A:
            return 3  # Rotate to bank 3 for second A buffer
        elif buffer_class == BufferClass.DOUBLE_BUFFER_B:
            return 0  # Share bank 0 (in non-overlapping region) or use 1
        elif buffer_class == BufferClass.STACK:
            return 3  # Stack at end of bank 3
        return None  # Any bank

    # -- GEMM layout generation --

    def layout_gemm(self, m_tile: int, k_tile: int, n_tile: int,
                    double_buffered: bool = True,
                    dtype_in_bytes: int = 2,
                    dtype_out_bytes: int = 4) -> BankLayout:
        """Create a complete L1 buffer layout for GEMM.

        Args:
            m_tile, k_tile, n_tile: Tile dimensions
            double_buffered: Whether to allocate second copies for DMA pipelining
            dtype_in_bytes: Input element size (2 for bf16)
            dtype_out_bytes: Output element size (4 for fp32)

        Returns:
            BankLayout with all buffer placements
        """
        layout = BankLayout()

        a_size = m_tile * k_tile * dtype_in_bytes   # e.g., 64×64×2 = 8192
        b_size = k_tile * n_tile * dtype_in_bytes   # e.g., 64×32×2 = 4096
        c_size = m_tile * n_tile * dtype_out_bytes  # e.g., 64×32×4 = 8192

        # Primary buffers
        for cls, name, size in [
            (BufferClass.A_OPERAND, "A", a_size),
            (BufferClass.B_OPERAND, "B", b_size),
            (BufferClass.C_ACCUMULATOR, "C", c_size),
        ]:
            placement = self.allocate(name, size, cls)
            if placement is None:
                raise MemoryError(f"Cannot allocate {name} ({size}B) in L1")
            layout.placements.append(placement)

        # Double-buffer copies
        if double_buffered:
            a2 = self.allocate("A_db", a_size, BufferClass.DOUBLE_BUFFER_A)
            b2 = self.allocate("B_db", b_size, BufferClass.DOUBLE_BUFFER_B)
            if a2 is None or b2 is None:
                raise MemoryError(f"Cannot allocate double buffers in L1")
            layout.placements.extend([a2, b2])

        # Stack allocation (minimal — just for spills)
        stack_size = 256  # bytes
        stack = self.allocate("stack", stack_size, BufferClass.STACK)
        if stack:
            layout.placements.append(stack)

        # Update bank usage summary
        for p in layout.placements:
            layout.bank_usage[p.bank] = max(
                layout.bank_usage.get(p.bank, 0),
                p.end_offset,
            )

        return layout

    # -- 2D stride analysis --

    def analyze_2d_stride(self, m_tile: int, k_tile: int,
                           element_bytes: int = 2) -> dict:
        """Analyze 2D access stride for bank conflict patterns.

        When loading a column from an M×K matrix stored row-major:
        - Stride between consecutive elements = M elements × element_bytes
        - If stride_bytes % (num_banks × bank_interleave) == 0, every access
          hits the same bank → pathological conflict

        Fix: pad M dimension by +1 element to break alignment.
        """
        stride_bytes = m_tile * element_bytes
        bank_interleave = 4  # bytes (32-bit interleave on AIE2P)
        bank_cycle = self.num_banks * bank_interleave  # 16 bytes

        is_pathological = (stride_bytes % bank_cycle == 0)

        result = {
            "stride_bytes": stride_bytes,
            "bank_cycle_bytes": bank_cycle,
            "is_pathological": is_pathological,
            "recommendation": None,
            "padded_m": m_tile,
            "conflict_rate": 0.0,
        }

        if is_pathological:
            # Pad M by +1 to break alignment
            padded_m = m_tile + 1
            padded_stride = padded_m * element_bytes
            result["recommendation"] = (
                f"Pad M dimension from {m_tile} to {padded_m} "
                f"(stride {stride_bytes} → {padded_stride}) to break bank alignment"
            )
            result["padded_m"] = padded_m
            result["conflict_rate"] = 1.0  # 100% conflicts on column access
        else:
            # Compute actual conflict rate
            # Number of stride values that align with bank cycle
            stride_mod = stride_bytes % bank_cycle
            if stride_mod == 0:
                result["conflict_rate"] = 1.0
            elif stride_mod in (4, 8, 12):
                result["conflict_rate"] = 0.5  # Every other access hits same bank
            else:
                result["conflict_rate"] = 0.0  # No conflicts

        return result

    # -- Full GEMM optimization report --

    def optimize_gemm_layout(self, m_tile: int, k_tile: int, n_tile: int,
                             double_buffered: bool = True) -> dict:
        """Full GEMM buffer layout optimization with recommendations.

        Returns a detailed report including:
        - Per-bank buffer assignment
        - Bank conflict analysis
        - Size breakdown
        - Padding recommendations
        - Alternative tile size suggestions if current doesn't fit
        """
        self.reset()

        # Check if current config fits
        try:
            layout = self.layout_gemm(m_tile, k_tile, n_tile, double_buffered)
            fits = layout.fits_in_l1
        except MemoryError:
            fits = False
            layout = None

        # 2D stride analysis
        stride_a = self.analyze_2d_stride(m_tile, k_tile)
        stride_b = self.analyze_2d_stride(n_tile, k_tile, element_bytes=2)

        # Build report
        report = {
            "tile_dims": {"m": m_tile, "k": k_tile, "n": n_tile},
            "double_buffered": double_buffered,
            "fits_in_l1": fits,
            "l1_bytes_used": layout.total_bytes if layout else 0,
            "l1_bytes_total": self.l1_bytes,
            "bank_usage": layout.bank_usage if layout else {},
            "placements": [
                {
                    "name": p.name,
                    "class": p.buffer_class.value,
                    "bank": p.bank,
                    "offset": p.offset,
                    "size": p.size_bytes,
                }
                for p in (layout.placements if layout else [])
            ],
            "dual_load_safe": (
                not layout.check_dual_load_conflict() if layout else False
            ),
            "stride_A": stride_a,
            "stride_B": stride_b,
            "recommendations": [],
        }

        # Generate recommendations
        if not fits:
            report["recommendations"].append({
                "priority": 1,
                "type": "size_reduction",
                "message": f"Layout exceeds L1 ({self.l1_bytes}B). "
                           f"Reduce tile dimensions or disable double buffering.",
                "suggestion": "Try smaller tiles like 32×32×16 or disable double buffering",
            })

        if layout and layout.check_dual_load_conflict():
            report["recommendations"].append({
                "priority": 1,
                "type": "bank_conflict",
                "message": "A and B share a bank — dual-load will cause bank conflicts",
                "suggestion": "Relocate B to a different bank",
            })

        if stride_a["is_pathological"]:
            report["recommendations"].append({
                "priority": 2,
                "type": "stride_conflict_A",
                "message": stride_a["recommendation"],
                "improvement": "Up to 2× (eliminates 1-cycle stall per load)",
            })

        if stride_b["is_pathological"]:
            report["recommendations"].append({
                "priority": 2,
                "type": "stride_conflict_B",
                "message": stride_b["recommendation"],
                "improvement": "Up to 2× (eliminates 1-cycle stall per load)",
            })

        return report

    def suggest_alternative_tiles(self, m_tile: int, k_tile: int, n_tile: int,
                                  double_buffered: bool = True) -> list[dict]:
        """Suggest alternative tile sizes that fit in L1 and avoid bank conflicts.

        Searches nearby tile sizes (±8 in each dimension) for conflict-free layouts.
        """
        alternatives = []
        for m in range(max(8, m_tile - 16), min(128, m_tile + 24), 8):
            for k in range(max(8, k_tile - 16), min(128, k_tile + 24), 8):
                for n in range(max(8, n_tile - 16), min(64, n_tile + 24), 8):
                    if m == m_tile and k == k_tile and n == n_tile:
                        continue
                    self.reset()
                    try:
                        layout = self.layout_gemm(m, k, n, double_buffered)
                        if layout.fits_in_l1:
                            stride = self.analyze_2d_stride(m, k)
                            has_conflicts = layout.check_dual_load_conflict()
                            alternatives.append({
                                "m": m, "k": k, "n": n,
                                "l1_bytes": layout.total_bytes,
                                "bank_conflict_free": not has_conflicts,
                                "stride_conflict_free": not stride["is_pathological"],
                                "score": (0 if not has_conflicts else -10) +
                                         (0 if not stride["is_pathological"] else -5) +
                                         (m * k * n),  # Larger tiles = higher throughput
                            })
                    except MemoryError:
                        continue

        alternatives.sort(key=lambda a: a["score"], reverse=True)
        return alternatives[:10]


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    alloc = BankAllocator()

    print("=" * 65)
    print("GAMA-Style Bank-Aware L1 Buffer Allocator — AIE2P")
    print("=" * 65)
    print(f"  L1: {alloc.l1_bytes // 1024} KB, {alloc.num_banks} banks × "
          f"{alloc.bank_bytes // 1024} KB")

    # Test: 64×64×32 BF16 GEMM (double buffered)
    print(f"\n{'—'*65}")
    print("Layout: 64×64×32 BF16 GEMM (double buffered)")
    report = alloc.optimize_gemm_layout(64, 64, 32, double_buffered=True)
    print(f"  Fits in L1:  {report['fits_in_l1']}")
    print(f"  L1 used:     {report['l1_bytes_used']} / {report['l1_bytes_total']} bytes "
          f"({report['l1_bytes_used']/report['l1_bytes_total']:.0%})")
    print(f"  Dual load:   {'SAFE' if report['dual_load_safe'] else 'CONFLICT!'}")
    print(f"  Bank usage:  {report['bank_usage']}")
    for p in report["placements"]:
        print(f"    {p['name']:6s} class={p['class']:3s} bank={p['bank']} "
              f"offset={p['offset']:5d} size={p['size']:5d}B")

    # 2D stride analysis
    print(f"\n  2D Stride Analysis:")
    for dim, s in [("A (M×K)", report['stride_A']), ("B (N×K)", report['stride_B'])]:
        print(f"    {dim}: stride={s['stride_bytes']}B, "
              f"pathological={s['is_pathological']}, "
              f"conflict_rate={s['conflict_rate']:.0%}")

    for r in report["recommendations"]:
        print(f"  [!] [{r['type']}] {r['message']}")

    # Test: alternative tile search
    print(f"\n{'—'*65}")
    print("Alternative Tile Search (conflict-free, fits in L1):")
    alts = alloc.suggest_alternative_tiles(64, 64, 32, double_buffered=True)
    for a in alts[:5]:
        flags = []
        if a['bank_conflict_free']:
            flags.append('bank-safe')
        if a['stride_conflict_free']:
            flags.append('stride-safe')
        print(f"  {a['m']:3d}×{a['k']:3d}×{a['n']:2d}  "
              f"L1={a['l1_bytes']:5d}B  score={a['score']:5d}  "
              f"{' '.join(flags)}")

    # Test: single-buffered layout
    print(f"\n{'—'*65}")
    print("Layout: 64×64×32 BF16 GEMM (single buffered)")
    report_single = alloc.optimize_gemm_layout(64, 64, 32, double_buffered=False)
    print(f"  Fits in L1:  {report_single['fits_in_l1']}")
    print(f"  L1 used:     {report_single['l1_bytes_used']}B "
          f"({report_single['l1_bytes_used']/report_single['l1_bytes_total']:.0%})")

    # Test: pathological stride
    print(f"\n{'—'*65}")
    print("Pathological Stride Test: M=64 vs M=65")
    for m in [64, 65]:
        stride = alloc.analyze_2d_stride(m, 64)
        print(f"  M={m}: stride={stride['stride_bytes']}B, "
              f"pathological={stride['is_pathological']}")
        if stride['recommendation']:
            print(f"    → {stride['recommendation']}")
