#!/usr/bin/env python3
"""Phase 5: BFP16 operand pre-packing for AIE2P mmul-ready format.

Pre-packing rearranges matrices from standard row-major into the blocked
tile-major format that aie::mmul<8,8,8> expects:

  A[M×K] → (M/8)×(K/8) tiles, each 8×8 row-major, contiguous in memory.
           Tiles are in row-major order: A_tile[0,0], A_tile[0,1], ...

  B[K×N] → (K/8)×(N/8) tiles, each 8×8 row-major, contiguous in memory.
           Tiles are in COLUMN-major tile order: B_tile[0,0], B_tile[1,0], ...
           This makes the K-reduction inner loop purely sequential.

Both packed layouts produce the SAME total element count as the source;
the rearrangement is a permutation that groups elements into 64-element
micro-tiles and orders those tiles for optimal streaming through the
AIE2P VLIW pipeline.

Usage (standalone, no IRON dependency):
    python3 experiments/pack_bfp16.py
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Tile dimensions (hard-coded for BFP16 8×8×8 micro-kernel)
# ---------------------------------------------------------------------------
R = 8  # micro-kernel M dimension
S = 8  # micro-kernel K dimension (shared)
T = 8  # micro-kernel N dimension
TILE_ELEMS = R * S  # 64 elements per 8×8 tile
TILE_BYTES_BF16 = TILE_ELEMS * 2   # 128 bytes per tile
TILE_BYTES_F32  = TILE_ELEMS * 4   # 256 bytes per accumulator tile


# =============================================================================
# Packing: row-major → mmul-ready blocked format
# =============================================================================

def pack_A_bfp16(A: np.ndarray, M: int, K: int) -> np.ndarray:
    """Pack A from row-major [M×K] into (M/8)×(K/8) blocked format.

    Packed layout::
        A_packed[tile_offset(z, ki)] = tile_data(z, ki)[r*8 + c]
        where tile_offset(z, ki) = (z * K_tiles + ki) * 64,
        and tile_data(z, ki) = A[z*8+r, ki*8+c].

    The inner K-reduction loop advances ``ki`` → purely sequential access.
    """
    assert M % R == 0 and K % S == 0, f"M={M},K={K} must be multiples of {R}x{S}"
    M_tiles = M // R
    K_tiles = K // S
    packed = np.empty(M * K, dtype=A.dtype)

    for z in range(M_tiles):          # row tile (M dimension)
        for ki in range(K_tiles):     # col tile (K dimension)
            tile_offset = (z * K_tiles + ki) * TILE_ELEMS
            for r in range(R):
                for c in range(S):
                    src = (z * R + r) * K + (ki * S + c)
                    dst = tile_offset + r * S + c
                    packed[dst] = A.flat[src]
    return packed


def pack_B_bfp16_colmaj(B: np.ndarray, K: int, N: int) -> np.ndarray:
    """Pack B from row-major [K×N] into column-major tile-blocked format.

    Packed layout::
        B_packed[tile_offset(nj, ki)] = tile_data(ki, nj)[r*8 + c]
        where tile_offset(nj, ki) = (nj * K_tiles + ki) * 64,
        and tile_data(ki, nj) = B[ki*8+r, nj*8+c].

    Tiles are column-major in (nj, ki) space → inner K loop is sequential:
        for nj fixed, B_tile[nj, ki+1] is 64 elements after B_tile[nj, ki].
    """
    assert K % S == 0 and N % T == 0, f"K={K},N={N} must be multiples of {S}x{T}"
    K_tiles = K // S
    N_tiles = N // T
    packed = np.empty(K * N, dtype=B.dtype)

    for nj in range(N_tiles):         # column tile (N dimension, OUTER)
        for ki in range(K_tiles):     # row tile (K dimension, INNER)
            tile_offset = (nj * K_tiles + ki) * TILE_ELEMS
            for r in range(S):        # S=8 rows within tile
                for c in range(T):    # T=8 cols within tile
                    src = (ki * S + r) * N + (nj * T + c)
                    dst = tile_offset + r * T + c
                    packed[dst] = B.flat[src]
    return packed


# =============================================================================
# Unpacking: mmul-ready format → row-major (for verification)
# =============================================================================

def unpack_C_f32(C_packed: np.ndarray, M: int, N: int) -> np.ndarray:
    """Unpack C from blocked format back to row-major [M×N].

    C is stored in the same (M/8)×(N/8) blocked format (row-major tile order,
    row-major within each 8×8 tile).
    """
    assert M % R == 0 and N % T == 0
    M_tiles = M // R
    N_tiles = N // T
    C = np.empty((M, N), dtype=C_packed.dtype)

    for z in range(M_tiles):
        for nj in range(N_tiles):
            tile_offset = (z * N_tiles + nj) * TILE_ELEMS
            for r in range(R):
                for c in range(T):
                    src = tile_offset + r * T + c
                    dst = (z * R + r, nj * T + c)
                    C[dst] = C_packed.flat[src]
    return C


# =============================================================================
# Convenience: pack both operands for a GEMM problem
# =============================================================================

def pack_gemm_operands(
    A: np.ndarray, B: np.ndarray, M: int, K: int, N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack A and B for a BFP16 GEMM C[M×N] += A[M×K] × B[K×N].

    Returns (A_packed, B_packed) as flat 1-D arrays.
    """
    A_packed = pack_A_bfp16(A, M, K)
    B_packed = pack_B_bfp16_colmaj(B, K, N)
    return A_packed, B_packed


# =============================================================================
# Validation
# =============================================================================

def _verify():
    """Verify packing correctness by computing a small GEMM."""
    M, K, N = 32, 64, 16
    rng = np.random.default_rng(42)
    A = (rng.random((M, K)) * 2 - 1).astype(np.float32)
    B = (rng.random((K, N)) * 2 - 1).astype(np.float32)

    A_packed = pack_A_bfp16(A, M, K)
    B_packed = pack_B_bfp16_colmaj(B, K, N)

    # Simulate what the packed kernel does: for each (z,j) tile, reduce over K
    C_packed = np.zeros(M * N, dtype=np.float32)
    M_tiles, K_tiles = M // R, K // S
    N_tiles = N // T

    for z in range(M_tiles):
        for nj in range(N_tiles):
            # Load C tile (accumulator)
            c_offset = (z * N_tiles + nj) * TILE_ELEMS
            C_tile = C_packed[c_offset:c_offset + TILE_ELEMS].reshape(R, T)

            for ki in range(K_tiles):
                a_offset = (z * K_tiles + ki) * TILE_ELEMS
                b_offset = (nj * K_tiles + ki) * TILE_ELEMS
                A_tile = A_packed[a_offset:a_offset + TILE_ELEMS].reshape(R, S)
                B_tile = B_packed[b_offset:b_offset + TILE_ELEMS].reshape(S, T)
                C_tile += A_tile @ B_tile

            C_packed[c_offset:c_offset + TILE_ELEMS] = C_tile.ravel()

    C_got = unpack_C_f32(C_packed, M, N)
    C_expected = A @ B

    max_err = np.max(np.abs(C_got - C_expected))
    print(f"  M={M} K={K} N={N}")
    print(f"  Packed sizes: A={A_packed.size}, B={B_packed.size}, C={C_packed.size}")
    print(f"  Max error: {max_err:.2e}")
    if max_err < 1e-5:
        print("  ✓ Packing verified — results match numpy reference")
    else:
        print("  ✗ Packing ERROR — results differ!")
        raise AssertionError(f"Packing verification failed: max_err={max_err}")

    # Check tile access pattern
    print(f"\n  Access pattern analysis:")
    print(f"    A: {M_tiles}×{K_tiles} tiles, row-major tile order")
    print(f"       Inner K loop: Δoffset = {TILE_ELEMS} elems = {TILE_BYTES_BF16} B (sequential ✓)")
    print(f"    B: {K_tiles}×{N_tiles} tiles, column-major tile order")
    print(f"       Inner K loop: Δoffset = {TILE_ELEMS} elems = {TILE_BYTES_BF16} B (sequential ✓)")
    print(f"    C: {M_tiles}×{N_tiles} tiles, row-major tile order")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5: BFP16 Operand Pre-Packing — Verification")
    print("=" * 60)
    _verify()
    print("\n" + "=" * 60)
    print("Packing library ready.")
    print("=" * 60)
