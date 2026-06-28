//===- kernel_gemm_bfp16_packed.cc -----------------------------*- C++ -*-===//
//
// Phase 5: Pre-packed BFP16 GEMM micro-kernel for AIE2P.
//
// TWO MODES (controlled by -DB_COL_MAJ compile flag):
//
//   Mode 1 — Column-major B (-DB_COL_MAJ):
//     B is stored N×K column-major, streamed in column-major tile order.
//     Inner loop loads are SEQUENTIAL (no stride), but each loaded 8×8 tile
//     is column-major and needs aie::transpose(→row-major) before mmul.mac().
//     This matches the reference mm.cc b_row_maj=false path.
//
//   Mode 2 — CPU Pre-packed (no -DB_COL_MAJ):
//     B is pre-packed on CPU via pack_bfp16.py into column-major TILE order
//     with row-major WITHIN each 8×8 tile. No runtime transpose needed.
//     Purely sequential loads, data already in mmul-ready format.
//
// Both modes share the same A layout (row-major tile order, row-major within
// tile → always sequential A loads).
//
// Key property: ALL loads in the K-reduction inner loop are SEQUENTIAL.
//   for ki in 0..K/8:
//     A_tile = load(pA);  pA += 64;   // sequential
//     B_tile = load(pB);  pB += 64;   // sequential (no stride!)
//     C.mac(A_tile, B_tile);
//
// Based on mlir-aie/aie_kernels/aie2p/mm.cc (Apache 2.0 w/ LLVM exceptions).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

using namespace aie;

// =============================================================================
// Zero kernel — accumulator initialization between K-steps
// =============================================================================

template <int M, int N>
void zero_vectorized_f32(float *__restrict cOut) {
  constexpr int vectorSize = 16; // 16 floats = 512-bit vector

  const aie::accum<accfloat, vectorSize> acc =
      aie::zeros<accfloat, vectorSize>();

  for (int i = 0; i < M * N / vectorSize; i++) {
    aie::store_v(cOut + i * vectorSize, acc.template to_vector<float>());
  }
}

// =============================================================================
// 2×2 mmul kernel — PACKED LAYOUT
//
// All loads in the inner K-loop are SEQUENTIAL pointer increments.
// No strides, no transposes, no column-major B in DRAM.
//
// Template parameters:
//   rowA = M / 8   (number of 8-row tiles in M)
//   colA = K / 8   (number of 8-element tiles in K reduction)
//   colB = N / 8   (number of 8-col tiles in N)
//   r,s,t = 8,8,8  (BFP16 micro-kernel)
//
// Packed A layout (row-major tile order):
//   A_packed[((z * colA) + i) * 64 + (r*8 + c)]
//     = A[ (z*8 + r) * K_global + (i*8 + c) ]
//
// Packed B layout (column-major tile order):
//   B_packed[((j * colA) + i) * 64 + (r*8 + c)]
//     = B[ (i*8 + r) * N_global + (j*8 + c) ]
//
// This means B[:, j, i+1] is exactly 64 elements after B[:, j, i] in memory.
// =============================================================================

template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
[[clang::always_inline]]
static inline void matmul_vectorized_2x2_mmul_packed(
    const T_in *__restrict pA,
    const T_in *__restrict pB,
    T_out *__restrict pC)
{
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
  // For BFP16 emulation: r=8, s=8, t=8
  //   size_A = r*s = 64 elements (1024 bits = 2×512-bit vector regs)
  //   size_B = s*t = 64 elements (1024 bits = 2×512-bit vector regs)
  //   size_C = r*t = 64 elements (2048 bits = 4×512-bit acc regs)
  //   512 MACs per instruction

  // Outer loops over M and N tile positions
  for (unsigned z = 0; z < rowA; z += 2) {

    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2) {

      // ---- A base pointers: A is row-major in tile space ----
      // A[z, :] starts at offset (z * colA) * 64
      const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
      const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;

      // ---- B base pointers: B is COLUMN-MAJOR in tile space ----
      // B[:, j] starts at offset (j * colA) * 64
      // B[:, j+1] starts at offset ((j+1) * colA) * 64
      const T_in *__restrict pB1 = pB + (j * colA) * MMUL::size_B;
      const T_in *__restrict pB2 = pB + ((j + 1) * colA) * MMUL::size_B;

      // Load previous partial sums from C
      aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
      aie::vector<T_out, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
      aie::vector<T_out, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
      aie::vector<T_out, MMUL::size_C> acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

      MMUL C00(acc_C00);
      MMUL C01(acc_C01);
      MMUL C10(acc_C10);
      MMUL C11(acc_C11);

      // ==================================================================
      // K-reduction inner loop — ALL LOADS ARE SEQUENTIAL (II=1 target)
      //
      // B_COL_MAJ mode: B is column-major within each tile → transpose needed.
      // Pre-packed mode: B already row-major within each tile → no transpose.
      //
      // VLIW bundle (7-way) per iteration:
      //   Slot 0 (scalar):  loop counter update
      //   Slot 1 (move):    —
      //   Slot 2 (move):    —
      //   Slot 3 (VLDA):    A0 = load_v<64>(pA1); pA1 += 64;
      //   Slot 4 (VLDB):    B0 = load_v<64>(pB1); pB1 += 64;
      //   Slot 5 (VST):     —
      //   Slot 6 (VMAC):    C00.mac(A0, B0)
      //
      // 2nd cycle (interleaved):
      //   Slot 3 (VLDA):    A1 = load_v<64>(pA2); pA2 += 64;
      //   Slot 4 (VLDB):    B1 = load_v<64>(pB2); pB2 += 64;
      //   Slot 6 (VMAC):    C01.mac(A0, B1)
      //
      // With 4 accumulators interleaved (C00,C01,C10,C11),
      // the 8-cycle mmul pipeline depth is fully hidden.
      // ==================================================================
      _Pragma("clang loop min_iteration_count(8)")
#ifndef NO_II1
      _Pragma("clang loop pipeline_initiation_interval(1)")
#endif
      for (unsigned i = 0; i < colA; ++i) {
        // Load A tiles — SEQUENTIAL
        aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;

        // Load B tiles — SEQUENTIAL (no stride!)
#ifdef B_COL_MAJ
        // B is column-major within the 8×8 tile → transpose to row-major
        // (matches reference mm.cc b_row_maj=false path)
        aie::vector<T_in, MMUL::size_B> B0 =
            aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
        pB1 += MMUL::size_B;
        aie::vector<T_in, MMUL::size_B> B1 =
            aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
        pB2 += MMUL::size_B;
#else
        // Pre-packed: B already row-major within tile, no transpose needed
        aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;
#endif

        // MAC: 4 independent accumulators → hides 8-cycle mmul pipeline depth
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      // Store accumulated results
      aie::store_v(pC1, C00.template to_vector<T_out>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<T_out>());
      pC1 += MMUL::size_C;
      aie::store_v(pC2, C10.template to_vector<T_out>());
      pC2 += MMUL::size_C;
      aie::store_v(pC2, C11.template to_vector<T_out>());
      pC2 += MMUL::size_C;
    }
  }
}

// =============================================================================
// Typed entry wrappers
// =============================================================================

template <unsigned m, unsigned k, unsigned n>
[[clang::always_inline]]
static inline void matmul_vectorized_8x8x8_bf16_f32_packed(
    const bfloat16 *__restrict pA,
    const bfloat16 *__restrict pB,
    float *__restrict pC)
{
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0, "M must be divisible by 16 (2×8)");
  static_assert(k % s == 0,      "K must be divisible by 8");
  static_assert(n % (2 * t) == 0, "N must be divisible by 16 (2×8)");

  matmul_vectorized_2x2_mmul_packed<bfloat16, float,
      (m / r), (k / s), (n / t), r, s, t>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
[[clang::always_inline]]
static inline void matmul_vectorized_8x8x8_i8_i32_packed(
    const int8 *__restrict pA,
    const int8 *__restrict pB,
    int32 *__restrict pC)
{
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0, "M must be divisible by 16 (2×8)");
  static_assert(k % s == 0,      "K must be divisible by 8");
  static_assert(n % (2 * t) == 0, "N must be divisible by 16 (2×8)");

  matmul_vectorized_2x2_mmul_packed<int8, int32,
      (m / r), (k / s), (n / t), r, s, t>(pA, pB, pC);
}

// =============================================================================
// Compile-time dimension defaults
// =============================================================================

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

// Preprocessor guards — same pattern as experiment_9a, so each
// ExternalFunction compiles a .o containing exactly one entry point.
#if !defined(MATMUL_ONLY) && !defined(ZERO_ONLY)
#define MATMUL_ONLY
#define ZERO_ONLY
#endif

// =============================================================================
// Zero kernel for int32 output (must be outside extern "C" — template)
// =============================================================================

template <int M, int N>
void zero_vectorized_i32(int32 *__restrict cOut) {
  constexpr int vectorSize = 16; // 16 int32 = 512-bit vector
  const aie::accum<accfloat, vectorSize> acc =
      aie::zeros<accfloat, vectorSize>();
  for (int i = 0; i < M * N / vectorSize; i++) {
    aie::store_v(cOut + i * vectorSize, acc.template to_vector<int32>());
  }
}

// =============================================================================
// Extern "C" entry points (called from IRON ExternalFunction)
// =============================================================================

extern "C" {

// ---- BF16 → F32 (with BFP16 emulation) ----

#ifdef MATMUL_ONLY
void gemm_bf16_f32_bfp16_packed(const bfloat16 *__restrict pA,
                                const bfloat16 *__restrict pB,
                                float *__restrict pC) {
  matmul_vectorized_8x8x8_bf16_f32_packed<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}

void gemm_i8_i32_packed(const int8 *__restrict pA,
                         const int8 *__restrict pB,
                         int32 *__restrict pC) {
  matmul_vectorized_8x8x8_i8_i32_packed<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}
#endif

#ifdef ZERO_ONLY
void zero_f32_packed(float *__restrict cOut) {
  zero_vectorized_f32<DIM_M, DIM_N>(cOut);
}

void zero_i32_packed(int32 *__restrict cOut) {
  zero_vectorized_i32<DIM_M, DIM_N>(cOut);
}
#endif

} // extern "C"
