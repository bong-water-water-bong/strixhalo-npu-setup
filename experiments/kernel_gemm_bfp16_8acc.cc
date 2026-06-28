//===- kernel_gemm_bfp16_8acc.cc ---------------------------------*- C++ -*-===//
//
// Phase 6: 2×4 mmul expansion with 8 accumulators for AIE2P.
//
// KEY INSIGHT: With 4 accumulators (2×2 expansion) and 8-cycle mmul pipeline
// depth, each accumulator is reused every 4 MACs → 4-cycle distance between
// same-accumulator uses → 4-cycle pipeline stall (can't achieve II=1).
//
// 2×4 expansion (8 accumulators): same accumulator used every 8 MACs →
// matches 8-cycle mmul pipeline depth → II=1 CAPABLE.
//
// Layout:
//   M: 2 position rows × 8 rows = 16 rows (m=32, z+=2)
//   N: 4 position cols × 8 cols  = 32 cols (n=32, j+=4)
//   K: k/8 K-steps (colA)
//
// 8 accumulators: C00,C01,C02,C03 (for A0) + C10,C11,C12,C13 (for A1)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

using namespace aie;

// =============================================================================
// Zero kernels
// =============================================================================

template <int M, int N>
void zero_vectorized_f32(float *__restrict cOut) {
  constexpr int vectorSize = 16;
  const aie::accum<accfloat, vectorSize> acc =
      aie::zeros<accfloat, vectorSize>();
  for (int i = 0; i < M * N / vectorSize; i++) {
    aie::store_v(cOut + i * vectorSize, acc.template to_vector<float>());
  }
}

template <int M, int N>
void zero_vectorized_i32(int32 *__restrict cOut) {
  constexpr int vectorSize = 16;
  const aie::accum<accfloat, vectorSize> acc =
      aie::zeros<accfloat, vectorSize>();
  for (int i = 0; i < M * N / vectorSize; i++) {
    aie::store_v(cOut + i * vectorSize, acc.template to_vector<int32>());
  }
}

// =============================================================================
// 2×4 mmul expansion — 8 accumulators
//
// Template parameters:
//   rowA = M / r     colA = K / s     colB = N / t
//   r,s,t = 8,8,8
//
// Requires: colB >= 4 (for the 4-column N expansion)
//           rowA >= 2 (for the 2-row M expansion)
// =============================================================================

template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_2x4_mmul_packed(
    const T_in *__restrict pA,
    const T_in *__restrict pB,
    T_out *__restrict pC)
{
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  // Outer loop over M positions (2 at a time)
  for (unsigned z = 0; z < rowA; z += 2) {

    // Outer loop over N positions (4 at a time for 8 accumulators)
    for (unsigned j = 0; j < colB; j += 4) {

      // ---- C output pointers (4 N positions per M row) ----
      T_out *__restrict pC1 = pC + (z * colB + j) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + j) * MMUL::size_C;

      // ---- A base pointers (M positions z and z+1) ----
      const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
      const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;

      // ---- B base pointers (N positions j, j+1, j+2, j+3) ----
      const T_in *__restrict pB1 = pB + (j * colA) * MMUL::size_B;
      const T_in *__restrict pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
      const T_in *__restrict pB3 = pB + ((j + 2) * colA) * MMUL::size_B;
      const T_in *__restrict pB4 = pB + ((j + 3) * colA) * MMUL::size_B;

      // ---- Load 8 accumulators from C ----
      aie::vector<T_out, MMUL::size_C> vC00 = aie::load_v<MMUL::size_C>(pC1);
      aie::vector<T_out, MMUL::size_C> vC01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
      aie::vector<T_out, MMUL::size_C> vC02 = aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
      aie::vector<T_out, MMUL::size_C> vC03 = aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

      aie::vector<T_out, MMUL::size_C> vC10 = aie::load_v<MMUL::size_C>(pC2);
      aie::vector<T_out, MMUL::size_C> vC11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
      aie::vector<T_out, MMUL::size_C> vC12 = aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C);
      aie::vector<T_out, MMUL::size_C> vC13 = aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C);

      MMUL C00(vC00), C01(vC01), C02(vC02), C03(vC03);
      MMUL C10(vC10), C11(vC11), C12(vC12), C13(vC13);

      // ================================================================
      // SOFTWARE-PIPELINED K-REDUCTION LOOP (8 accumulators)
      //
      // MAC-then-load SWP pattern:
      //   PROLOGUE: Load iteration 0's A, B[0..3]
      //   KERNEL:   MAC with current data, THEN load next iteration
      //   EPILOGUE: MAC with last loaded data
      //
      // The 8 MACs take 8+ cycles to execute. During those cycles,
      // the next iteration's 6 loads (2A + 4B) execute on VLDA/VLDB,
      // completing well before the next MAC phase needs them.
      //
      // 8 accumulators: each used once per 8 MACs → matches 8-cycle
      // mmul pipeline → NO RAW stalls on accumulators.
      // ================================================================

      // ---- PROLOGUE: Load iteration 0 ----
      aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
      pA1 += MMUL::size_A;
      aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
      pA2 += MMUL::size_A;

#ifdef B_COL_MAJ
      aie::vector<T_in, MMUL::size_B> B0 =
          aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
      pB1 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B1 =
          aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
      pB2 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B2 =
          aie::transpose(aie::load_v<MMUL::size_B>(pB3), t, s);
      pB3 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B3 =
          aie::transpose(aie::load_v<MMUL::size_B>(pB4), t, s);
      pB4 += MMUL::size_B;
#else
      aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
      pB2 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB3);
      pB3 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B3 = aie::load_v<MMUL::size_B>(pB4);
      pB4 += MMUL::size_B;
#endif

      // ---- KERNEL: MAC then load (iterations 1 to colA-1) ----
      for (unsigned i = 1; i < colA; ++i) {

        // First: 8 MACs with previously loaded data
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C02.mac(A0, B2);
        C03.mac(A0, B3);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
        C12.mac(A1, B2);
        C13.mac(A1, B3);

        // Then: load next iteration's data (overlaps with MAC pipeline drain)
        A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;

#ifdef B_COL_MAJ
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
        pB1 += MMUL::size_B;
        B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
        pB2 += MMUL::size_B;
        B2 = aie::transpose(aie::load_v<MMUL::size_B>(pB3), t, s);
        pB3 += MMUL::size_B;
        B3 = aie::transpose(aie::load_v<MMUL::size_B>(pB4), t, s);
        pB4 += MMUL::size_B;
#else
        B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;
        B2 = aie::load_v<MMUL::size_B>(pB3);
        pB3 += MMUL::size_B;
        B3 = aie::load_v<MMUL::size_B>(pB4);
        pB4 += MMUL::size_B;
#endif
      }

      // ---- EPILOGUE: MAC with last loaded data ----
      C00.mac(A0, B0);
      C01.mac(A0, B1);
      C02.mac(A0, B2);
      C03.mac(A0, B3);
      C10.mac(A1, B0);
      C11.mac(A1, B1);
      C12.mac(A1, B2);
      C13.mac(A1, B3);

      // ---- Store 8 accumulators to C ----
      aie::store_v(pC1, C00.template to_vector<T_out>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<T_out>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C02.template to_vector<T_out>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C03.template to_vector<T_out>());
      pC1 += MMUL::size_C;

      aie::store_v(pC2, C10.template to_vector<T_out>());
      pC2 += MMUL::size_C;
      aie::store_v(pC2, C11.template to_vector<T_out>());
      pC2 += MMUL::size_C;
      aie::store_v(pC2, C12.template to_vector<T_out>());
      pC2 += MMUL::size_C;
      aie::store_v(pC2, C13.template to_vector<T_out>());
      pC2 += MMUL::size_C;
    }
  }
}

// =============================================================================
// Typed entry wrappers
// =============================================================================

template <unsigned m, unsigned k, unsigned n>
[[clang::always_inline]]
static inline void matmul_vectorized_8x8x8_bf16_f32_8acc(
    const bfloat16 *__restrict pA,
    const bfloat16 *__restrict pB,
    float *__restrict pC)
{
  constexpr int r = 8, s = 8, t = 8;
  static_assert(m % (2 * r) == 0);  // M divisible by 16 for 2-position expansion
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);  // N divisible by 32 for 4-position expansion
  matmul_vectorized_2x4_mmul_packed<bfloat16, float,
      (m / r), (k / s), (n / t), r, s, t>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
[[clang::always_inline]]
static inline void matmul_vectorized_8x8x8_i8_i32_8acc(
    const int8 *__restrict pA,
    const int8 *__restrict pB,
    int32 *__restrict pC)
{
  constexpr int r = 8, s = 8, t = 8;
  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);
  matmul_vectorized_2x4_mmul_packed<int8, int32,
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

#if !defined(MATMUL_ONLY) && !defined(ZERO_ONLY)
#define MATMUL_ONLY
#define ZERO_ONLY
#endif

// =============================================================================
// Extern "C" entry points
// =============================================================================

extern "C" {

#ifdef MATMUL_ONLY
void gemm_bf16_f32_bfp16_8acc(const bfloat16 *__restrict pA,
                               const bfloat16 *__restrict pB,
                               float *__restrict pC) {
  matmul_vectorized_8x8x8_bf16_f32_8acc<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}

void gemm_i8_i32_8acc(const int8 *__restrict pA,
                       const int8 *__restrict pB,
                       int32 *__restrict pC) {
  matmul_vectorized_8x8x8_i8_i32_8acc<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}
#endif

#ifdef ZERO_ONLY
void zero_f32_8acc(float *__restrict cOut) {
  zero_vectorized_f32<DIM_M, DIM_N>(cOut);
}

void zero_i32_8acc(int32 *__restrict cOut) {
  zero_vectorized_i32<DIM_M, DIM_N>(cOut);
}
#endif

} // extern "C"
