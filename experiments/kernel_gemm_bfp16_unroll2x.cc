//===- kernel_gemm_bfp16_unroll2x.cc ----------------------------*- C++ -*-===//
//
// Variant of kernel_gemm_bfp16_packed.cc with 2× unrolled inner K-loop.
// Provides Peano's VLIW scheduler with 8 independent loads + 8 MACs per
// iteration, giving it more ILP to pack into VLIW bundles.
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

// =============================================================================
// 2×2 mmul kernel — 2× UNROLLED inner K-loop
//
// The inner K-loop is manually unrolled by 2×, processing two 8×8 micro-tiles
// per iteration. This doubles the number of operations visible to Peano's
// VLIW scheduler per iteration, helping it find a better instruction schedule.
//
// 8 loads + 8 MACs = 16 operations per loop iteration vs 8 in the base kernel.
// =============================================================================

template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
[[clang::always_inline]]
static inline void matmul_vectorized_2x2_mmul_packed_unroll2x(
    const T_in *__restrict pA,
    const T_in *__restrict pB,
    T_out *__restrict pC)
{
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  for (unsigned z = 0; z < rowA; z += 2) {

    T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2) {

      const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
      const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
      const T_in *__restrict pB1 = pB + (j * colA) * MMUL::size_B;
      const T_in *__restrict pB2 = pB + ((j + 1) * colA) * MMUL::size_B;

      aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
      aie::vector<T_out, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
      aie::vector<T_out, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
      aie::vector<T_out, MMUL::size_C> acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

      MMUL C00(acc_C00);
      MMUL C01(acc_C01);
      MMUL C10(acc_C10);
      MMUL C11(acc_C11);

      // ================================================================
      // 2× UNROLLED K-reduction loop
      //
      // Two K-steps per iteration → 2× the ILP for VLIW scheduling.
      // The 4 accumulators hide mmul pipeline latency across both steps.
      // ================================================================
      _Pragma("clang loop min_iteration_count(4)")  // colA/2 iterations
      for (unsigned i = 0; i < colA; i += 2) {

        // ---- K-step i: load + transpose ----
        aie::vector<T_in, MMUL::size_A> A0_i0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<T_in, MMUL::size_A> A1_i0 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;

#ifdef B_COL_MAJ
        aie::vector<T_in, MMUL::size_B> B0_i0 =
            aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
        pB1 += MMUL::size_B;
        aie::vector<T_in, MMUL::size_B> B1_i0 =
            aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
        pB2 += MMUL::size_B;
#else
        aie::vector<T_in, MMUL::size_B> B0_i0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        aie::vector<T_in, MMUL::size_B> B1_i0 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;
#endif

        // ---- K-step i+1: load + transpose ----
        aie::vector<T_in, MMUL::size_A> A0_i1 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<T_in, MMUL::size_A> A1_i1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;

#ifdef B_COL_MAJ
        aie::vector<T_in, MMUL::size_B> B0_i1 =
            aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
        pB1 += MMUL::size_B;
        aie::vector<T_in, MMUL::size_B> B1_i1 =
            aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
        pB2 += MMUL::size_B;
#else
        aie::vector<T_in, MMUL::size_B> B0_i1 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        aie::vector<T_in, MMUL::size_B> B1_i1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;
#endif

        // ---- MACs interleaved across both steps ----
        // Step i: chain 0 and 1
        C00.mac(A0_i0, B0_i0);
        C01.mac(A0_i0, B1_i0);
        // Step i+1: chain 0 and 1
        C00.mac(A0_i1, B0_i1);
        C01.mac(A0_i1, B1_i1);
        // Step i: chain 2 and 3
        C10.mac(A1_i0, B0_i0);
        C11.mac(A1_i0, B1_i0);
        // Step i+1: chain 2 and 3
        C10.mac(A1_i1, B0_i1);
        C11.mac(A1_i1, B1_i1);
      }

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
static inline void matmul_vectorized_8x8x8_bf16_f32_packed_unroll2x(
    const bfloat16 *__restrict pA,
    const bfloat16 *__restrict pB,
    float *__restrict pC)
{
  constexpr int r = 8, s = 8, t = 8;
  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);
  matmul_vectorized_2x2_mmul_packed_unroll2x<bfloat16, float,
      (m / r), (k / s), (n / t), r, s, t>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
[[clang::always_inline]]
static inline void matmul_vectorized_8x8x8_i8_i32_packed_unroll2x(
    const int8 *__restrict pA,
    const int8 *__restrict pB,
    int32 *__restrict pC)
{
  constexpr int r = 8, s = 8, t = 8;
  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);
  matmul_vectorized_2x2_mmul_packed_unroll2x<int8, int32,
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
// Zero kernel for int32 output
// =============================================================================

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
// Extern "C" entry points
// =============================================================================

extern "C" {

#ifdef MATMUL_ONLY
void gemm_bf16_f32_bfp16_packed_unroll2x(const bfloat16 *__restrict pA,
                                          const bfloat16 *__restrict pB,
                                          float *__restrict pC) {
  matmul_vectorized_8x8x8_bf16_f32_packed_unroll2x<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}

void gemm_i8_i32_packed_unroll2x(const int8 *__restrict pA,
                                  const int8 *__restrict pB,
                                  int32 *__restrict pC) {
  matmul_vectorized_8x8x8_i8_i32_packed_unroll2x<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}
#endif

#ifdef ZERO_ONLY
void zero_f32_packed_unroll2x(float *__restrict cOut) {
  zero_vectorized_f32<DIM_M, DIM_N>(cOut);
}

void zero_i32_packed_unroll2x(int32 *__restrict cOut) {
  zero_vectorized_i32<DIM_M, DIM_N>(cOut);
}
#endif

} // extern "C"
