//===- kernel_gemm_bfp16_swp.cc ---------------------------------*- C++ -*-===//
//
// Phase 6: Manually software-pipelined BFP16 GEMM micro-kernel for AIE2P.
//
// APPROACH: Prologue/Kernel/Epilogue SWP
//
//   The inner K-loop is decomposed into three phases:
//
//   PROLOGUE (iteration 0):
//     Load A[0], B[0] — first set of operands
//     MAC with A[0], B[0] — first compute
//
//   KERNEL (iterations 1 to colA-1):
//     PREFETCH A[i], B[i] — loads for current iteration
//     MAC with A[i-1], B[i-1] — compute from previous iteration's prefetch
//     → Loads of iteration i overlap with MACs of iteration i-1
//
//   EPILOGUE (after loop):
//     MAC with A[last], B[last] — final compute
//
//   The critical insight: by the time MACs execute, the loads from the
//   PREVIOUS loop iteration have already completed. This separates each
//   load from its first MAC use by ~4 VLIW bundles, hiding load latency.
//
//   For B_COL_MAJ: transpose is in the load phase, so it also completes
//   before MACs need the data.
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
// 2×2 mmul kernel — MANUALLY SOFTWARE-PIPELINED
//
// The loop is decomposed into prologue/kernel/epilogue phases.
// Within the kernel, loads for iteration i execute while MACs for
// iteration i-1 are still in flight, hiding load and transpose latency.
//
// Template parameters:
//   rowA = M / 8    colA = K / 8    colB = N / 8
//   r,s,t = 8,8,8
// =============================================================================

template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_2x2_mmul_swp(
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

      // Initialize accumulators from C
      aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
      aie::vector<T_out, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
      aie::vector<T_out, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
      aie::vector<T_out, MMUL::size_C> acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

      MMUL C00(acc_C00);
      MMUL C01(acc_C01);
      MMUL C10(acc_C10);
      MMUL C11(acc_C11);

      // ================================================================
      // SOFTWARE-PIPELINED K-REDUCTION LOOP
      //
      // Structure: MAC-then-load ordering avoids vector copies.
      //
      //   PROLOGUE:  Load iteration 0's operands (no MAC yet)
      //   KERNEL:    MAC with current data, THEN load next iteration
      //   EPILOGUE:  MAC with last loaded data
      //
      // The key insight: loads from kernel iteration i execute AFTER
      // the MACs of iteration i, giving ~4 VLIW bundles for load
      // latency to resolve before the next iteration's MACs need them.
      //
      // No vector copy/swap needed — A0,A1,B0,B1 are simply reassigned.
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
#else
      aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
      pB2 += MMUL::size_B;
#endif

      // ---- KERNEL: MAC then load (iterations 1 to colA-1) ----
      for (unsigned i = 1; i < colA; ++i) {

        // First: MAC with data loaded in previous iteration (or prologue)
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);

        // Then: load next iteration's data (will be used in next iter)
        A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;

#ifdef B_COL_MAJ
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
        pB1 += MMUL::size_B;
        B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
        pB2 += MMUL::size_B;
#else
        B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;
#endif
      }

      // ---- EPILOGUE: MAC with last loaded data ----
      C00.mac(A0, B0);
      C01.mac(A0, B1);
      C10.mac(A1, B0);
      C11.mac(A1, B1);

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
static inline void matmul_vectorized_8x8x8_bf16_f32_swp(
    const bfloat16 *__restrict pA,
    const bfloat16 *__restrict pB,
    float *__restrict pC)
{
  constexpr int r = 8, s = 8, t = 8;
  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);
  matmul_vectorized_2x2_mmul_swp<bfloat16, float,
      (m / r), (k / s), (n / t), r, s, t>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
[[clang::always_inline]]
static inline void matmul_vectorized_8x8x8_i8_i32_swp(
    const int8 *__restrict pA,
    const int8 *__restrict pB,
    int32 *__restrict pC)
{
  constexpr int r = 8, s = 8, t = 8;
  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);
  matmul_vectorized_2x2_mmul_swp<int8, int32,
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
void gemm_bf16_f32_bfp16_swp(const bfloat16 *__restrict pA,
                              const bfloat16 *__restrict pB,
                              float *__restrict pC) {
  matmul_vectorized_8x8x8_bf16_f32_swp<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}

void gemm_i8_i32_swp(const int8 *__restrict pA,
                      const int8 *__restrict pB,
                      int32 *__restrict pC) {
  matmul_vectorized_8x8x8_i8_i32_swp<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}
#endif

#ifdef ZERO_ONLY
void zero_f32_swp(float *__restrict cOut) {
  zero_vectorized_f32<DIM_M, DIM_N>(cOut);
}

void zero_i32_swp(int32 *__restrict cOut) {
  zero_vectorized_i32<DIM_M, DIM_N>(cOut);
}
#endif

} // extern "C"
