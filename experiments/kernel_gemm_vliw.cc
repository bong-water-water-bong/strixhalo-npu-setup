//===- kernel_gemm_vliw.cc --------------------------------------*- C++ -*-===//
//
// Explicit VLIW-scheduled GEMM kernel for AIE2P.
//
// Uses Chess pragmas for II=1 when compiled with Chess (closed-source).
// Falls back to 4× manual unrolling when compiled with Peano/aiecc.
//
// KEY INSIGHT: 4 accumulator chains + 4× K-loop unrolling = 16 independent
// MACs per iteration. Each accumulator reused once per 16 MACs → 16-cycle
// distance matches 8-cycle mmul pipeline with margin. Peano's VLIW scheduler
// gets maximum ILP to find a good schedule.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
using namespace aie;

// =============================================================================
// Chess/Peano-agnostic SWP pragmas
// =============================================================================
#ifdef __chess__
#define SWP_PRAGMA(ops, free) \
  [[chess::prepare_for_pipelining]] \
  [[chess::modulo_scheduling_budget_ratio(100)]] \
  [[chess::peel_pipelined_loop(ops)]] \
  [[chess::keep_free_for_pipelining(free)]]
#else
#define SWP_PRAGMA(ops, free)
#endif

// =============================================================================
// Zero kernels
// =============================================================================
template <int M, int N>
void zero_f32(float *__restrict cOut) {
  constexpr int V = 16;
  auto acc = aie::zeros<accfloat, V>();
  for (int i = 0; i < M * N / V; i++)
    aie::store_v(cOut + i * V, acc.template to_vector<float>());
}

template <int M, int N>
void zero_i32(int32 *__restrict cOut) {
  constexpr int V = 16;
  auto acc = aie::zeros<accfloat, V>();
  for (int i = 0; i < M * N / V; i++)
    aie::store_v(cOut + i * V, acc.template to_vector<int32>());
}

// =============================================================================
// 2×2 mmul — 4× UNROLLED K-loop for maximum ILP
//
// Each K-iteration processes 4 consecutive 8×8 micro-tiles.
// 4 accumulators × 4 unroll = 16 MAC operations per loop iteration.
// Peano's scheduler sees 16 independent MACs + 16 loads → much more
// flexibility for VLIW bundle packing than the 4+4 in the 2× unrolled version.
//
// For the Chess compiler: SWP_PRAGMA enables modulo scheduling.
// =============================================================================
template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
static inline void matmul_vliw(
    const T_in *__restrict pA,
    const T_in *__restrict pB,
    T_out *__restrict pC)
{
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
  constexpr unsigned SZ_A = MMUL::size_A;  // 64
  constexpr unsigned SZ_B = MMUL::size_B;  // 64
  constexpr unsigned SZ_C = MMUL::size_C;  // 64

  for (unsigned z = 0; z < rowA; z += 2) {
    T_out *pC1 = pC + (z * colB) * SZ_C;
    T_out *pC2 = pC + ((z + 1) * colB) * SZ_C;

    for (unsigned j = 0; j < colB; j += 2) {
      const T_in *pA1 = pA + (z * colA) * SZ_A;
      const T_in *pA2 = pA + ((z + 1) * colA) * SZ_A;
      const T_in *pB1 = pB + (j * colA) * SZ_B;
      const T_in *pB2 = pB + ((j + 1) * colA) * SZ_B;

      // Initialize accumulators from C
      MMUL C00(aie::load_v<SZ_C>(pC1));
      MMUL C01(aie::load_v<SZ_C>(pC1 + SZ_C));
      MMUL C10(aie::load_v<SZ_C>(pC2));
      MMUL C11(aie::load_v<SZ_C>(pC2 + SZ_C));

      // 4× unrolled K-loop: 16 MACs + 16 loads per iteration
      // Accumulator reuse: C00 used at MACs 0,4,8,12 → 4-cycle gap
      // (not enough for 8-cycle mmul, but Peano may reorder)
      SWP_PRAGMA(16, 8)
      _Pragma("clang loop unroll_count(4)")
      for (unsigned i = 0; i < colA; ++i) {
        aie::vector<T_in, SZ_A> A0 = aie::load_v<SZ_A>(pA1); pA1 += SZ_A;
        aie::vector<T_in, SZ_A> A1 = aie::load_v<SZ_A>(pA2); pA2 += SZ_A;

#ifdef B_COL_MAJ
        aie::vector<T_in, SZ_B> B0 = aie::transpose(
            aie::load_v<SZ_B>(pB1), t, s); pB1 += SZ_B;
        aie::vector<T_in, SZ_B> B1 = aie::transpose(
            aie::load_v<SZ_B>(pB2), t, s); pB2 += SZ_B;
#else
        aie::vector<T_in, SZ_B> B0 = aie::load_v<SZ_B>(pB1); pB1 += SZ_B;
        aie::vector<T_in, SZ_B> B1 = aie::load_v<SZ_B>(pB2); pB2 += SZ_B;
#endif

        C00.mac(A0, B0); C01.mac(A0, B1);
        C10.mac(A1, B0); C11.mac(A1, B1);
      }

      aie::store_v(pC1, C00.template to_vector<T_out>()); pC1 += SZ_C;
      aie::store_v(pC1, C01.template to_vector<T_out>()); pC1 += SZ_C;
      aie::store_v(pC2, C10.template to_vector<T_out>()); pC2 += SZ_C;
      aie::store_v(pC2, C11.template to_vector<T_out>()); pC2 += SZ_C;
    }
  }
}

// =============================================================================
// Typed wrappers
// =============================================================================
template <unsigned m, unsigned k, unsigned n>
static inline void gemm_bf16_f32_vliw(
    const bfloat16 *pA, const bfloat16 *pB, float *pC) {
  constexpr int r=8,s=8,t=8;
  static_assert(m%(2*r)==0 && k%s==0 && n%(2*t)==0);
  matmul_vliw<bfloat16,float,m/r,k/s,n/t,r,s,t>(pA,pB,pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void gemm_i8_i32_vliw(
    const int8 *pA, const int8 *pB, int32 *pC) {
  constexpr int r=8,s=8,t=8;
  static_assert(m%(2*r)==0 && k%s==0 && n%(2*t)==0);
  matmul_vliw<int8,int32,m/r,k/s,n/t,r,s,t>(pA,pB,pC);
}

// =============================================================================
// Extern "C" entry points
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

extern "C" {

#ifdef MATMUL_ONLY
void gemm_bf16_f32_vliw(const bfloat16 *pA, const bfloat16 *pB, float *pC) {
  gemm_bf16_f32_vliw<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}
void gemm_i8_i32_vliw(const int8 *pA, const int8 *pB, int32 *pC) {
  gemm_i8_i32_vliw<DIM_M, DIM_K, DIM_N>(pA, pB, pC);
}
#endif

#ifdef ZERO_ONLY
void zero_f32_vliw(float *c) { zero_f32<DIM_M, DIM_N>(c); }
void zero_i32_vliw(int32 *c) { zero_i32<DIM_M, DIM_N>(c); }
#endif

} // extern "C"
