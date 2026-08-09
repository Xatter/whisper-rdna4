// common.h -- shared WMMA fragment helpers, tile loaders, and small device
// utilities for the rocm-whisper custom kernel set (gfx1201 / RDNA4).
//
// Owned by Agent B (kernels K1 layernorm, K2 gemm_wmma, K3 attention_encoder).
// Agent C's K4 (attention_decode) and K5 (logits) may include this header too
// (K5's fused GEMM+argmax epilogue reuses the WMMA fragment loaders below).
// Keep this file dependency-free (just hip_runtime.h) and stable -- changing
// a signature here breaks both agents' kernels.
//
// ---------------------------------------------------------------------------
// WMMA fragment layout (gfx12 / RDNA4), per
// amd-r9700-docs/gpuopen/using-matrix-cores-rdna4.md (the ONLY authoritative
// source for this -- the RDNA3 layout duplicates elements and is WRONG here):
//
//   Intrinsic: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c) -> d
//     a, b : 8 x fp16 (packed in a 128-bit VGPR pair), one 16x16 tile spread
//            across a 32-lane wave, 8 elements/lane (16*16/32 = 8).
//     c, d : 8 x fp32, same per-lane distribution.
//     A is consumed as COLUMN-major, B/C/D as ROW-major -- but this is a
//     *hardware wiring* fact, not something the caller has to transpose by
//     hand: if both A and B are stored as ordinary row-major 16x16 blocks in
//     memory, the two loader loops below (different index formulas for A vs
//     B) feed the hardware correctly and it computes plain D = A*B + C, no
//     in-register transpose needed (RDNA4 has none -- cheatsheet SS11).
//
//   Per-lane indexing (lane = threadIdx.x % 32, MUST be a lane id in [0,32),
//   i.e. blockDim.x must be a multiple of 32 and lanes of one wave must map
//   to contiguous threadIdx.x):
//     laneWrapped = lane % 16   (0..15)
//     laneGroup   = lane / 16   (0 or 1)
//     for ele in 0..7:
//       row = ele + laneGroup * 8         // 0..15, split into two 8-halves
//       col = laneWrapped                 // 0..15, fixed per lane
//       A_frag[ele] = A_tile[row * ldA + col]   // A read as if row-major;
//                                                // hardware treats it as A^T
//                                                // internally (== the
//                                                // "column-major" contract)
//       B_frag[ele] = B_tile[row * ldB + col]   // wait -- see loaders below,
//                                                // the *index roles* of
//                                                // row/col swap between A
//                                                // and B; spelled out
//                                                // precisely in the loader
//                                                // functions, not in prose,
//                                                // to avoid transcription
//                                                // bugs. Trust the code.
//     D_frag[ele] written the same way B is read (row-major, row=ele+group*8,
//     col=laneWrapped) -- so D can chain directly into the next matmul's B
//     operand with zero lane shuffling (gpuopen doc SS "Implementing a simple
//     MLP").
//
// ---------------------------------------------------------------------------

#pragma once
#include <hip/hip_runtime.h>

// ---- vector types for the WMMA intrinsic ----------------------------------
typedef _Float16 fp16_t;
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float     float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float     float4v __attribute__((ext_vector_type(4)));

#define WMMA_TILE 16          // WMMA operates on 16x16x16 tiles only.
#define WAVE_SIZE 32          // gfx1201 native wavefront -- build wave32 only
                               // (see build.sh -mno-wavefrontsize64 and the
                               // -Rpass-analysis check every kernel does).

// ---- raw intrinsic wrapper --------------------------------------------------
__device__ __forceinline__ float8 wmma_mma_f16(half8 a, half8 b, float8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// ---- fragment loaders --------------------------------------------------
// All three assume `lane` == threadIdx.x % 32 and that the source tile's
// top-left element is `tile[0]`, with `ld` = elements between consecutive
// *logical rows* of that source buffer (NOT necessarily 16 -- e.g. our GEMM
// LDS staging buffers use ld = BLOCK_K or BLOCK_N, larger than the 16x16
// sub-tile being read out of them).
//
// A operand: source stored ordinary row-major (row-major "A[m][k]"), read
// with row = lane's fixed M-position... no: per the doc's worked example,
// the A loader's *lane-fixed* index is the row (laneWrapped), and the
// per-element index (ele+group*8) is the column. This is what makes the
// hardware treat A as transposed relative to B without any real transpose.
__device__ __forceinline__ half8 wmma_load_a(const fp16_t* tile, int ld, int lane) {
  const int lw = lane & 15;
  const int lg = lane >> 4;
  half8 f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    f[e] = tile[lw * ld + (e + lg * 8)];
  }
  return f;
}

// B operand: source stored ordinary row-major ("B[k][n]"), read with the
// per-element index (ele+group*8) as the row and the lane-fixed laneWrapped
// as the column -- i.e. each lane owns one fixed column, 8 rows of it.
__device__ __forceinline__ half8 wmma_load_b(const fp16_t* tile, int ld, int lane) {
  const int lw = lane & 15;
  const int lg = lane >> 4;
  half8 f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    f[e] = tile[(e + lg * 8) * ld + lw];
  }
  return f;
}

// D/C operand: row-major, same index shape as the B loader (row=ele+group*8,
// col=laneWrapped). Store fp32 accumulator fragment to a row-major fp32
// buffer (used for scratch/epilogue only -- steady-state accumulation stays
// in registers).
__device__ __forceinline__ void wmma_store_d_f32(float* tile, int ld, int lane, float8 f) {
  const int lw = lane & 15;
  const int lg = lane >> 4;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    tile[(e + lg * 8) * ld + lw] = f[e];
  }
}

// Store D fragment to a row-major fp16 buffer, applying a per-lane transform
// `xform(value, row_in_tile, col_in_tile)` first (used for bias/gelu/residual
// epilogues so the cast to fp16 happens exactly once, after the epilogue op).
template <typename F>
__device__ __forceinline__ void wmma_store_d_fp16_xform(fp16_t* tile, int ld, int lane, float8 f, F xform) {
  const int lw = lane & 15;
  const int lg = lane >> 4;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int row = e + lg * 8;
    tile[row * ld + lw] = (fp16_t)xform(f[e], row, lw);
  }
}

// Row/col within a 16x16 fragment for element `e`, given `lane` -- exposed so
// epilogues can compute a per-fragment index (e.g. bias[col] or
// residual[(row_global)*N + col_global]) without duplicating the layout math.
__device__ __forceinline__ void wmma_frag_rowcol(int lane, int e, int& row, int& col) {
  row = e + (lane >> 4) * 8;
  col = lane & 15;
}

// ---- GELU (exact erf form, matches torch F.gelu(approximate='none')) ------
__device__ __forceinline__ float gelu_f32(float x) {
  // 0.5 * x * (1 + erf(x / sqrt(2)))
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// ---- pack/unpack two fp32 -> one fp16x2 (compiler-friendly downcast) ------
__device__ __forceinline__ half8 pack_f32x8_to_half8(float8 v) {
  half8 h;
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = (fp16_t)v[i];
  return h;
}

// ---- block-wide reduction (sum) across an arbitrary number of wave32
// waves, via LDS. Used by K1 layernorm (mean/var) and K3 (softmax row sums
// when a query tile spans >1 wave -- K3 keeps one wave per query tile so it
// only needs the intra-wave shuffle reduction below, but this is kept here
// since K1 needs the cross-wave version and both kernels share the header).
__device__ __forceinline__ float wave_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor(v, off, 32);
  }
  return v;
}

// blockReduceSum: `scratch` must have room for (blockDim.x/32) floats.
// Every thread gets the full-block sum back (broadcast via scratch[0]).
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  v = wave_reduce_sum(v);
  int lane = threadIdx.x & 31;
  int wave = threadIdx.x >> 5;
  int nwaves = blockDim.x >> 5;
  if (lane == 0) scratch[wave] = v;
  __syncthreads();
  float total = 0.0f;
  if (wave == 0) {
    float x = (lane < nwaves) ? scratch[lane] : 0.0f;
    x = wave_reduce_sum(x);
    if (lane == 0) scratch[0] = x;
  }
  __syncthreads();
  total = scratch[0];
  __syncthreads(); // protect scratch reuse by callers doing a 2nd reduction
  return total;
}

// ---- epilogue kind (used by K2 gemm_wmma.hip; declared here so K5 can
// share the same enum if it wants). ------------------------------------
enum class EpilogueKind : int {
  BIAS = 0,
  BIAS_GELU = 1,
  BIAS_RESIDUAL = 2,
};
