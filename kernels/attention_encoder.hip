// attention_encoder.hip -- K3: non-causal encoder flash attention.
// gfx1201 / RDNA4, wave32. FA-2 style online softmax, both matmuls fp16 WMMA,
// fp32 accumulate/softmax. head_dim=64, seq padded 1500->1504 (next multiple
// of 16, the WMMA tile granularity -- NOT a multiple of 128 or 64, so both
// the last Q-tile (96 of 128 rows) and the last K/V tile (32 of 64 rows) are
// genuine partial tiles; both remainders are themselves multiples of 16 so
// no sub-16 masking is ever needed, only two boundary checks: (1) array
// bounds against S_pad (zero-fill loads / skip stores past S_pad -- avoids
// reading/writing outside the padded buffer), and (2) softmax -inf masking
// against S_real=1500 (excludes the 4 padding positions 1500..1503 from the
// softmax). Per DESIGN.md SS6, (2) is evaluated only on the one K/V tile
// that straddles S_real (kv0=1472..1503); every earlier tile skips the
// check entirely (a block-uniform branch, not a per-lane compare in the hot
// loop).
//
// ---------------------------------------------------------------------------
// WHY THIS KERNEL NEEDS AN LDS ROUND-TRIP FOR P (softmax probs)
// ---------------------------------------------------------------------------
// QK^T's output (scores, D-fragment) has row=query (via e+laneGroup*8),
// col=kv (via lane&15) -- the hardware's fixed D-fragment convention
// (common.h). PV's A-operand (P) must be loaded via wmma_load_a, whose
// fixed-per-lane index is treated as A's ROW -- i.e. it wants query on the
// lane-fixed axis, kv on the e-varying axis. That is the TRANSPOSE of what
// QK^T produced. RDNA4 has no in-register transpose (cheatsheet SS11); AMD's
// documented workaround is an identity-matrix WMMA multiply, which still
// costs a WMMA issue slot. This kernel instead writes P to a small per-wave
// LDS scratch tile (plain row-major store, matching the D-fragment's own
// addressing) and reloads it with wmma_load_a for the PV matmul. This is
// pure LDS traffic -- it does not touch the shared matrix/vector issue port
// at all, so it is strictly cheaper than the identity-matmul trick under the
// "count non-matmul instructions in the inner loop" rule (cheatsheet SS2).
// Same LDS round-trip is also how K needs to reach the QK^T matmul: K's
// natural layout is (kv, head_dim) but QK^T's B-operand needs (head_dim, kv)
// -- i.e. actually-transposed data, not just a different fragment-index
// convention -- so K is DMA'd into LDS transposed (Kt_lds[d][s] = K[s][d]),
// a strided LDS *write* (not a matmul), padded by +1 column to detune the
// resulting bank-conflict pattern.
//
// LDS budget (single-buffered K/V -- no room left for double buffering at
// this tile size, see "what I'd fix next" in the report):
//   Q_lds  128 x 64  fp16 = 16384 B  (loaded once, resident whole kernel)
//   Kt_lds  64 x 65  fp16 =  8320 B  (+1 pad column)
//   V_lds   64 x 64  fp16 =  8192 B
//   P_lds  8 x (16x64) fp16 = 16384 B (one 16x64 scratch tile per wave)
//   total ~49280 B, under the 64 KB/block budget with room to spare.
//
// VGPR / occupancy history (see -Rpass-analysis output in the report): the
// first version used 4 waves x WAVE_M=32 (FRAGS_M=2) and materialized a full
// scores[FRAGS_M][KV_SUB][8] = 64-float live array to batch the online-
// softmax update over the whole 64-wide K/V tile at once. That measured at
// 256 VGPR/lane with 3 VGPR SPILLED and only 2 waves/SIMD occupancy -- a
// real correctness/perf risk (spills round-trip through memory), far over
// the <=168 VGPR budget DESIGN.md SS6 asks for so V_EXP_F32 can overlap
// another wave's WMMA. Fixed two ways, both applied: (1) N_WAVES doubled to
// 8 (WAVE_M=16, FRAGS_M=1), halving all FRAGS_M-scaled persistent state;
// (2) the online-softmax update granularity shrunk from 64-wide (batch all
// 4 kv sub-tiles' scores before updating m/l/O) to 16-wide (update
// immediately after each kv sub-tile's WMMA, fold that sub-tile's PV
// contribution in before moving to the next) -- this drops the persistent
// "scores" array from 64 floats to an 8-float transient that dies before
// the next sub-tile's WMMA even issues. Trade-off: 4x more rescale/update
// vector-op sequences per 64-wide tile instead of 1x, which is a real tax
// under the "vector ops steal WMMA issue slots ~1:1" rule (cheatsheet SS2)
// -- accepted deliberately because register spilling is categorically worse
// than extra vector ops. Re-measured VGPR/occupancy after both fixes is in
// the report.

#include "common.h"

#define HEAD_DIM 64
#define Q_TILE 128
#define KV_TILE 64
#define N_WAVES 8
#define WAVE_M (Q_TILE / N_WAVES)      // 16
#define FRAGS_M (WAVE_M / 16)          // 1
#define HD_CHUNKS (HEAD_DIM / 16)      // 4
#define KV_SUB (KV_TILE / 16)          // 4
#define KT_LD (HEAD_DIM + 1)           // 65, bank-conflict pad for the transposed K write
#define NEG_BIG (-1.0e30f)

extern "C" __global__ __launch_bounds__(32 * N_WAVES) void attn_encoder_kernel(
    const fp16_t* __restrict__ Q,  // (BH, S_pad, 64), scale already folded in
    const fp16_t* __restrict__ K,  // (BH, S_pad, 64)
    const fp16_t* __restrict__ V,  // (BH, S_pad, 64)
    fp16_t* __restrict__ O,        // (BH, S_pad, 64)
    int S_pad, int S_real) {
  const int bh = blockIdx.y;
  const int q_tile = blockIdx.x;
  const int q_base = q_tile * Q_TILE;

  const fp16_t* Qbh = Q + (size_t)bh * S_pad * HEAD_DIM;
  const fp16_t* Kbh = K + (size_t)bh * S_pad * HEAD_DIM;
  const fp16_t* Vbh = V + (size_t)bh * S_pad * HEAD_DIM;
  fp16_t* Obh = O + (size_t)bh * S_pad * HEAD_DIM;

  __shared__ __align__(16) fp16_t Q_lds[Q_TILE * HEAD_DIM];
  __shared__ fp16_t Kt_lds[HEAD_DIM][KT_LD];
  __shared__ __align__(16) fp16_t V_lds[KV_TILE * HEAD_DIM];
  __shared__ fp16_t P_lds[N_WAVES][WAVE_M * KV_TILE];

  const int lane = threadIdx.x;
  const int wave_id = threadIdx.y;
  const int tid = wave_id * 32 + lane;
  constexpr int NTHREADS = 32 * N_WAVES;
  const int lw = lane & 15;
  const int lg = lane >> 4;
  const int wave_q_off = wave_id * WAVE_M;

  // ---- load Q tile once (vectorized, S_pad-bounds masked) ---------------
  {
    constexpr int total8 = (Q_TILE * HEAD_DIM) / 8;
    for (int i = tid; i < total8; i += NTHREADS) {
      int elem0 = i * 8;
      int row = elem0 / HEAD_DIM;
      int col = elem0 % HEAD_DIM;
      int grow = q_base + row;
      half8 v;
      if (grow < S_pad) {
        v = *reinterpret_cast<const half8*>(Qbh + (size_t)grow * HEAD_DIM + col);
      } else {
        v = half8{0, 0, 0, 0, 0, 0, 0, 0};
      }
      *reinterpret_cast<half8*>(&Q_lds[row * HEAD_DIM + col]) = v;
    }
  }
  __syncthreads();

  // ---- persistent online-softmax state ------------------------------
  float m_i[FRAGS_M][8];
  float l_i[FRAGS_M][8];
  float8 Oacc[FRAGS_M][HD_CHUNKS];
#pragma unroll
  for (int fm = 0; fm < FRAGS_M; ++fm) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      m_i[fm][e] = NEG_BIG;
      l_i[fm][e] = 0.0f;
    }
#pragma unroll
    for (int hc = 0; hc < HD_CHUNKS; ++hc) Oacc[fm][hc] = float8{0, 0, 0, 0, 0, 0, 0, 0};
  }

  const int kv_tiles = (S_pad + KV_TILE - 1) / KV_TILE;

  for (int kt = 0; kt < kv_tiles; ++kt) {
    int kv0 = kt * KV_TILE;
    bool needs_mask = (kv0 + KV_TILE) > S_real;  // true only on the boundary tile

    // ---- load K transposed into LDS (Kt_lds[d][s] = K[kv0+s][d]) ------
    {
      constexpr int total = KV_TILE * HEAD_DIM;
      for (int i = tid; i < total; i += NTHREADS) {
        int s = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int gs = kv0 + s;
        fp16_t v = (gs < S_pad) ? Kbh[(size_t)gs * HEAD_DIM + d] : (fp16_t)0.0f;
        Kt_lds[d][s] = v;
      }
    }
    // ---- load V naturally (kv, head_dim), vectorized, S_pad-masked ----
    {
      constexpr int total8 = (KV_TILE * HEAD_DIM) / 8;
      for (int i = tid; i < total8; i += NTHREADS) {
        int elem0 = i * 8;
        int row = elem0 / HEAD_DIM;
        int col = elem0 % HEAD_DIM;
        int gs = kv0 + row;
        half8 v;
        if (gs < S_pad) {
          v = *reinterpret_cast<const half8*>(Vbh + (size_t)gs * HEAD_DIM + col);
        } else {
          v = half8{0, 0, 0, 0, 0, 0, 0, 0};
        }
        *reinterpret_cast<half8*>(&V_lds[row * HEAD_DIM + col]) = v;
      }
    }
    __syncthreads();

    // ---- interleaved, 16-wide granularity: QK^T(ksub) -> online update ->
    // PV(ksub) -> next ksub. Keeps only one 8-float `scores` transient live
    // at a time instead of batching all KV_SUB=4 sub-tiles (see the file
    // header comment on why -- avoids the register spill the batched
    // version measured).
    for (int ksub = 0; ksub < KV_SUB; ++ksub) {
#pragma unroll
      for (int fm = 0; fm < FRAGS_M; ++fm) {
        // ---- QK^T for this 16-wide kv sub-tile only ----
        float8 acc = float8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kd = 0; kd < HD_CHUNKS; ++kd) {
          const fp16_t* aptr = &Q_lds[(wave_q_off + fm * 16) * HEAD_DIM + kd * 16];
          half8 a_frag = wmma_load_a(aptr, HEAD_DIM, lane);
          const fp16_t* bptr = &Kt_lds[kd * 16][ksub * 16];
          half8 b_frag = wmma_load_b(bptr, KT_LD, lane);
          acc = wmma_mma_f16(a_frag, b_frag, acc);
        }
        float scores[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float v = acc[e];
          if (needs_mask) {
            int kv_pos = kv0 + ksub * 16 + lw;
            if (kv_pos >= S_real) v = NEG_BIG;
          }
          scores[e] = v;
        }

        // ---- online softmax update (row = query, reduce over kv = lane&15) --
        float p[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float local_max = scores[e];
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) local_max = fmaxf(local_max, __shfl_xor(local_max, off, 16));

          float m_new = fmaxf(m_i[fm][e], local_max);
          float alpha = expf(m_i[fm][e] - m_new);

          float pv = expf(scores[e] - m_new);
          p[e] = pv;
          float local_sum = pv;
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) local_sum += __shfl_xor(local_sum, off, 16);

          l_i[fm][e] = l_i[fm][e] * alpha + local_sum;
          m_i[fm][e] = m_new;
#pragma unroll
          for (int hc = 0; hc < HD_CHUNKS; ++hc) Oacc[fm][hc][e] *= alpha;
        }

        // write this sub-tile's un-normalized probs to this wave's P_lds
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int row = fm * 16 + e + lg * 8;  // 0..(WAVE_M-1) within the wave
          P_lds[wave_id][row * KV_TILE + ksub * 16 + lw] = (fp16_t)p[e];
        }

        // ---- PV: fold this sub-tile's contribution in immediately -----
        const fp16_t* paptr = &P_lds[wave_id][(fm * 16) * KV_TILE + ksub * 16];
        half8 p_frag = wmma_load_a(paptr, KV_TILE, lane);
#pragma unroll
        for (int hc = 0; hc < HD_CHUNKS; ++hc) {
          const fp16_t* bptr = &V_lds[(ksub * 16) * HEAD_DIM + hc * 16];
          half8 v_frag = wmma_load_b(bptr, HEAD_DIM, lane);
          Oacc[fm][hc] = wmma_mma_f16(p_frag, v_frag, Oacc[fm][hc]);
        }
      }
    }

    __syncthreads();  // K/V LDS about to be overwritten by the next tile
  }

  // ---- epilogue: normalize by l_i, write O (S_pad-bounds masked) --------
#pragma unroll
  for (int fm = 0; fm < FRAGS_M; ++fm) {
#pragma unroll
    for (int hc = 0; hc < HD_CHUNKS; ++hc) {
      float8 f = Oacc[fm][hc];
      int col = hc * 16 + lw;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = q_base + wave_q_off + fm * 16 + e + lg * 8;
        if (row >= S_pad) continue;
        float l = l_i[fm][e];
        float v = (l > 0.0f) ? (f[e] / l) : 0.0f;
        Obh[(size_t)row * HEAD_DIM + col] = (fp16_t)v;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host launcher (ctypes ABI).
// Q/K/V/O are (BH, S_pad, 64) fp16, contiguous. Q must already have the
// attention scale (1/8) folded in by the caller (kernels_api.py does this).
// ---------------------------------------------------------------------------
extern "C" int launch_attention_encoder(const void* Q, const void* K,
                                         const void* V, void* O, int BH,
                                         int S_pad, int S_real, void* stream) {
  if (S_pad % 16 != 0) return -1001;
  dim3 grid((S_pad + Q_TILE - 1) / Q_TILE, BH);
  dim3 block(32, N_WAVES);
  attn_encoder_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
      (const fp16_t*)Q, (const fp16_t*)K, (const fp16_t*)V, (fp16_t*)O, S_pad,
      S_real);
  return (int)hipGetLastError();
}
