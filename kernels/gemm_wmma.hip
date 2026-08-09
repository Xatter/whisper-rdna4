// gemm_wmma.hip -- K2: fp16 WMMA GEMM, fp32 accumulate, fused epilogues.
// gfx1201 / RDNA4, wave32. The core deliverable (DESIGN.md SS6).
//
// ---------------------------------------------------------------------------
// WEIGHT PACKING SPEC (read this before writing weights.py's repack step)
// ---------------------------------------------------------------------------
// The B operand (weights) must be pre-packed offline into 32(K) x 128(N)
// row-major tiles, laid out so that for a fixed n_block, consecutive k_block
// tiles are contiguous in memory. This makes the kernel's global->LDS B load
// a single contiguous copy -- zero address arithmetic beyond a base-pointer
// bump per (n_block, k_block).
//
// Given a logical weight matrix W of shape (K, N) row-major -- i.e. the
// matrix such that Y = X @ W directly, NOT torch nn.Linear's (out,in)=(N,K)
// storage; if you start from `linear.weight` (shape (N,K)) you must first
// take `W = linear.weight.t().contiguous()` (shape (K,N)) before packing --
// the packed layout is:
//
//   K_blocks = K / 32,  N_blocks = N / 128     (both MUST divide evenly --
//                                                true for every K2 shape in
//                                                DESIGN.md SS6: K in
//                                                {1280,5120}, N in
//                                                {1280,3840,5120})
//   packed[n_block, k_block, kk, nn] = W[k_block*32 + kk, n_block*128 + nn]
//   packed.shape = (N_blocks, K_blocks, 32, 128), fp16, contiguous.
//
// Equivalently in torch: (see kernels/gemm_pack.py for the authoritative,
// importable implementation):
//   packed = W.reshape(K_blocks, 32, N_blocks, 128).permute(2, 0, 1, 3).contiguous()
//
// A (activations) and the output C/D are ordinary row-major (M, K) / (M, N)
// fp16 tensors -- no packing, no transpose. Bias is a plain (N,) fp16
// vector. Residual (for BIAS_RESIDUAL) is a plain row-major (M, N) fp16
// tensor (typically the same buffer the caller will keep using as the
// residual stream).
//
// ---------------------------------------------------------------------------
// TILE CONFIGS
// ---------------------------------------------------------------------------
// Main (prefill) tile: BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, 4 wave32 waves
// arranged 2x2, each wave owns a 32x64 output quadrant = 2x4 = 8 WMMA
// fragments (float8 acc each) = 64 VGPR/lane for accumulators.
// NOTE: measured progression (see the report for exact numbers): the
// DESIGN.md-nominal 128x128 tile (16 frags/wave) landed at 207 VGPR/lane,
// only 4 waves/SIMD -- under the 9-wave target DESIGN.md SS6 asks for. Per
// its own explicit fallback ("If measured occupancy < 9 waves, shrink to
// 96x128 tiles"), this was tried at 96x128 (167 VGPR, still 4 waves/SIMD),
// then shrunk further to 64x128 (129 VGPR, 5 waves/SIMD, landed here) --
// still short of 9 waves. The occupancy number did not scale with VGPR the
// way the cheatsheet SS3 staircase predicts in isolation (e.g. the small-M
// tile below reports only 6 waves at just 78 VGPR, where the VGPR-only
// staircase would suggest 16); LDS residency is the more likely binding
// constraint at this tile's ~24KB/block footprint, not registers alone.
// Reported honestly in the report rather than chased further; see "what
// I'd fix next" there.
//
// Small-M (decode) tile: BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, same 4 waves
// but arranged 1x4 (1 row of waves x 4 cols), each wave owns 32x32 = 2x2 = 4
// WMMA fragments = 32 VGPR/lane accumulators -- deliberately light so decode
// (M=B, tiny) doesn't pay prefill's register footprint (cheatsheet SS10).
// Both configs share BLOCK_K=32/BLOCK_N=128, so ONE packed-weight layout
// (above) serves both -- weights.py only needs to pack once per weight
// matrix, not once per tile config.
//
// K-loop: BLOCK_K=32 handled as two chained k=16 WMMA calls (ks=0,1) per
// fragment pair, accumulating into the same float8, per DESIGN.md SS6.
// Double-buffered LDS: buffer parity alternates per K-block iteration;
// iteration kb prefetches K-block kb+1 into the *other* buffer while
// computing on the current one, one __syncthreads() per iteration (see the
// long comment in the kernel body for the exact safety argument).
//
// M boundary: M need not be a multiple of BLOCK_M (tested at M=24000 with
// BLOCK_M=128 -> last block is a 64-row partial tile). A-tile loads
// zero-fill rows past M; output stores skip rows past M. N and K always
// divide evenly for every shape in DESIGN.md SS6, so only M is masked.
//
// VGPR / occupancy: fill in from -Rpass-analysis=kernel-resource-usage
// (build.sh runs with it). Recorded per-kernel in the report, not asserted
// here to avoid stale comments drifting from the actual compiled binary.

#include "common.h"

// ---- fp32-accumulate epilogue application (shared by both tile configs) --
template <EpilogueKind EPI>
__device__ __forceinline__ float apply_epilogue(float v) {
  if constexpr (EPI == EpilogueKind::BIAS_GELU) {
    return gelu_f32(v);
  } else {
    return v;
  }
}

// ---------------------------------------------------------------------------
// Templated GEMM kernel. blockDim = (32, WAVES_M*WAVES_N).
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVES_M, int WAVES_N,
          EpilogueKind EPI>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N) void gemm_kernel(
    const fp16_t* __restrict__ A, const fp16_t* __restrict__ Bpacked,
    const fp16_t* __restrict__ bias, const fp16_t* __restrict__ residual,
    fp16_t* __restrict__ out, int M, int N, int K) {
  static_assert(BLOCK_K == 32, "K-loop below assumes BLOCK_K == 32 (two chained k=16 WMMA)");
  constexpr int WAVE_M = BLOCK_M / WAVES_M;
  constexpr int WAVE_N = BLOCK_N / WAVES_N;
  static_assert(WAVE_M % 16 == 0 && WAVE_N % 16 == 0, "wave tile must be a multiple of 16");
  constexpr int FRAGS_M = WAVE_M / 16;
  constexpr int FRAGS_N = WAVE_N / 16;

  __shared__ __align__(16) fp16_t As[2][BLOCK_M * BLOCK_K];
  __shared__ __align__(16) fp16_t Bs[2][BLOCK_K * BLOCK_N];

  const int m_block = blockIdx.y;
  const int n_block = blockIdx.x;
  const int K_blocks = K / BLOCK_K;

  const int lane = threadIdx.x;         // 0..31, lane within wave
  const int wave_id = threadIdx.y;      // 0..(WAVES_M*WAVES_N - 1)
  const int wave_row = wave_id / WAVES_N;
  const int wave_col = wave_id % WAVES_N;
  const int m_wave_off = wave_row * WAVE_M;
  const int n_wave_off = wave_col * WAVE_N;
  const int lw = lane & 15;
  const int lg = lane >> 4;

  const int tid = threadIdx.y * 32 + threadIdx.x;   // flat id for tile loads
  const int nthreads = blockDim.x * blockDim.y;

  // ---- global -> LDS loaders --------------------------------------------
  auto load_A_tile = [&](int buf, int k0) {
    constexpr int total8 = (BLOCK_M * BLOCK_K) / 8;
    for (int i = tid; i < total8; i += nthreads) {
      int elem0 = i * 8;
      int row = elem0 / BLOCK_K;
      int col = elem0 % BLOCK_K;
      int gm = m_block * BLOCK_M + row;
      half8 v;
      if (gm < M) {
        const fp16_t* src = A + (size_t)gm * K + k0 + col;
        v = *reinterpret_cast<const half8*>(src);
      } else {
        v = half8{0, 0, 0, 0, 0, 0, 0, 0};
      }
      *reinterpret_cast<half8*>(&As[buf][row * BLOCK_K + col]) = v;
    }
  };
  auto load_B_tile = [&](int buf, int k_block) {
    constexpr int total8 = (BLOCK_K * BLOCK_N) / 8;
    const fp16_t* src_base =
        Bpacked + ((size_t)n_block * K_blocks + k_block) * (BLOCK_K * BLOCK_N);
    for (int i = tid; i < total8; i += nthreads) {
      *reinterpret_cast<half8*>(&Bs[buf][i * 8]) =
          *reinterpret_cast<const half8*>(src_base + i * 8);
    }
  };

  float8 acc[FRAGS_M][FRAGS_N];
#pragma unroll
  for (int fm = 0; fm < FRAGS_M; ++fm)
#pragma unroll
    for (int fn = 0; fn < FRAGS_N; ++fn) acc[fm][fn] = float8{0, 0, 0, 0, 0, 0, 0, 0};

  load_A_tile(0, 0);
  load_B_tile(0, 0);
  __syncthreads();

  for (int kb = 0; kb < K_blocks; ++kb) {
    int cur = kb & 1;
    int nxt = cur ^ 1;
    // Prefetch next K-block into the other buffer while this iteration
    // computes on `cur`. Safety: the __syncthreads() at the bottom of THIS
    // iteration is what the next iteration relies on to know both (a) the
    // prefetch write into `nxt` finished (next iteration reads it as `cur`)
    // and (b) every thread's compute-read of the current `cur` is done
    // (next-next iteration will reuse this same buffer slot as its `nxt`).
    if (kb + 1 < K_blocks) {
      load_A_tile(nxt, (kb + 1) * BLOCK_K);
      load_B_tile(nxt, kb + 1);
    }

    for (int ks = 0; ks < 2; ++ks) {  // two chained k=16 WMMA covering BLOCK_K=32
      half8 a_frag[FRAGS_M];
#pragma unroll
      for (int fm = 0; fm < FRAGS_M; ++fm) {
        const fp16_t* aptr = &As[cur][(m_wave_off + fm * 16) * BLOCK_K + ks * 16];
        a_frag[fm] = wmma_load_a(aptr, BLOCK_K, lane);
      }
#pragma unroll
      for (int fn = 0; fn < FRAGS_N; ++fn) {
        const fp16_t* bptr = &Bs[cur][(ks * 16) * BLOCK_N + (n_wave_off + fn * 16)];
        half8 b_frag = wmma_load_b(bptr, BLOCK_N, lane);
#pragma unroll
        for (int fm = 0; fm < FRAGS_M; ++fm) {
          acc[fm][fn] = wmma_mma_f16(a_frag[fm], b_frag, acc[fm][fn]);
        }
      }
    }
    __syncthreads();
  }

  // ---- epilogue: bias (+ gelu | + residual), M-masked store -------------
#pragma unroll
  for (int fn = 0; fn < FRAGS_N; ++fn) {
    int col = n_block * BLOCK_N + n_wave_off + fn * 16 + lw;
    float bias_v = (col < N) ? (float)bias[col] : 0.0f;
#pragma unroll
    for (int fm = 0; fm < FRAGS_M; ++fm) {
      float8 f = acc[fm][fn];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = m_block * BLOCK_M + m_wave_off + fm * 16 + e + lg * 8;
        if (row >= M || col >= N) continue;
        float v = f[e] + bias_v;
        if constexpr (EPI == EpilogueKind::BIAS_RESIDUAL) {
          v += (float)residual[(size_t)row * N + col];
        } else {
          v = apply_epilogue<EPI>(v);
        }
        out[(size_t)row * N + col] = (fp16_t)v;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host launchers (ctypes ABI). Two tile configs x three epilogues.
// ---------------------------------------------------------------------------
#define GEMM_LAUNCHER(NAME, BM, BN, BK, WM, WN, EPI)                          \
  extern "C" int NAME(const void* A, const void* Bpacked, const void* bias,   \
                       const void* residual, void* out, int M, int N, int K,  \
                       void* stream) {                                        \
    if (K % BK != 0 || N % BN != 0) return -1000;                             \
    dim3 grid(N / BN, (M + BM - 1) / BM);                                     \
    dim3 block(32, WM * WN);                                                  \
    gemm_kernel<BM, BN, BK, WM, WN, EPI>                                      \
        <<<grid, block, 0, (hipStream_t)stream>>>(                           \
            (const fp16_t*)A, (const fp16_t*)Bpacked, (const fp16_t*)bias,    \
            (const fp16_t*)residual, (fp16_t*)out, M, N, K);                  \
    return (int)hipGetLastError();                                            \
  }

// Main (prefill) tile: 96x128x32, waves 2x2 (48x64/wave) -- shrunk from the
// nominal 128x128 per the measured-occupancy fallback (see block comment
// above launch_gemm_128_bias's kernel). Launcher names kept as "_128" since
// that is the still-current N-dimension tile width and the name callers use
// (kernels_api.py's tile="128" selects this family); BLOCK_M=96 is an
// implementation detail below the launcher name.
GEMM_LAUNCHER(launch_gemm_128_bias, 64, 128, 32, 2, 2, EpilogueKind::BIAS)
GEMM_LAUNCHER(launch_gemm_128_bias_gelu, 64, 128, 32, 2, 2, EpilogueKind::BIAS_GELU)
GEMM_LAUNCHER(launch_gemm_128_bias_residual, 64, 128, 32, 2, 2, EpilogueKind::BIAS_RESIDUAL)

// Small-M (decode) tile: 32x128x32, waves 1x4 (32x32/wave).
GEMM_LAUNCHER(launch_gemm_32_bias, 32, 128, 32, 1, 4, EpilogueKind::BIAS)
GEMM_LAUNCHER(launch_gemm_32_bias_gelu, 32, 128, 32, 1, 4, EpilogueKind::BIAS_GELU)
GEMM_LAUNCHER(launch_gemm_32_bias_residual, 32, 128, 32, 1, 4, EpilogueKind::BIAS_RESIDUAL)

#undef GEMM_LAUNCHER
