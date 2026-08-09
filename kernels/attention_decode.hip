// attention_decode.hip
//
// K4 — decode-step attention (DESIGN.md section 6/7), owned by Agent C.
//
// Two entry points:
//   launch_self_attn_decode  — append new K/V into the fp16 cache, then
//                              softmax(q.K^T / 8).V causal-decode over t<=step.
//   launch_cross_attn_decode — same math, no append, fixed T=1500, reads the
//                              precomputed cross-attention K/V cache.
//
// Design (see DECODE_INTEGRATION.md / final report for the full rationale
// and for the microbench history that motivated the multi-wave revision
// below):
//   - One workgroup per (batch b, head h) pair. Grid = dim3(H, B).
//   - head_dim (Dh) is fixed at 64 for whisper large-v3-turbo, so within any
//     one wave, each lane owns exactly 2 head_dim elements: d = lane and
//     d = lane + 32. This is what makes both the K/V cache reads AND the
//     new-K/V append fully coalesced: for a fixed t, all 32 lanes of a wave
//     touch 32 CONSECUTIVE fp16 elements (the innermost, contiguous dim of
//     the [.., T, 64] cache layout) in one shot, twice (low half / high
//     half).
//   - A first version of this kernel used exactly ONE wave (32 threads) per
//     workgroup, serially walking t=0..T-1. It passed correctness at every
//     shape but LOST to the torch op sequence at B=8/cross-attn (T=1500):
//     0.51x. Root cause: at B=8, B*H=160 workgroups of 1 wave each only
//     fill 160 of the 1024 wave slots the card has (64 CU x 16 waves), and
//     a single wave walking 1500 sequential t-iterations has only one
//     outstanding coalesced load in flight at a time -- not enough
//     memory-level parallelism to hit the bandwidth roof, and not enough
//     concurrent workgroups to hide it via occupancy either. See the
//     microbench table in the final report for before/after numbers.
//   - Fix: split each (b,h) workgroup's T range across NWAVES=4 waves
//     (128 threads/workgroup). Each wave computes scores/PV-partials for
//     its own contiguous T-chunk independently (same coalesced-per-t
//     pattern as before, just NWAVES-way parallel), writing into disjoint
//     regions of a shared LDS score row -- no cross-wave contention there.
//     Softmax stats (max, sum) need one honest cross-wave combine: each
//     wave reduces its LDS-strided subset with the same intra-wave shuffle
//     trick as before, one lane per wave (lane 0) stashes the per-wave
//     partial into a tiny LDS array (<=8 floats), then EVERY thread does a
//     redundant NWAVES-way scalar combine (cheap: NWAVES=4 iterations, no
//     further sync needed since it only reads the tiny array). PV partials
//     are combined the same way, once, at the very end, via a
//     [NWAVES][32]-shaped LDS scratch that only wave 0 reads back to
//     produce the final 64-wide output.
//   - This degrades gracefully to something close to the original 1-wave
//     kernel when T is tiny (e.g. self-attn step=0, T=1): waves 1..3 simply
//     get an empty chunk and fall through to the (cheap) reduction/sync
//     steps, so there is no correctness hazard, only a few extra
//     __syncthreads() of pure overhead at the smallest T.
//   - All accumulation (dot products, running max/sum, PV accumulate) is
//     fp32, converting to/from fp16 only at cache read/write boundaries, per
//     the cheatsheet's stateful/recurrent-kernel warning (section 13.3):
//     numeric drift across 448 decode steps must be bounded by matching the
//     torch reference's accumulation order/precision, not just "close
//     enough" per step.
//   - No WMMA anywhere in this file: M=1 per sequence per step means the
//     matrix cores are idle at this shape regardless (cheatsheet section 6),
//     so this is a pure wave32 vector/LDS kernel by design -- "multi-wave"
//     here means multiple wave32 waves cooperating in one workgroup, NOT
//     wave64. Every intra-wave reduction below is an explicit 32-lane
//     __shfl_xor tree.
//
// current_len / step convention (documented per the task's requirement):
//   DESIGN.md section 7 runs decode in lock-step across all B sequences in
//   a batch (every sequence advances exactly one position per host-side
//   loop iteration, and finished sequences are just masked/ignored downstream
//   rather than dropped from the batch). Because of that lock-step
//   invariant, a single host-side integer step counter identifies the
//   current write/attend position for EVERY sequence in the batch
//   simultaneously — there is no per-sequence divergence in cache length to
//   track, so passing `step` by value (instead of a device-side int32
//   tensor read back or a per-sequence length array) is exact, not an
//   approximation, and avoids a host<->device sync that a real per-sequence
//   device-side length would otherwise force if the host ever needed to
//   branch on it. `step` is 0-indexed: step=0 is the first generated token.
//   The kernel appends the new K/V at cache position `step` and then
//   attends over t in [0, step] inclusive (T_valid = step + 1).
//
// Coalescing note: this file does not depend on Agent B's common.h (it did
// not exist yet when this file was written — see final report). Everything
// needed is defined locally.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVE 32
#define HEAD_DIM 64
#define ATTN_SCALE 0.125f            // 1 / sqrt(64)
#define NWAVES 4                     // waves cooperating per (b,h) workgroup
#define BLOCK_THREADS (NWAVES * WAVE)

// Compile-time LDS capacities. Runtime T_max / T_cross must be <= these.
#define SELF_T_CAP 448
#define CROSS_T_CAP 1500

// ---------------------------------------------------------------------
// Intra-wave (32-lane) butterfly reductions. Named dec_* (not
// wave_reduce_sum) to avoid any ambiguity with Agent B's common.h helper of
// the same name once both files link into one .so -- this file
// intentionally does not include common.h (see file header), so there is
// no compile-time collision, but device-scope __forceinline__ functions
// are per-translation-unit anyway; the rename is purely so a human reading
// both files side by side isn't confused about which one runs.
// ---------------------------------------------------------------------
__device__ __forceinline__ float dec_wave_reduce_sum(float v) {
#pragma unroll
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, WAVE);
    }
    return v;
}

__device__ __forceinline__ float dec_wave_reduce_max(float v) {
#pragma unroll
    for (int offset = WAVE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(v, offset, WAVE);
        v = other > v ? other : v;
    }
    return v;
}

// ---------------------------------------------------------------------
// Core decode-attention body, shared by the self-attn and cross-attn
// entry points. `score` is a caller-provided LDS array with capacity
// >= T. `T` is the number of valid key/value positions to attend over
// (runtime value, <= the LDS capacity of `score`). Must be launched with
// blockDim.x == BLOCK_THREADS (NWAVES full wave32 waves).
//
// q_row, k_base, v_base, out_row all point at the first of the 64
// contiguous fp16 elements for this (b, h) pair (i.e. head_dim is
// innermost / contiguous, matching the DESIGN.md cache layout).
// k_base/v_base point at t=0 of a [T, HEAD_DIM] fp16 region (a single
// (b,h) slab of the full [B,H,T_max_or_cross,Dh] cache).
// ---------------------------------------------------------------------
__device__ __forceinline__ void decode_attn_core(
    const __half* __restrict__ q_row,
    const __half* __restrict__ k_base,
    const __half* __restrict__ v_base,
    __half* __restrict__ out_row,
    float* __restrict__ score,
    int T)
{
    const int tid = threadIdx.x;         // 0..BLOCK_THREADS-1
    const int lane = tid & (WAVE - 1);   // 0..31 within this thread's wave
    const int wave = tid >> 5;           // 0..NWAVES-1

    // Load and scale q once per lane (folds the 1/8 scale into q instead of
    // paying it T times in the score loop -- cheatsheet section 15 style
    // fusion). Every wave redundantly loads/scales the same q -- 2 cheap
    // reads, not worth avoiding via LDS staging.
    float q0 = __half2float(q_row[lane]) * ATTN_SCALE;
    float q1 = __half2float(q_row[lane + WAVE]) * ATTN_SCALE;

    // This wave's contiguous T-chunk.
    const int chunk = (T + NWAVES - 1) / NWAVES;
    const int t_start = wave * chunk;
    const int t_end = min(t_start + chunk, T);

    // ---- Pass 1: cooperative Q.K^T over this wave's t-chunk, coalesced K reads ----
    for (int t = t_start; t < t_end; ++t) {
        const __half* krow = k_base + (size_t)t * HEAD_DIM;
        float k0 = __half2float(krow[lane]);
        float k1 = __half2float(krow[lane + WAVE]);
        float partial = q0 * k0 + q1 * k1;
        float dot = dec_wave_reduce_sum(partial);
        if (lane == 0) {
            score[t] = dot;
        }
    }
    __syncthreads();  // publish every wave's slice of `score` to the block

    // ---- Softmax stats from the LDS score row ----
    // Stage A: each wave reduces a block-strided subset (touches LDS only).
    float local_max = -INFINITY;
    for (int t = tid; t < T; t += BLOCK_THREADS) {
        float s = score[t];
        local_max = s > local_max ? s : local_max;
    }
    float wave_local_max = dec_wave_reduce_max(local_max);

    __shared__ float wmax[NWAVES];
    __shared__ float wsum[NWAVES];
    if (lane == 0) wmax[wave] = wave_local_max;
    __syncthreads();

    // Stage B: NWAVES is tiny (4) -- every thread just scans it directly,
    // cheaper than another shuffle/sync round.
    float row_max = wmax[0];
#pragma unroll
    for (int w = 1; w < NWAVES; ++w) {
        row_max = wmax[w] > row_max ? wmax[w] : row_max;
    }

    float local_sum = 0.0f;
    for (int t = tid; t < T; t += BLOCK_THREADS) {
        local_sum += expf(score[t] - row_max);
    }
    float wave_local_sum = dec_wave_reduce_sum(local_sum);
    if (lane == 0) wsum[wave] = wave_local_sum;
    __syncthreads();

    float row_sum = 0.0f;
#pragma unroll
    for (int w = 0; w < NWAVES; ++w) {
        row_sum += wsum[w];
    }
    float inv_sum = 1.0f / row_sum;

    // ---- Pass 2: cooperative P.V over this wave's t-chunk, coalesced V reads ----
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int t = t_start; t < t_end; ++t) {
        const __half* vrow = v_base + (size_t)t * HEAD_DIM;
        float v0 = __half2float(vrow[lane]);
        float v1 = __half2float(vrow[lane + WAVE]);
        float wgt = expf(score[t] - row_max) * inv_sum;
        acc0 += wgt * v0;
        acc1 += wgt * v1;
    }

    // Combine each wave's partial PV accumulator across NWAVES via LDS;
    // only wave 0 needs to produce the final output.
    __shared__ float acc0w[NWAVES][WAVE];
    __shared__ float acc1w[NWAVES][WAVE];
    acc0w[wave][lane] = acc0;
    acc1w[wave][lane] = acc1;
    __syncthreads();

    if (wave == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
#pragma unroll
        for (int w = 0; w < NWAVES; ++w) {
            sum0 += acc0w[w][lane];
            sum1 += acc1w[w][lane];
        }
        out_row[lane] = __float2half(sum0);
        out_row[lane + WAVE] = __float2half(sum1);
    }
}

// ---------------------------------------------------------------------
// Self-attention decode step kernel.
//
// q, k_new, v_new, out: [B, H, HEAD_DIM] fp16, contiguous.
// k_cache, v_cache:     [B, H, T_max, HEAD_DIM] fp16, contiguous, mutated
//                        in place (append at position `step`).
// ---------------------------------------------------------------------
__global__ void self_attn_decode_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_new,
    const __half* __restrict__ v_new,
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    __half* __restrict__ out,
    int B, int H, int step, int T_max)
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & (WAVE - 1);
    const int wave = tid >> 5;
    if (b >= B || h >= H) return;

    const size_t bh = (size_t)b * H + h;
    const __half* q_row = q + bh * HEAD_DIM;
    const __half* knew_row = k_new + bh * HEAD_DIM;
    const __half* vnew_row = v_new + bh * HEAD_DIM;
    __half* k_slab = k_cache + bh * (size_t)T_max * HEAD_DIM;
    __half* v_slab = v_cache + bh * (size_t)T_max * HEAD_DIM;
    __half* out_row = out + bh * HEAD_DIM;

    // ---- Append new K/V at cache position `step` (coalesced, 2x32) ----
    // Only wave 0 does this; the other NWAVES-1 waves would just write the
    // identical bytes (harmless but wasted traffic), so gate it.
    if (wave == 0) {
        __half* k_dst = k_slab + (size_t)step * HEAD_DIM;
        __half* v_dst = v_slab + (size_t)step * HEAD_DIM;
        k_dst[lane] = knew_row[lane];
        k_dst[lane + WAVE] = knew_row[lane + WAVE];
        v_dst[lane] = vnew_row[lane];
        v_dst[lane + WAVE] = vnew_row[lane + WAVE];
    }
    __syncthreads();  // make the append visible to every wave before any reads it back

    __shared__ float score[SELF_T_CAP];
    const int T = step + 1;  // attend over [0, step] inclusive
    decode_attn_core(q_row, k_slab, v_slab, out_row, score, T);
}

extern "C" int launch_self_attn_decode(
    const void* q, const void* k_new, const void* v_new,
    void* k_cache, void* v_cache, void* out,
    int B, int H, int step, int T_max, void* stream)
{
    if (step < 0 || step >= T_max) return (int)hipErrorInvalidValue;
    dim3 grid(H, B, 1);
    dim3 block(BLOCK_THREADS, 1, 1);
    self_attn_decode_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        (const __half*)q, (const __half*)k_new, (const __half*)v_new,
        (__half*)k_cache, (__half*)v_cache, (__half*)out,
        B, H, step, T_max);
    return (int)hipGetLastError();
}

// ---------------------------------------------------------------------
// Cross-attention decode step kernel (no append; T fixed by T_cross).
//
// q, out:              [B, H, HEAD_DIM] fp16, contiguous.
// k_cache_x, v_cache_x: [B, H, T_cross, HEAD_DIM] fp16, precomputed once
//                        per chunk, read-only here.
// ---------------------------------------------------------------------
__global__ void cross_attn_decode_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache_x,
    const __half* __restrict__ v_cache_x,
    __half* __restrict__ out,
    int B, int H, int T_cross)
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (b >= B || h >= H) return;

    const size_t bh = (size_t)b * H + h;
    const __half* q_row = q + bh * HEAD_DIM;
    const __half* k_slab = k_cache_x + bh * (size_t)T_cross * HEAD_DIM;
    const __half* v_slab = v_cache_x + bh * (size_t)T_cross * HEAD_DIM;
    __half* out_row = out + bh * HEAD_DIM;

    __shared__ float score[CROSS_T_CAP];
    decode_attn_core(q_row, k_slab, v_slab, out_row, score, T_cross);
}

extern "C" int launch_cross_attn_decode(
    const void* q, const void* k_cache_x, const void* v_cache_x, void* out,
    int B, int H, int T_cross, void* stream)
{
    if (T_cross <= 0 || T_cross > CROSS_T_CAP) return (int)hipErrorInvalidValue;
    dim3 grid(H, B, 1);
    dim3 block(BLOCK_THREADS, 1, 1);
    cross_attn_decode_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        (const __half*)q, (const __half*)k_cache_x, (const __half*)v_cache_x,
        (__half*)out, B, H, T_cross);
    return (int)hipGetLastError();
}
