// layernorm.hip -- K1: plain LayerNorm and fused residual-add + LayerNorm.
// gfx1201 / RDNA4, wave32. fp32 math, fp16 in/out. One workgroup per row.
//
// VGPR / occupancy (fill in from -Rpass-analysis=kernel-resource-usage on
// gpu-host; see build log):
//   layernorm_kernel:                 VGPRs/lane = <see report>, waves = <see report>
//   fused_residual_layernorm_kernel:  VGPRs/lane = <see report>, waves = <see report>
// Both kernels are simple (no WMMA, no large live ranges across a loop), so
// they are expected to land at or near the 16-wave / 100% occupancy step
// (<=96 VGPR/lane) -- this is a bandwidth-bound kernel, occupancy just needs
// to be enough to hide the global load/store latency, not maximized.
//
// Block size: 256 threads (8 wave32 waves) per row of COLS=1280 -> 5
// elements/thread. Two-pass mean/var (pass 1: mean; pass 2: E[(x-mean)^2]),
// both reductions via common.h's block_reduce_sum (wave shuffle + LDS).

#include "common.h"

#define LN_COLS 1280
#define LN_THREADS 256
#define LN_PER_THREAD (LN_COLS / LN_THREADS)  // 5

// ---------------------------------------------------------------------------
// Plain LayerNorm: out = (x - mean) / sqrt(var + eps) * gamma + beta
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(LN_THREADS) void layernorm_kernel(
    const fp16_t* __restrict__ x, const fp16_t* __restrict__ gamma,
    const fp16_t* __restrict__ beta, fp16_t* __restrict__ out, int rows,
    int cols, float eps) {
  __shared__ float scratch[LN_THREADS / 32];
  int row = blockIdx.x;
  if (row >= rows) return;
  const fp16_t* xrow = x + (size_t)row * cols;
  fp16_t* orow = out + (size_t)row * cols;

  float local[LN_PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < LN_PER_THREAD; ++i) {
    int c = threadIdx.x + i * LN_THREADS;
    float v = (c < cols) ? (float)xrow[c] : 0.0f;
    local[i] = v;
    sum += v;
  }
  float mean = block_reduce_sum(sum, scratch) / (float)cols;

  float sqsum = 0.0f;
#pragma unroll
  for (int i = 0; i < LN_PER_THREAD; ++i) {
    int c = threadIdx.x + i * LN_THREADS;
    if (c < cols) {
      float d = local[i] - mean;
      sqsum += d * d;
    }
  }
  float var = block_reduce_sum(sqsum, scratch) / (float)cols;
  float rstd = rsqrtf(var + eps);

#pragma unroll
  for (int i = 0; i < LN_PER_THREAD; ++i) {
    int c = threadIdx.x + i * LN_THREADS;
    if (c < cols) {
      float g = (float)gamma[c];
      float b = (float)beta[c];
      float y = (local[i] - mean) * rstd * g + b;
      orow[c] = (fp16_t)y;
    }
  }
}

// ---------------------------------------------------------------------------
// Fused residual-add + LayerNorm: s = residual + y; out_normed = LN(s);
// out_residual = s (fp16) -- the updated residual stream, so callers that
// need to keep accumulating (e.g. the conv-stem + positional-embedding add
// feeding encoder layer 0's first LN) get it in the same kernel instead of a
// separate elementwise-add + 2 barriers.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(LN_THREADS) void fused_residual_layernorm_kernel(
    const fp16_t* __restrict__ residual, const fp16_t* __restrict__ y,
    const fp16_t* __restrict__ gamma, const fp16_t* __restrict__ beta,
    fp16_t* __restrict__ out_normed, fp16_t* __restrict__ out_residual,
    int rows, int cols, float eps) {
  __shared__ float scratch[LN_THREADS / 32];
  int row = blockIdx.x;
  if (row >= rows) return;
  const fp16_t* rrow = residual + (size_t)row * cols;
  const fp16_t* yrow = y + (size_t)row * cols;
  fp16_t* onrow = out_normed + (size_t)row * cols;
  fp16_t* orrow = out_residual + (size_t)row * cols;

  float local[LN_PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < LN_PER_THREAD; ++i) {
    int c = threadIdx.x + i * LN_THREADS;
    float v = 0.0f;
    if (c < cols) {
      v = (float)rrow[c] + (float)yrow[c];
      orrow[c] = (fp16_t)v;  // updated residual stream, written once
    }
    local[i] = v;
    sum += v;
  }
  float mean = block_reduce_sum(sum, scratch) / (float)cols;

  float sqsum = 0.0f;
#pragma unroll
  for (int i = 0; i < LN_PER_THREAD; ++i) {
    int c = threadIdx.x + i * LN_THREADS;
    if (c < cols) {
      float d = local[i] - mean;
      sqsum += d * d;
    }
  }
  float var = block_reduce_sum(sqsum, scratch) / (float)cols;
  float rstd = rsqrtf(var + eps);

#pragma unroll
  for (int i = 0; i < LN_PER_THREAD; ++i) {
    int c = threadIdx.x + i * LN_THREADS;
    if (c < cols) {
      float g = (float)gamma[c];
      float b = (float)beta[c];
      float yv = (local[i] - mean) * rstd * g + b;
      onrow[c] = (fp16_t)yv;
    }
  }
}

// ---------------------------------------------------------------------------
// Host launchers (ctypes ABI: raw pointers + hipStream_t, return hipError_t)
// ---------------------------------------------------------------------------
extern "C" int launch_layernorm(const void* x, const void* gamma,
                                 const void* beta, void* out, int rows,
                                 int cols, float eps, void* stream) {
  dim3 grid(rows);
  dim3 block(LN_THREADS);
  layernorm_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
      (const fp16_t*)x, (const fp16_t*)gamma, (const fp16_t*)beta,
      (fp16_t*)out, rows, cols, eps);
  return (int)hipGetLastError();
}

extern "C" int launch_fused_residual_layernorm(const void* residual,
                                                const void* y,
                                                const void* gamma,
                                                const void* beta,
                                                void* out_normed,
                                                void* out_residual, int rows,
                                                int cols, float eps,
                                                void* stream) {
  dim3 grid(rows);
  dim3 block(LN_THREADS);
  fused_residual_layernorm_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
      (const fp16_t*)residual, (const fp16_t*)y, (const fp16_t*)gamma,
      (const fp16_t*)beta, (fp16_t*)out_normed, (fp16_t*)out_residual, rows,
      cols, eps);
  return (int)hipGetLastError();
}
