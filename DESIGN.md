# rocm-whisper: Whisper large-v3-turbo on 2x AMD R9700 (RDNA4 / gfx1201)

Design document. Author: orchestrator (Fable). Implementors and reviewers: read this fully
before touching code. The architecture facts below are sourced from
`/Users/user/code/amd-r9700-docs/` — cite-worthy files are referenced inline.

## 1. Goal and baseline

Transcribe long audio (podcast corpus) with Whisper large-v3-turbo as fast as possible on
one machine: **gpu-host** (`user@gpu-host`), 2x AMD Radeon AI PRO R9700 (gfx1201,
32 GB, 640 GB/s, 64 CU), Ryzen 5 9600X, 30 GB RAM, ROCm 7.2 at `/opt/rocm`.

**Baseline to beat:** insanely-fast-whisper-rocm (HF transformers pipeline, fp16)
previously measured on this exact machine: **overall RTF 0.0488** (≈20.5x real time)
over a 5-file, 29 578 s (8.2 h) corpus at
`~/insanely-fast-whisper-rocm/bench-audio/` (same files as `~/bench_audio`
plus one). Its per-file transcripts are in `bench-results/*.json` — these are our
quality reference.

**Baseline actual configuration** (verified by adversarial review R2 reading the
baseline source — corrects earlier assumptions in this document): the repo externally
pre-splits audio into non-overlapping 30 s chunk files (`chunk_overlap=0.0` — NO
stride, despite HF's usual default) and passes them to the HF pipeline one at a time,
so its **effective batch size is 1** (`batch_size=4` is specified but never engages).
Greedy, temperature 0, language en, task transcribe, `no_repeat_ngram_size=3`,
`return_timestamps=True` (chunk timestamps → ~19% more decode tokens than a
notimestamps run). Its JSON metadata mislabels the GPUs as "7900 XTX x2"; the run
provably executed on the R9700s (HSA_OVERRIDE_GFX_VERSION=12.0.0 in run_bench.sh).
Fair-comparison consequences: (a) our hard-cut 30 s no-overlap mode IS the
apples-to-apples chunking; (b) headline timing compares our wall (post-model-load,
audio I/O included) against baseline `wall_time_seconds` (same boundaries); (c) the
report must caveat the timestamp-token difference and state that batching is the
primary architectural lever; (d) our decode needs `no_repeat_ngram_size=3` parity
(GPU-side, in K5) before quality divergence numbers are meaningful.

**Success:** ≥2x faster than baseline (RTF ≤ 0.024) with transcript quality parity
(WER vs baseline transcripts ≤ ~3%, spot-checked sanity). Stretch: ≥4x with dual GPU.

## 2. Why the baseline is slow — the three structural levers

The HF pipeline at batch=4 leaves most of the machine idle:

1. **Batch tiling.** The encoder is pure prefill: 1500 mel frames per 30 s chunk,
   arithmetic intensity grows with M = B×1500. At batch 4 the GEMMs are too narrow to
   saturate 64 CUs of WMMA; at B=16–32 per GPU (M=24k–48k) every linear layer is a
   dense, matrix-core-bound GEMM. VRAM permits B=64+ (weights 1.6 GB fp16; activations
   ~0.5 GB at B=32). The card's 2nd-gen "AI accelerator" WMMA units only pay off
   at exactly this regime (cheatsheet §6: batch=1 decode is 2 ops/byte, two orders of
   magnitude below the ~300 ops/byte FP16 ridge; batched prefill is right of the ridge).
2. **Batched decode.** Whisper-turbo's decoder is only 4 layers, but per token it reads
   ~157 MB of decoder weights + ~133 MB logit head (fp16). At batch=1 that is pure
   bandwidth: ~0.45 ms/token minimum. Decoding B=32 chunks in lock-step shares each
   weight read across 32 sequences → per-chunk decode cost drops ~an order of magnitude.
   HF already batches, but at 4, with per-step Python/launch overhead and CPU syncs.
3. **Fusion + GPU residency.** RDNA4 has no separate matrix pipe (cheatsheet §2): every
   stray vector op, barrier, or kernel-launch bubble subtracts ~1:1 from WMMA
   throughput. And every CPU round trip in the decode loop stalls all 64 CUs
   (cheatsheet §13.4: one project found 151 round trips per token). We keep the entire
   decode step GPU-resident, fuse elementwise ops into GEMM epilogues, and fuse the
   decode attention path into single kernels.

## 3. Model facts (large-v3-turbo)

- Checkpoint: `~/.cache/whisper/large-v3-turbo.pt` (OpenAI format, 809 M params).
- Mel: 128 bins, n_fft 400, hop 160, 16 kHz; 30 s → 3000 frames → conv stem → 1500.
- Encoder: conv1d(128→1280, k3 s1) + GELU, conv1d(1280→1280, k3 s2) + GELU,
  +sinusoidal pos; **32 layers**, pre-LN; d=1280, 20 heads × head_dim 64, FFN 5120,
  GELU; final LN. Self-attention has **no mask** (bidirectional, seq 1500) and **no RoPE**
  (learned/sinusoidal absolute pos only) — simplest possible flash-attention shape.
- Decoder: **4 layers**, d=1280, 20 heads, causal self-attn with KV cache, cross-attn
  over encoder output (K/V computed **once** per chunk, reused every step), FFN 5120,
  learned positional embedding, tied embedding/logit head, vocab **51866**.
- Special tokens: sot=50258, en=50259, transcribe=50360, notimestamps=50364,
  eot=50257. We decode greedy with
  `[sot, en, transcribe, notimestamps]` prompt, suppress OpenAI's standard
  non-speech token set + blank, max 448 positions, stop at eot.
  (No timestamp rules — declared openly in the report. Review R2 measured the
  baseline's timestamp overhead at ~19% more decode tokens, not the ~10% first
  assumed here; the final report adjusts/caveats accordingly.)

## 4. Numeric formats

- All GEMMs fp16 inputs, **fp32 WMMA accumulate** (`v_wmma_f32_16x16x16_f16`,
  intrinsic `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` — see
  `amd-r9700-docs/gpuopen/using-matrix-cores-rdna4.md` for exact VGPR layout: each
  lane holds 8 elements; A is column-major, B/C/D row-major).
- LayerNorm in fp32 math on fp16 tensors. Softmax in fp32.
- KV cache fp16.
- INT8 encoder GEMM (2x rate, cheatsheet §1) is a **stretch goal only** — do not start
  it before everything else is done and reviewed. FP8 not in scope.

## 5. System architecture

Host: Python + PyTorch (torch 2.x **rocm7.2 wheels** — must match system ROCm major
version; the rocm6.3-torch + hipcc-7.2 combination was tested and the runtimes clash).
Custom kernels: HIP C++ compiled with `/opt/rocm/bin/hipcc -fPIC -shared
--offload-arch=gfx1201 -O2`, one `libwhisper_kernels.so`, loaded via **ctypes**;
kernels receive raw `tensor.data_ptr()` pointers plus the torch stream
(`torch.cuda.current_stream().cuda_stream`). This pattern is smoke-tested
(`~/r9700-whisper/smoke/`). No torch C++ extension API — keeps the ABI surface zero.

Everything ships in `castria-experiments/experiments/rocm-whisper/` (Mac) and is
rsynced to `user@gpu-host:~/r9700-whisper/` for build + run. Never rsync with
`--delete`. Layout:

```
rocm-whisper/
  DESIGN.md                  (this file)
  kernels/
    common.h                 WMMA fragment helpers, tile loaders (shared)
    layernorm.hip            K1
    gemm_wmma.hip            K2 (+ epilogue variants)
    attention_encoder.hip    K3
    attention_decode.hip     K4 (+ KV append)
    logits.hip               K5 (suppress + argmax fusion)
    build.sh                 hipcc build → libwhisper_kernels.so
    test_*.py                per-kernel correctness + microbench vs torch
  whisper_rocm/
    __init__.py
    weights.py               load large-v3-turbo.pt → fp16, pre-packed layouts
    audio.py                 mel (torch.stft on GPU, batched), chunking 30s/no-overlap
    model.py                 encoder/decoder host orchestration; every custom kernel
                             behind a flag with a torch fallback (USE_KERNELS env/arg)
    decode.py                batched greedy loop, GPU-resident
    tokenizer.py             tiktoken (from openai-whisper package) + suppress list
  bench/
    bench_e2e.py             corpus benchmark → JSON (RTF per file + overall)
    bench_baseline_rerun.sh  re-run insanely-fast-whisper as-is for parity
    check_quality.py         WER ours-vs-baseline transcripts
  RESULTS.md
```

### Execution flow per audio file

1. Load + resample to 16 kHz mono (ffmpeg/soundfile, CPU, overlapped with GPU work).
2. Cut into non-overlapping 30 s chunks (same as baseline: 375 chunks for 11 235 s).
3. For each batch of B chunks (default 16/GPU, sweep 8/16/32):
   a. Mel on GPU (batched torch.stft, fp32→fp16). ~ms scale.
   b. Encoder forward, batched: (B·1500, 1280) GEMMs via K2, attention via K3.
   c. Cross-K/V precompute for all 4 decoder layers (one GEMM each over encoder out).
   d. Batched greedy decode: lock-step over B sequences, GPU-resident (see §7).
4. Detokenize per chunk (CPU, tiktoken), concatenate chunk texts in order.
5. Two GPUs = two worker processes (`HIP_VISIBLE_DEVICES=0` / `1` — **never expose
   device 2, it is the Ryzen iGPU**), each pulls whole files from a shared work queue;
   RTF measured over the total corpus wall time.

## 6. Custom kernels

Priority order — each lands only if it (a) passes tolerance vs the torch reference
(fp16 tolerance: max-abs ≤ 5e-2 on unit-scale tensors, or rel ≤ 1%), and (b) beats the
torch equivalent in a microbench at the real shapes. A kernel that loses stays behind
its flag as documentation, and the torch path ships. Report honestly per kernel.

**K1 — fused residual+LayerNorm** (`x = LN(residual + y)`, and plain LN).
One workgroup per row (1280 elems), wave32, two-pass mean/var in LDS, fp32 math,
write fp16. Replaces 3 torch kernels + 2 barriers per use; used ~65x per encoder pass.

**K2 — WMMA GEMM, fp16 in / fp32 acc, fused epilogues.**
Shapes that matter: M ∈ {B·1500 (encoder), B (decode)}, N ∈ {1280, 3840 (fused QKV),
5120}, K ∈ {1280, 5120}. Tile 128×128×32 per workgroup (4 waves of wave32, each wave
owns a 64×64 quadrant = 16 WMMA fragments), double-buffered LDS tiles, K=32 per outer
step (two chained k=16 WMMA per fragment). Epilogue variants (template/`constexpr`):
  - `BIAS` (all projections)
  - `BIAS_GELU` (FFN up; saves writing+re-reading the (M,5120) intermediate: ~1 GB of
    traffic per encoder layer at B=32)
  - `BIAS_RESIDUAL` (proj-out and FFN-down: adds the residual stream in-register)
Weights are **pre-packed offline** (weights.py) into the exact tiled layout K2 reads,
so the kernel inner loop does zero address arithmetic beyond tile pointer bumps and no
transposes (RDNA4 has no in-register transpose — cheatsheet §11).
VGPR budget: accumulators 16 frags × 8 f32 = 128 VGPR/lane — that is the 12-wave
(75%) occupancy step per cheatsheet §3's staircase; operand frags + addressing land
~160 → 9–10 waves. If measured occupancy < 9 waves, shrink to 96×128 tiles. Watch
`-Rpass-analysis=kernel-resource-usage` (see `amd-r9700-docs/compiler-toolchain/
llvm-amdgpu-usage-guide.md`) for exact VGPR counts at build time.

**K3 — encoder flash attention** (non-causal, no mask, seq 1500 padded to 1504,
B·20 heads of head_dim 64). FA-2 style: Q tile 128 rows × 64, K/V tiles 64×64
streamed through LDS, online softmax in fp32, exp via `V_EXP_F32` (separate
transcendental unit — it overlaps WMMA only if ≥2 waves resident per SIMD, so keep
VGPRs ≤ 168; cheatsheet §2). QK^T and PV both fp16 WMMA. Attention scale 1/8 folded
into Q at projection time (one less vector op in the inner loop — cheatsheet §15).
Padding rows masked with -inf bias on the score tile edge only (single compare on the
boundary tile, not in the hot loop).

**K4 — decode-step attention, fully fused per layer.**
Self-attn: one workgroup per (batch b, head h): append new K/V to cache (fp16,
layout [B, H, T_max, 64] contiguous in T so appends are coalesced), then score q·K
over t≤T in LDS tiles, fp32 softmax, PV accumulate. M=1 → **no WMMA** (matrix cores
are irrelevant at decode, cheatsheet §6); pure wave32 dot-product kernel, bandwidth
bound by design. Cross-attn: same kernel without append, T=1500, K/V from the
precomputed cross cache.
Decoder GEMMs (M=B): use K2 with a small-M tile variant (32×128; spec-constant-style
`constexpr` — one binary per tile config, mirroring the Vulkan spec-constant lesson
in cheatsheet §10: decode must not pay prefill's register footprint).

**K5 — logits fusion.** Final GEMM (B×1280 · 1280×51866) with epilogue: add suppress
mask (precomputed -inf vector), then block argmax → global argmax (two-stage) →
writes next-token ids [B] and per-sequence finished flags directly on GPU. Kills the
biggest per-step sync (a 51866-wide logit tensor never leaves the GPU).

**K6 (only if time permits) — decoder microfusions:** LN+QKV preamble fusion,
embedding-add fusion. Skip unless review finds launch overhead still dominant.

## 7. GPU-resident decode loop

Per step, per layer: K1(LN) → K2(QKV, M=B) → K4(self) → K2(proj+residual) → K1 →
K4(cross) → K2(proj+residual) → K1 → K2(FFN up+GELU) → K2(FFN down+residual);
then final K1 + K5. ≈ 40 kernel launches per step at B sequences — no Python between
them beyond the launch calls themselves (all shapes static; tensors preallocated).
- EOS handling: K5 maintains `finished[B]` on GPU; finished sequences are forced to
  eot and their outputs ignored. Host checks a pinned-memory copy of
  `all_finished` **every 8 steps** (lagged, async) — no per-step sync.
- Launch overhead target: ~40 launches × ~8 µs ≈ 0.3 ms/step CPU side. Acceptable at
  B≥16 (amortized ≤20 µs per chunk-token). If profiling shows launch gaps dominate,
  capture the step as a HIP graph via `torch.cuda.CUDAGraph` (works on ROCm; our
  ctypes launches are on the capture stream so they are captured too) — treat as an
  optimization pass, not day-one requirement.
- Token cap 448; typical podcast 30 s chunk ≈ 60–120 tokens. **Never cap below eot
  for benchmarking** (the old triton-whisper bench.py capped at 100 tokens and used a
  wrong start token — both invalidate its numbers; do not reuse that code).

## 8. Correctness protocol (non-negotiable)

Reference: `openai-whisper` package (pip) running the same checkpoint in fp32 on CPU
(or fp32 GPU) — it is the ground-truth implementation of every stage.
1. Stage-by-stage tensor diff on one fixed 30 s clip, in order: mel → conv stem →
   encoder block 0 → encoder out → decoder logits at step 0 → first 20 greedy tokens.
   Find the **first** divergent stage (debugging playbook, cheatsheet §13). Only then
   look at kernels.
2. Every custom kernel additionally has a standalone `test_*.py` vs the torch op at
   real shapes (including the padding edge: M=1500·B non-multiple-of-tile, T=1 first
   decode step, T=448 last).
3. End-to-end: word-level divergence of our transcript vs the baseline's
   `bench-results/*.json` text on all 5 files, computed as WER with the baseline as
   reference. This is a DIVERGENCE measure between two non-ground-truth transcripts,
   not an error rate — report it as such (R2 finding F6). Target ≤3% after
   no_repeat_ngram parity lands; large divergence = investigate before benchmarking.

## 9. Benchmark protocol

- Corpus: the 5 mp3 files in `~/insanely-fast-whisper-rocm/bench-audio/` (29 578 s).
- Measure per file: wall seconds (everything after model+weights load: audio decode,
  mel, encoder, decode, detokenize), RTF = wall/duration; plus overall RTF.
  One warmup batch before timing (compile/alloc effects).
- Configs reported: single-GPU B∈{8,16,32}, dual-GPU best-B. Plus per-stage timing
  breakdown (mel/encode/decode) at the headline config, and kernel-level A/B table
  (each K# on/off) so the report shows where the wins actually came from.
- Baseline: rerun `insanely-fast-whisper-rocm` once as-is to confirm ≈0.0488 on the
  current driver; report both the recorded and rerun numbers.
- Measurement hygiene (cheatsheet §14): release build only, no profilers in the
  timed path, `torch.cuda.synchronize()` around timed regions, report the ROCm/driver
  versions in RESULTS.md, exclude the iGPU, both benchmark processes must not share
  a GPU while timing. GPU timing sanity: `rocm-smi` sclk during runs (device may be
  in low-power state after WoL — first warmup batch also serves as clock ramp).

## 10. Milestones and go/no-go

- **M0** env: rocm7.2 torch venv at `~/r9700-whisper/.venv` + smoke interop rerun. (orchestrator — done/in progress)
- **M1** pure-torch batched pipeline, correct + benched. This alone (batching, GPU
  residency, dual GPU) probably beats the baseline — it is the safety net. Gate: WER
  parity vs baseline on ≥1 full file. (Agent A)
- **M2** kernels K1–K5 land one at a time behind flags, each with tolerance test +
  microbench. Gate per kernel: correct AND faster, else torch path stays. (Agents B, C)
- **M3** adversarial review (Opus): kernels vs cheatsheet rules (occupancy staircase,
  wave32 everywhere, LDS bank conflicts, WMMA fragment layout, numerics drift across
  the KV cache and long files), harness validity (no hidden caps, no timing games,
  batch-boundary handling of the last partial batch, straggler chunk accounting).
- **M4** final sweep + RESULTS.md + report.

## 11. Ownership and coordination

- Agent A (Sonnet): `whisper_rocm/` host package, bench/, M1.
- Agent B (Sonnet): `kernels/` K1, K2, K3 (+ common.h, build.sh, tests). GPU 0 for timing.
- Agent C (Sonnet): `kernels/` K4, K5 (+ decode.py integration with A). GPU 1 for timing.
- File ownership is exclusive as listed; shared touchpoints (model.py flags, decode.py)
  belong to A — B/C deliver kernels + python wrappers in `kernels/`, A integrates.
- Timing runs on gpu-host: take `mkdir ~/r9700-whisper/.lock-gpu0` (or gpu1) as a
  mutex; remove after. Correctness runs don't need the lock.
- All agents: read `amd-r9700-docs/RDNA4_ARCHITECTURE_CHEATSHEET.md` §2, §3, §4 before
  writing any kernel, and `gpuopen/using-matrix-cores-rdna4.md` before touching WMMA.
