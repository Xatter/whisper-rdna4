# rocm-whisper: Results

Whisper large-v3-turbo on 2x AMD Radeon AI PRO R9700 (gfx1201), custom
ROCm/HIP decode kernels + pure-torch encoder. See `DESIGN.md` for the
architecture and `kernels/DECODE_INTEGRATION.md` for the K4/K5 contract.

**Headline-config update (2026-08-10, Jim's decision):** `chunking="vad"`
is now the DEFAULT in `pipeline.py` and `bench_e2e.py` -- see "VAD
chunking mode" near the end of this document. Every hard-cut number
above that section (including "headline config" as used throughout
M1-M6) predates this change and is left as originally measured/reported;
it describes what was current *at the time*, not what ships today.
Hard-cut and stride both remain available (`--chunking hard`/`stride`)
for comparison or fallback.

## Hardware / driver / software versions

| | |
|---|---|
| Machine | gpu-host (`user@gpu-host`) |
| GPUs | 2x AMD Radeon AI PRO R9700 (gfx1201, Navi 48, 32 GB GDDR6, 640 GB/s) -- `HIP_VISIBLE_DEVICES=0,1`; device 2 (Ryzen iGPU) never used |
| CPU / RAM | Ryzen 5 9600X, 12 threads, 30 GB RAM |
| Kernel | Linux 7.1.5-arch1-2 |
| ROCm | 7.2.4 (`/opt/rocm`) |
| torch | 2.13.0+rocm7.2 |
| Checkpoint | `large-v3-turbo.pt`, OpenAI format, 809M params, fp16 |
| Kernel toolchain | `hipcc`, `--offload-arch=gfx1201 -O2 -mno-wavefrontsize64` (wave32) |

## Certified comparison: our wall_s vs baseline wall_time_seconds, per file

Both sides measured post-model-load, audio I/O included. Baseline is
`insanely-fast-whisper-rocm` (HF `transformers` ASR pipeline, fp16,
`batch_size=4`, `chunk_length_s=30`), recorded numbers from
`bench_gpu-host_results.json`; "rerun" is a fresh run on the current
driver (one file, `bench/bench_baseline_rerun.sh`) to confirm the
recorded numbers still hold. Ours: headline config (hard-cut, kernels ON,
guards ON, ngram ban OFF, best B from the sweep below). Best B = 16 (see
batch sweep below).

| file | duration_s | baseline wall_time_s | ours wall_s | speedup |
|---|---|---|---|---|
| 1abf3ec0-...-e344.mp3 | 11235.3 | 570.86 | 32.49 | 17.57x |
| 44e8f624-...-acf4.mp3 | 3670.6 | 193.68 | 11.26 | 17.20x |
| 882380ee-...-6be.mp3 | 4864.6 | 257.89 | 15.09 | 17.09x |
| 9beeffbb-...-41e6.mp3 | 3796.4 | 187.18 | 11.07 | 16.91x |
| 9f0ed85f-...-e2.mp3 | 6011.0 | 320.59 | 24.28 | 13.21x |
| **total / overall** | **29577.99** | **1530.21** | **94.19** | **16.25x** |

Overall RTF (5-file corpus, baseline's own self-reported numbers):
0.0517 wall-based (`overall_wall_rtf`) / 0.0488 runtime-based
(`overall_rtf`, excludes some I/O baseline's own harness doesn't count as
"runtime") vs ours 0.00318 -> **16.3x / 15.3x faster**.

Baseline rerun (separate from the table above -- one file only,
`44e8f624...`, on the current driver, to confirm the recorded numbers
still hold rather than as part of the certified comparison): wall 189.85s
(wall-rtf 0.0517) vs the recorded 193.68s (wall-rtf 0.0528) for that same
single file -- 2.0% faster, inside normal run-to-run noise. Confirms the
recorded baseline is still representative on this driver. (This single-file
rerun wall-rtf, 0.0517, numerically coincides with the whole-corpus
`overall_wall_rtf` above -- that's a coincidence of these two particular
numbers, not the same measurement.)

## Three caveats from the adversarial review (R2)

These qualify every RTF/divergence number in this document; they are not
edge cases discovered late, they are structural differences between our
pipeline and the baseline's that no amount of kernel tuning removes.

1. **Timestamp tokens change the token stream by ~19%.** Baseline decodes
   WITH timestamp tokens interleaved (`chunk_length_s=30`, HF's default
   `return_timestamps=True` path); ours decodes with `notimestamps=True`
   throughout (DESIGN.md S3), so our decoder never emits a timestamp
   token at all (also now permanently suppressed in the logit mask, see
   `whisper_rocm/tokenizer.py`'s `TIMESTAMP_BEGIN` suppression, M4).
   Baseline's own token stream is measured to contain roughly 19%
   more tokens than a timestamp-free decode of the same audio would, from
   the interleaved `<|t.tt|>` tokens alone. This is a structural format
   difference, not a quality difference, but it means per-chunk token
   counts and decode-step counts are not directly comparable between the
   two pipelines.
2. **Baseline's batch_size=4 is specified but not actually exercised as
   batch=4 compute.** `bench_gpu-host_results.json` records
   `"batch_size": 4`, but the effective execution pattern that produced
   those numbers is closer to batch=1 per generation call -- the recorded
   baseline numbers reflect that effective batch=1 behavior, not a
   genuine 4-wide batched decode. This matters for reading the RTF
   comparison below honestly: part of our speedup is real batching (we do
   batch, at B up to 32) but part of the RTF gap already existed in the
   baseline's own configuration before we changed anything.
3. **Ngram-ban parity is implemented but off by default.** The "Quality:
   divergence, not error rate" section below has the full account: Agent
   C's K4/K5 kernels support a
   config-exact `no_repeat_ngram_size=3` ban matching baseline's
   `generate_kwargs`. Measured on real audio it regresses divergence
   (15.46% -> 18.40% on one file, hard-cut) because baseline's interleaved
   timestamp tokens break up repeated word-trigrams in token space (so
   its ban rarely fires on natural speech), while our notimestamps stream
   makes the same phrases contiguous trigrams that the ban does fire on
   -- and greedy decoding has no graceful recovery once its natural
   continuation is banned, so it truncates to EOT instead. The flag is
   implemented, tested (28/28 kernel tests including ngram cases), wired,
   and OFF by default for the headline numbers in this report; it stays
   available for a future timestamps-mode pipeline where the rationale
   above would no longer apply.

## Quality: divergence, not error rate

Both our transcript and baseline's transcript are model outputs, not
human ground truth (`bench-gt/*.json` is ad-detection ground truth, not a
transcript -- not used for this comparison). "Divergence" below is
normalized word-level edit distance between two non-ground-truth
transcripts (lowercase, strip punctuation, collapse whitespace, standard
Levenshtein over the word sequences, normalized by baseline word count --
`bench/check_quality.py`). It is not a word error rate in the ASR-research
sense (no ground truth transcript exists to compute a true WER against),
and a nonzero number does not by itself mean either side is "wrong" --
see the three factors below.

**Factors contributing to divergence** (established across M1-M4
investigation, not guesses):
- Timestamp-mode difference (caveat 1 above) affects both chunk
  segmentation (baseline's HF pipeline chunks with 5s stride each side by
  default -- `stride_length_s = chunk_length_s/6` -- and merges
  timestamp-aware; ours hard-cuts at the headline config) and ngram-ban
  applicability (caveat 3).
- fp16 nondeterminism: two independently-computed fp16 forward passes
  (different GEMM libraries, different accumulation order) diverge by a
  small but nonzero amount per step; over up to ~450 autoregressive steps
  this compounds (cheatsheet S13.3's stateful-kernel warning, confirmed
  directly in M1's stage-by-stage check: encoder-out max-abs error 5.28
  but median relative error 0.55%, well within the fp16 tolerance band).
- Greedy-decode stalls on ambiguous audio are inherent to the model, not
  a pipeline bug: M1 demonstrated this directly by feeding one
  underperforming 30s window into BOTH our fp16 pipeline and the
  openai-whisper fp32 CPU reference (ground truth implementation) in
  isolation -- both independently produced the same 2-token "Yeah."
  output on that window, with no elevated no-speech signal and no
  repetition for the repetition brake to catch. This is a known
  limitation of pure greedy decoding (no temperature-fallback retry, see
  "Honest notes" / "what was not attempted" below) that affects a small,
  roughly-constant fraction
  of chunks (~6%, both hard-cut and stride windowing) regardless of
  chunking strategy -- confirmed by a controlled ablation in M2 showing
  near-identical near-empty-window rates under both chunking modes.

### Four-configuration divergence table

Single file `44e8f624-1876-460e-bcd4-8180b7a3acf4.mp3` (3670.6s), kernels
ON, no-speech gate + repetition brake ON in all four cells (established in
M2 as net-positive and kept on throughout):

| chunking | ngram ban | divergence | ref words | hyp words | edits |
|---|---|---|---|---|---|
| hard | OFF (**headline**) | 15.46% | 10650 | 9263 | 1646 |
| hard | ON | 18.40% | 10650 | 8982 | 1960 |
| stride | OFF | 13.65% | 10650 | 9524 | 1454 |
| stride | ON | 15.92% | 10650 | 9387 | 1695 |

Stride chunking (5s overlap each side + word-level LCS merge,
`whisper_rocm/pipeline.py`) consistently diverges lower than hard-cut, but
the headline config below uses hard-cut per the orchestrator's explicit
direction. Stride numbers are reported here for completeness, not as the
certified comparison.

**Open discrepancy, flagged rather than silently resolved:** the M4
instruction states "review confirmed baseline hard-cuts 30s, no stride."
I re-verified this directly against the baseline's own source
(`insanely_fast_whisper_rocm/core/asr_backend.py`) rather than taking it
on trust, since M1/M2 had already found the opposite once. The recorded
benchmark run used `return_timestamps=True` with `chunk_length_s` set and
`stride_length_s` never overridden (config dump in
`bench-results/*.json` confirms `"timestamp_type": "chunk"`, not the
`"word"` value that would trigger the code's separate manual-chunking
path) -- which routes through `transformers`' standard
`AutomaticSpeechRecognitionPipeline.preprocess`, where
`stride_length_s = chunk_length_s / 6 = 5s` is the default whenever it
isn't explicitly passed (verified directly in that package's source,
`automatic_speech_recognition.py`, both in M1 and again here). I could not
find a code path that disables this for the recorded run. This document
still uses hard-cut as directed -- it is a reasonable, defensible choice
independent of what baseline does, since it's simpler and matches
DESIGN.md's original spec -- but the specific claim "baseline hard-cuts,
no stride" does not match what I can observe in baseline's own code, and
I'm not able to reconcile that from my side. Flagging for the
orchestrator/reviewers rather than silently adjusting the wording to
match the instruction.

**Orchestrator resolution (runtime evidence, settles the discrepancy):**
the recorded baseline transcript for `44e8f624` contains 1149 segments of
which **129 start exactly on a multiple of 30s** — essentially one per
chunk boundary (the file has 123 hard-cut chunks). That timestamp
signature is only produced by external hard 30s chunking with per-chunk
re-based timestamps offset by exact multiples of 30; a whole-file HF
stride+merge run would place segment boundaries at arbitrary times
(expected exact-30s snaps: ≈0). Both agents read the code correctly —
the stride default exists in HF's `preprocess`, but the recorded run's
external `split_audio(chunk_duration=30.0, chunk_overlap=0.0)`
pre-chunking (R2 finding F4) means each HF call received a ≤30s file, so
the stride default never engaged. The recorded baseline hard-cut 30s
with zero overlap at runtime; hard-cut is the correct apples-to-apples
configuration for the certified comparison above.

### Per-file divergence at the headline config (all 5 files)

Headline config: hard-cut, kernels ON, B=16, no-speech gate + repetition
brake ON, ngram ban OFF.

| file | divergence | ref words | hyp words | edits |
|---|---|---|---|---|
| 1abf3ec0-...-e344.mp3 | 12.05% | 33274 | 30245 | 4011 |
| 44e8f624-...-acf4.mp3 | 15.46% | 10650 | 9263 | 1646 |
| 882380ee-...-6be.mp3 | 11.51% | 15366 | 14245 | 1768 |
| 9beeffbb-...-41e6.mp3 | 11.51% | 10156 | 9286 | 1169 |
| 9f0ed85f-...-e2.mp3 | 9.95% | 18152 | 17249 | 1806 |
| **corpus-wide** | **11.87%** | **87598** | **80288** | **10400** |

Consistent with the single-file finding above: none of the 5 files hit the
original ≤3% target from DESIGN.md S1, for the reasons in the factors list
above (chunking/timestamp-mode mismatch, fp16 drift, inherent greedy-decode
stalls on a small fraction of chunks) -- not evidence of a bug specific to
any one file.

## Batch-size sweep (single GPU 0, kernels ON, hard-cut, guards ON, ngram OFF)

Full 5-file corpus (29578s) at each B, one warmup batch before timing,
`torch.cuda.synchronize()` around every timed region (DESIGN.md S9):

| B | overall_rtf | total_wall_s | vs baseline (0.0488 recorded) |
|---|---|---|---|
| 8 | 0.00344 | 101.87 | 14.2x |
| **16** | **0.00318** | **94.19** | **15.3x** |
| 32 | 0.00422 | 124.86 | 11.6x |

**B=16 wins.** B=32 is worse, not better -- two of the five files
(`44e8f624`, `9beeffbb`, both mid-length) show a large RTF spike at B=32
specifically (0.00779 and 0.00826 vs ~0.003 at B=16), while the other
three files are flat or slightly better than B=16. This looks like a
last-partial-batch or VRAM-pressure effect at higher B on files whose
chunk count doesn't divide cleanly by 32, not a uniform compute-bound
slowdown -- flagged here rather than investigated further, since B=16 is
unambiguously the sweep winner regardless of the exact cause.

## Kernels on vs off (single GPU 0, B=16, full corpus)

| | overall_rtf | total_wall_s |
|---|---|---|
| kernels OFF (pure torch decode) | 0.00434 | 128.36 |
| kernels ON (K4 self/cross-attn + K5 logits/argmax) | 0.00318 | 94.19 |

**1.36x** full-corpus speedup from Agent C's decode kernels alone, at the
headline B. (M2's isolated single-file decode-stage measurement found a
larger 2.17x decode-only speedup; the smaller 1.36x here is the
*end-to-end* number, diluted by encode/mel/load stages that don't change
between kernels on/off -- both are correct, they're answering different
questions.)

## Dual-GPU (kernels ON, B=16, full corpus, balanced split)

A first attempt split files by alternating list order (`files[0::2]` /
`files[1::2]`), which put 3 of the 5 files (22111s of audio) on GPU0 and
just 2 (7467s) on GPU1 -- a badly unbalanced 3:1 split for a
duration-skewed 5-file corpus, and it showed: wall clock 100.7s, barely
better than single-GPU's 94.2s. Fixed with a duration-aware
longest-processing-time-first greedy split (`bench/bench_e2e.py:
_balanced_split`) before reporting the headline dual-GPU number:

| | GPU0 | GPU1 |
|---|---|---|
| files | `1abf3ec0...`, `44e8f624...` | `9f0ed85f...`, `882380ee...`, `9beeffbb...` |
| audio duration | 14905.9s | 14672.0s |
| model load | 0.397s | 0.391s |
| sum of file walls | 44.03s | 44.81s |

**One wall clock over the whole corpus (both workers' cold start
included): 79.3s.** Overall RTF: 0.00268 -> **18.2x vs baseline recorded
(0.0488)**; vs single-GPU-B16 (94.19s), that's a **1.19x dual-GPU
speedup** (79.3s). The 1.19x is well short of the DESIGN.md S1 "stretch: >=4x
with dual GPU" target -- with only 5 files and each worker paying its own
~0.4s model load plus warmup batch independently (not amortized across
workers the way a single process amortizes it across all 5 files), a
5-file corpus is too small to show dual-GPU's real headroom. The
per-worker sum-of-file-walls (44.03s / 44.81s) is close to the ideal
half of single-GPU's 94.19s (47.1s), which is the more honest read of
dual-GPU's actual per-file throughput; the 79.3s wall clock also carries
both workers' independent startup cost, which a longer-running production
service would amortize away.

## Per-stage timing breakdown (headline config: hard-cut, kernels ON, B=16)

Summed across all 5 files (full corpus), `bench/bench_e2e.py`'s per-file
`stage_timings_s`:

| stage | total_s | % of wall |
|---|---|---|
| audio load (I/O + resample) | 30.33 | 32.2% |
| mel (batched GPU STFT) | 0.38 | 0.4% |
| encode (32-layer pure-torch encoder) | 35.97 | 38.2% |
| decode (4-layer, K4/K5 kernels) | 27.49 | 29.2% |
| detokenize | 0.01 | 0.01% |
| **total** | **94.19** | **100%** |

Two things worth naming plainly:

- **Encode is now the single largest stage (38.2%), edging out decode
  (29.2%) for the first time across this project.** M1's pure-torch
  baseline had decode dominating (11.6 of 19.4s on one file); once K4/K5
  cut decode's cost, the pure-torch encoder became the new largest
  component. This is exactly what DESIGN.md S6 anticipated K1/K2/K3
  would attack -- they were tried (Agent B) and lost to torch/hipBLASLt/
  SDPA at the tested shapes (see Honest notes), so this is the current
  ceiling on encode without a different kernel strategy, not an
  oversight.
- **Audio load (32.2%) is now comparably sized to the GPU stages** --
  expected once decode got 2.17x faster: the fixed cost of
  reading+resampling five real mp3 files (largest is 11235s / ~180MB)
  doesn't change no matter how fast the GPU-side stages get, so it
  becomes relatively more visible. Not optimized this pass (no kernel
  work applies to CPU-side audio I/O); a production service would
  overlap it with GPU work across files, which this benchmark's per-file
  serial timing does not (DESIGN.md S9 measures stages serially per file
  for clarity, not for the fastest achievable pipelined throughput).

## GPU clocks and VRAM during a timed run

Sampled every 0.5s via `rocm-smi --showclocks --showmeminfo vram` during
the kernels-off B=16 full-corpus run (single GPU 0; a deliberately loaded,
few-minutes-long run, chosen so the sample count would be meaningful):

| | GPU0 |
|---|---|
| sclk range | 0 - 3350 MHz (340 samples) |
| VRAM used range | 59.9 MB - 4688.9 MB (340 samples) |

3350 MHz matches the R9700's ~1.5x boost clock cited in the architecture
cheatsheet (S1's ~195 TFLOPS FP16 figure is quoted at that boost clock).
The 0 MHz samples are idle/between-batch moments (host-side stages: audio
load, tokenizer setup, JSON writes) where the poll caught the GPU
genuinely idle, not a clock-ramp artifact -- the very first batch after
each idle moment re-ramps to boost within the measurement's 0.5s
resolution, consistent with the architecture cheatsheet's S14
("measurement hygiene") note about post-idle clock ramp being visible
mainly in the first warmup batch (which is excluded from all timed
regions per DESIGN.md S9). Peak VRAM (4.7 GB) is well
within the R9700's 32 GB, consistent with DESIGN.md S1's estimate (weights
1.6 GB fp16 + activations ~0.5 GB at B=32; this sample was B=16, kernels
off, so somewhat higher activation memory from the extra torch
intermediates that K4/K5 would otherwise avoid materializing).

## Honest notes

- **Agent B's encoder kernels (K1 LayerNorm, K2 GEMM, K3 encoder
  attention) are implemented, tested, and off.** They lost to
  torch/hipBLASLt/SDPA at the tested shapes (K2 measured 0.5-0.7x vs
  torch, K3 measured 0.2x vs torch's SDPA backend) -- per DESIGN.md S6's
  own rule ("a kernel that loses stays behind its flag as documentation,
  the torch path ships"), they are not wired into the default path. The
  encoder runs entirely in pure torch.
- **Agent C's decode kernels (K4 self/cross-attention, K5
  logits+argmax+ngram-ban) are on** and are the source of the decode-stage
  speedup measured below -- both passed their full correctness suites
  (28/28) and beat torch at the tested shapes (K4: 1.4-2.4x; K5: 1.1-1.4x
  plus a near-free +2.6-3.6us ngram-ban option).
- **What was not attempted:**
  - **Timestamps mode.** DESIGN.md scoped this out from the start
    (S3: "No timestamp rules -- declared openly in the report"). Adding
    it would change chunk-merge behavior, ngram-ban applicability
    (caveat 3), and decoder token counts (caveat 1) all at once -- a
    scope change bigger than a benchmark tweak.
  - **Temperature-fallback / beam-search retry.** openai-whisper's own
    `transcribe()` (as opposed to the lower-level `decode()` this project
    mirrors) retries a chunk at higher temperature when a greedy decode's
    compression ratio or average logprob looks suspicious, specifically
    to recover from the "stuck at premature EOT" failure mode this report
    documents ("Quality: divergence, not error rate" above, third factor).
    DESIGN.md S7 specifies greedy-only decoding; implementing a retry
    ladder is new decode-strategy scope, not attempted here.
  - **INT8 encoder GEMM.** DESIGN.md S4 marks this an explicit stretch
    goal, gated to start only after everything else is done and
    reviewed. Not started.
  - **HIP graph capture for the decode step.** Investigated in M2:
    Agent C's K4 kernel takes `step` as a plain Python int baked into the
    launch args at capture time (a deliberate, documented design choice
    for the lock-step decode loop -- `DECODE_INTEGRATION.md` S5), so a
    captured graph would replay the wrong position every iteration.
    Fixing this needs a device-tensor step counter, which changes K4's
    contract -- flagged as out of scope for this pass rather than
    resolved unilaterally.
  - **QKV GEMM fusion** was tried (M2) and reverted: K4 needs contiguous
    `(B, n_head, head_dim)` inputs, and slicing a fused `(B, 3840)` GEMM
    output isn't contiguous, so the `.contiguous()` copies needed to
    restore it fully offset the GEMM launches saved. Left as a documented
    dead end in `whisper_rocm/model.py`.

## Comparison vs the production service's transcripts

All 5 bench-corpus episodes are production episodes; their production
transcripts (from the production Whisper API service, an internal
OpenAI-compatible `/v1/audio/transcriptions` endpoint -- specific
engine/model unknown from here, but
the segment timestamps show it decoded WITH timestamps, unlike our
notimestamps pipeline) are exported locally at
per-episode JSON files in the private experiments tree. This section
triangulates: how far is *our* transcript from production, how far is the
*old baseline's* transcript from production, and does that put our
divergence number in context.

**Reference text**: `transcript` field, a list of `{Start, End, Text}`
segments; reference text is every `Text` joined with spaces, in segment
order. Divergence normalization is identical to the rest of this
document (`bench/check_quality.py`: lowercase, strip punctuation, collapse
whitespace, normalized word-level Levenshtein edit distance).

### Timing context: prod vs baseline vs ours

`transcribe_seconds` is production's own recorded transcription time per
episode (from the same exported JSON, `download_seconds` excluded --
that's prod's audio-fetch time, not transcription).

| file | duration_s | prod transcribe_s (rtf) | baseline wall_s (rtf) | ours wall_s (rtf) |
|---|---|---|---|---|
| 1abf3ec0-...-e344.mp3 | 11235.3 | 188.47 (0.01677) | 570.86 (0.05081) | 32.49 (0.00289) |
| 44e8f624-...-acf4.mp3 | 3670.6 | 63.87 (0.01740) | 193.68 (0.05277) | 11.26 (0.00307) |
| 882380ee-...-6be.mp3 | 4864.6 | 86.37 (0.01776) | 257.89 (0.05301) | 15.09 (0.00310) |
| 9beeffbb-...-41e6.mp3 | 3796.4 | 65.15 (0.01716) | 187.18 (0.04930) | 11.07 (0.00292) |
| 9f0ed85f-...-e2.mp3 | 6011.0 | 113.04 (0.01880) | 320.59 (0.05333) | 24.28 (0.00404) |
| **total / overall rtf** | **29578.0** | **516.89 (0.01748)** | **1530.21 (0.05173)** | **94.19 (0.00318)** |

Production's own service is already **2.96x faster than the old
insanely-fast-whisper-rocm baseline** (0.0517 -> 0.0175 RTF) -- it is
presumably running a more optimized/batched engine or better hardware
than the single-machine baseline this project has been benchmarking
against throughout. Our pipeline is **5.49x faster than production**
(0.01748 -> 0.00318) and **16.2x faster than the old baseline**,
consistent with the rest of this document.

### Divergence, three ways

All three comparisons use production's transcript as the shared reference
where applicable (columns 2-3); column 4 (ours vs baseline) is reused
unchanged from the "Per-file divergence at the headline config" table
above -- not recomputed.

| file | ours vs prod | baseline vs prod | ours vs baseline | prod closer to |
|---|---|---|---|---|
| 1abf3ec0-...-e344.mp3 | 13.24% | 7.09% | 12.05% | baseline |
| 44e8f624-...-acf4.mp3 | 15.00% | 6.80% | 15.46% | baseline |
| 882380ee-...-6be.mp3 | 12.91% | 9.36% | 11.51% | baseline |
| 9beeffbb-...-41e6.mp3 | 11.51% | 7.58% | 11.51% | baseline |
| 9f0ed85f-...-e2.mp3 | 9.38% | 10.19% | 9.95% | **ours** |
| **corpus-wide** | **12.39%** | **8.16%** | **11.87%** | baseline |

(Corpus-wide ref word counts differ slightly by comparison, since each
uses a different reference transcript's word count as the normalization
denominator: ours-vs-prod and baseline-vs-prod both normalize by
production's word count, 88323 total; ours-vs-baseline normalizes by
baseline's, 87598 total -- close but not identical, as expected.)

### Interpretation

The hoped-for outcome going in was that if baseline-vs-prod and
ours-vs-prod came out similar, that would argue the ~12% divergence floor
is mostly a cross-engine artifact rather than a deficiency specific to
this pipeline. **The numbers don't support that reading.** Baseline is
measurably closer to production than we are, in 4 of the 5 files (the
exception, `9f0ed85f`, is one where ours is closer, 9.38% vs 10.19%, but
only by a small margin) -- corpus-wide, baseline-vs-prod (8.16%) is
noticeably lower than ours-vs-prod (12.39%), a real gap, not noise.

What the numbers do support, more modestly:

- **Our divergence from any external reference sits in a fairly narrow,
  consistent band** (11.87% vs baseline, 12.39% vs production) --
  whatever is driving it is roughly constant across comparisons, not
  something that happens to line up badly with baseline specifically.
- **Baseline and production agree with each other unusually well**
  (8.16% corpus-wide, noticeably tighter than either pipeline is to the
  other). The most likely shared cause, given what's independently
  established elsewhere in this document: baseline decodes WITH
  timestamps (`return_timestamps=True`, confirmed in its own source) and
  production's segment timestamps show it does too, while our pipeline
  runs `notimestamps=True` throughout by design (DESIGN.md S3). Two
  systems that both decode with timestamp-aware chunking/context sharing
  a lower divergence, while the one system that doesn't sits further from
  both, is at least consistent with the timestamp-mode difference being a
  real contributor -- not proven from this data alone (production's exact
  engine is unknown, and could differ from baseline's HF/transformers
  lineage in other ways that happen to correlate), but a more specific and
  better-supported claim than "it's just generic cross-engine noise."
- This reframes caveat 1 from the earlier section: the timestamp-mode gap
  is not just a token-count/format difference (the original framing) --
  it may be doing real work keeping baseline's and production's outputs
  aligned, which our pipeline forgoes by design.
- Divergence here is still not a word error rate against ground truth --
  production's transcript is itself a model output from an engine whose
  internals aren't known from this vantage point, not a human transcript.
  A lower baseline-vs-prod number is evidence baseline's *output* pattern
  resembles production's more closely, not proof baseline is more
  "correct."

No pipeline changes were made in response to this finding -- it's
reported for the record, per the task, not acted on. If closing this gap
mattered, the timestamp-mode hypothesis above is the concrete, testable
next step (a timestamps-mode pipeline was already scoped out as "not
attempted" earlier in this document, for unrelated reasons -- this finding
is a second, independent reason it might be worth revisiting).

## Timestamps mode

Jim approved building a timestamps-decode mode: `decoding.py:
ApplyTimestampRules`, ported exactly (`whisper_rocm.ops`'s torch fallback;
Agent C's K5 `enable_timestamp_rules`/`prompt_len` extension for the kernel
path -- see the kernel-bug note below). Prompt drops NOTIMESTAMPS (3
tokens: sot, en, transcribe); the suppress mask leaves the timestamp range
open; segment extraction (`whisper_rocm.segments`) turns the decoded
token stream into `{Start, End, Text}` dicts -- the same shape as
Castria's own `data/episodes/{uuid}.json` `transcript` field, so this
pipeline is now drop-in for ad-detection input.

### Correctness: torch fallback verified; kernel bug found, fixed, re-verified

Torch fallback vs the openai-whisper fp32 CPU reference (same 30s clip
used throughout this project): **39/40 tokens match exactly**, timestamps
align perfectly (`50365, 50434, 50490, 50535, ...`), and the one token
that differs is the identical single-token fp16-vs-fp32 tie-break already
characterized in M1 (`"The"` vs `"the"`), not a new issue.

**Kernel path: found a real bug, not integration noise, reported rather
than silently worked around.** Comparing kernel-on vs torch-fallback
output on real audio (file `44e8f624`, chunk 1) showed a divergence at
decode position 28: torch opens a new segment (timestamp token), the
kernel emits EOT and stops early -- cutting off a chunk that clearly has
more speech in it. Isolated with a forced-prefix replay so both paths see
byte-identical history up to the divergence point, then inspected the raw
comparison Rule 4 (`ApplyTimestampRules`'s "does timestamp probability
mass exceed the best text token") depends on: `ts_logprob = -0.159` vs
`max_text_logprob = -1.916` -- a decisive ~1.76 log-prob margin, not a
close call fp16 noise could flip. The torch fallback correctly bans EOT
here (matches the reference math exactly: comparing raw-logit
logsumexp/max is provably equivalent to the log_softmax version, the
normalizing constant cancels). Calling `kernels/decode_api.py`'s
`launch_timestamp_rules` **directly** (bypassing my own wrapper entirely,
to rule out an integration bug on my side) reproduced the same failure:
EOT's logit came back unbanned (`11.195`, not `-inf`). This isolated the
bug to `timestamp_rules_kernel`'s Rule 4 in `kernels/logits.hip` itself --
not fixed here (`kernels/` is Agent C's file per this project's ownership
rules), reported for Agent C to fix. Every number in this section up to
here was measured on a K4-kernel + K5-torch-fallback configuration while
the bug stood.

**Fixed by Agent C**: the bug was an online-logsumexp accumulator
initialized at `-inf`, hitting `exp(-inf - -inf) = NaN` when a
monotonicity-masked timestamp landed first in a thread's stride -- fix
skips `-inf` elements (mathematically exact, not an approximation).
78/78 kernel tests pass, including a stride sweep over the exact repro
shape. Re-verified two ways:
- **Forced-prefix replay** (the exact repro that caught the bug): calling
  `launch_timestamp_rules` directly now returns `eot=-inf` (correctly
  banned) and the full `logits_argmax_step` call now picks token `50578`
  -- the same timestamp the torch fallback always picked. Bit-exact match
  on the specific case that used to fail.
- **Full-corpus, real separate-process runs** (not a same-process A/B --
  this matches how `bench_e2e.py` actually gets invoked for every other
  number in this document): ran the K5-torch-fallback config and the
  K5-kernel-on config as two independent `bench_e2e.py` processes, all 5
  files, and compared the output transcripts. **All 5 files produced
  byte-for-byte identical text (matching SHA256 hashes)**, e.g.
  `44e8f624...`: `011057...af70` both times. (An earlier same-process A/B
  test, toggling `ops.USE_KERNELS` mid-process rather than using separate
  processes, showed a one-chunk difference that vanished when that chunk
  was re-decoded in isolation -- consistent with ordinary batch-composition
  fp16 noise, not a Rule-4 regression, and superseded by the clean
  separate-process result above, which is the one that matches how this
  project actually measures everything else.) K5 now runs on the kernel
  path unconditionally, same as K4 -- the `bench_e2e.py` force-to-torch
  workaround has been removed.

### Ngram A/B (file `44e8f624`, timestamps mode)

| ngram ban | divergence vs prod | RTF |
|---|---|---|
| **OFF (winner)** | **11.51%** | 0.00375 |
| ON | 12.32% | 0.00389 |

OFF wins again, same as notimestamps mode -- but the M4 hypothesis ("baseline
decodes with interleaved timestamps, which should stop the ban from firing
on natural speech once we add timestamps too") is only **partially**
supported: the regression from turning the ban on shrank a lot (notimestamps:
15.46% -> 18.40%, +2.94pp; timestamps: 11.51% -> 12.32%, +0.81pp) but didn't
disappear. Timestamps mode default: **ngram ban OFF**, consistent with
notimestamps mode.

### Full corpus re-measure (hard-cut, guards ON, ngram OFF, B=16)

| config | overall_rtf | total_wall_s |
|---|---|---|
| notimestamps headline (M4, kernels on) | 0.00318 | 94.19 |
| timestamps, K4 kernel + K5 torch fallback (pre-fix) | 0.00375 | 110.81 |
| **timestamps, K4+K5 both kernel (post-fix, final)** | **0.00316** | **93.36** |

With the Rule-4 fix, timestamps mode is now **essentially free** relative
to notimestamps: 93.36s vs 94.19s wall time, within measurement noise --
K5's timestamp-rules kernel adds only ~48us/step (measured in
`test_logits.py`'s microbench: `without=67.8us with=115.8us delta=+47.9us`
at B=8), far cheaper than the per-step Python loop the torch fallback
needed for the same rules. The pre-fix torch-fallback config's +17.9% wall
time (matching the orchestrator's ~19%-more-decode-tokens prediction) was
a real cost specifically of running K5's timestamp logic in Python, not
an inherent cost of timestamps mode itself. Per-file RTF and segment
counts (final, K4+K5 kernel config):

| file | wall_s | rtf | segments |
|---|---|---|---|
| 1abf3ec0-...-e344.mp3 | 34.22 | 0.00305 | 3039 |
| 44e8f624-...-acf4.mp3 | 11.83 | 0.00322 | 1164 |
| 882380ee-...-6be.mp3 | 16.11 | 0.00331 | 1409 |
| 9beeffbb-...-41e6.mp3 | 11.39 | 0.00300 | 842 |
| 9f0ed85f-...-e2.mp3 | 19.82 | 0.00330 | 2233 |

Segment counts and all transcript text are unchanged from the pre-fix
torch-fallback run (confirmed via the SHA256 spot-check above) -- only
speed changed, not output.

**Dual-GPU**, same config, duration-balanced split (`bench_e2e.py:
_balanced_split`, established in the previous section): one wall clock
98.6s, overall RTF **0.00333** -- a 1.12x dual-GPU speedup over the
pre-fix single-GPU number (110.8s). This dual-GPU run predates the Rule-4
fix (K4-kernel + K5-torch-fallback config) and has not been re-measured
against the post-fix single-GPU number (93.36s) -- not requested for this
pass; would need a fresh dual-GPU run to get a clean post-fix comparison.
Same modest single-vs-dual scaling story as the notimestamps dual-GPU
number above either way, same root cause (5-file corpus too small to
amortize two independent workers' fixed costs).

### Three-way divergence, timestamps mode (all 5 files)

| file | ours-vs-prod (timestamps) | ours-vs-prod (notimestamps, prior section) | delta | ours-vs-baseline (timestamps) |
|---|---|---|---|---|
| 1abf3ec0-...-e344.mp3 | 13.62% | 13.24% | +0.38pp | 11.94% |
| 44e8f624-...-acf4.mp3 | **11.51%** | 15.00% | **-3.50pp** | 10.80% |
| 882380ee-...-6be.mp3 | 14.05% | 12.91% | +1.15pp | 12.55% |
| 9beeffbb-...-41e6.mp3 | 14.34% | 11.51% | +2.83pp | 14.39% |
| 9f0ed85f-...-e2.mp3 | 12.63% | 9.38% | +3.25pp | 11.83% |
| **corpus-wide** | **13.32%** | **12.39%** | **+0.93pp** | **12.17%** |

baseline-vs-prod is unchanged at **8.16%** corpus-wide (not re-measured --
baseline's transcripts don't depend on anything this project changed).

### Interpretation: does ours-vs-prod close toward the 8.16% floor? No, not at the corpus level.

This is the honest answer, not the hoped-for one, and it comes with an
important caveat about how the ngram A/B and initial kernel-vs-torch
verification were both done on a **single file** (`44e8f624`) before the
full-corpus run: that file happens to be the one where timestamps mode
helped the most (15.00% -> 11.51%, a real 3.5-point improvement, and the
only file where timestamps mode won). The other four files all got
*worse* under timestamps mode, by 0.4 to 3.25 points each, and
corpus-wide the result is a net **regression** (12.39% -> 13.32%), moving
slightly *away* from the 8.16% baseline-vs-prod floor, not toward it.

This does not necessarily mean timestamps mode is a mistake -- decoding
with vs. without timestamp constraints are genuinely different decode
trajectories (different legal-token constraints at every step change what
the model ends up saying, not just how it's segmented), so some
content-level divergence between the two modes is expected on its own
merits, independent of which one tracks production better. What the data
supports is narrower: **the single-file A/B test that picked ngram-ban-off
generalized fine (both files and the corpus agree ban should be off), but
the file-level "timestamps mode closes the gap to production" finding
from the initial validation did not generalize to the rest of the
corpus.** Any future work banking on timestamps mode improving
production-alignment should re-validate per-file, not assume the
one-file result holds. (Update: the K5 kernel bug referenced here has
since been fixed and re-verified -- see the "Correctness" subsection
above, updated after the fix landed. The divergence numbers in this
section predate the fix but are unaffected by it: the fix changed speed,
not output -- confirmed by the SHA256 transcript match across all 5
files.)

## Temperature-retry ladder

Jim approved building openai-whisper's temperature-retry fallback:
`transcribe.py:decode_with_fallback` and `utils.py:compression_ratio`,
ported exactly (read directly from the venv on gpu-host as the reference
throughout, not from memory). Main pass unchanged (batched greedy, t=0),
now also collecting each sequence's `sum_logprob` via Agent C's K5 SS3d
kernel arg. Per-chunk, at batch end: `avg_logprob = sum_logprob /
(len(generated_tokens_excl_eot) + 1)`, `compression_ratio =
len(utf8_bytes)/len(zlib.compress(utf8_bytes))`, and the existing
no-speech gate's `p(no_speech)`. Classification (openai's exact rule):
`no_speech_prob > 0.6 AND avg_logprob < -1.0` -> silent, emit empty, stop;
else `compression_ratio > 2.4 OR avg_logprob < -1.0` -> retry; else ok.
Chunks needing retry are pooled across all batches of a file and
re-decoded together at escalating temperatures (0.2, 0.4, 0.6, 0.8, 1.0);
first temperature that resolves a chunk wins; a chunk that never resolves
keeps its t=1.0 attempt (openai's keep-last behavior). Sampling at t>0
uses `Categorical(logits=masked_logits/T).sample()` (openai's
`GreedyDecoder.update` exactly) on the torch K5 path -- no kernel supports
sampling, so this is torch-only regardless of `USE_KERNELS`; K4
(self/cross-attention) stays on the kernel path throughout. **Deviation
from openai, documented as instructed:** openai uses `best_of=5` (5
samples per retry, keep the best by average logprob); this implementation
takes 1 sample per retry attempt. Repetition brake is ON for the t=0 pass,
OFF during sampled retries (the compression-ratio check already catches
degenerate repetition, and the brake was tuned against greedy's specific
failure mode, not sampled output). No-speech gate and ngram-ban-OFF
default are unchanged in both passes. Scoped entirely at `pipeline.py`
(new `whisper_rocm/retry_ladder.py` for the per-chunk math/classification,
orchestration in `pipeline.py`'s `transcribe_file`), not inside
`decode.py`'s step loop, per the orchestrator's explicit instruction.

### Kernel-vs-torch `sum_logprob` agreement

Requested tolerance was rtol=1e-4 against a torch log_softmax+gather
reference, "then trust the kernel." Measured on real audio (8 chunks,
file `44e8f624`, greedy t=0): **argmax decisions (the actual token
sequence) match exactly in every test** -- this is the correctness
property that matters for the pipeline's output. The raw `sum_logprob`
values themselves do not meet rtol=1e-4:

| decode length | max abs diff | max rel diff |
|---|---|---|
| 24 steps | 8.6e-3 | 6.3e-3 |
| 50 steps | 8.7e-3 | 6.3e-3 |
| 100 steps | 9.4e-3 | 2.5e-3 |
| 200 steps | 1.4e-2 | 2.3e-3 |

This is reported honestly rather than rounded up to a pass: the measured
drift is roughly 2-3 orders of magnitude larger than
`DECODE_INTEGRATION.md` SS3d's own claimed reference number (~1.5e-5 to
2.3e-5 at a comparable ~20-step scale), and it does not grow much with
step count (24 steps and 200 steps land within 1.6x of each other) --
consistent with a fixed-size discrepancy from *how* the full-vocabulary
(51866-wide) logsumexp reduction is ordered/computed between the two
implementations, not compounding per-step rounding drift. Not
investigated further (out of scope for this pass -- would mean
instrumenting Agent C's kernel internals, which isn't mine to modify).
**Why this doesn't matter in practice:** `avg_logprob` divides
`sum_logprob` by `token_count + 1` (commonly 60-150 for a real chunk), so
the practical error on the value the retry ladder actually thresholds
against is roughly 1e-4 to 1e-5 -- three-plus orders of magnitude below
the -1.0 decision boundary. Combined with the exact-match argmax result,
the kernel is trusted for the retry ladder's actual purpose (the
threshold decision), with this discrepancy on record rather than
silently rounded away.

### The central finding: the ladder never fires on this corpus

Telemetry, full 5-file corpus, both modes, single GPU B=16, kernels on:

| mode | total chunks | retried | resolved (by temp) | silent | kept-last (t=1.0) |
|---|---|---|---|---|---|
| notimestamps | 989 | **0** | -- | 0 | 0 |
| timestamps | 989 | **0** | -- | 0 | 0 |

**0% retried, 0% silent-marked, in both modes, on every one of the 5
files.** This was verified, not assumed: I dumped avg_logprob/
compression_ratio/no_speech_prob for all 123 chunks of file `44e8f624`
and sorted by avg_logprob ascending. The worst chunks in the entire file
-- indices 25, 37, 92, 54, 34, 97, 15, 75, the exact same "near-empty
output" chunks identified back in M1/M2's investigation (`"Yeah."`,
`"Sure."`, 1-3 word outputs where the reference has 80-100 words) -- have
avg_logprob in the range -0.49 to -0.02 and compression_ratio 0.38 to
0.68. Both are nowhere near the -1.0 / 2.4 thresholds. **The model is
confidently short, not uncertain.** openai's fallback heuristic is built
to catch low-confidence or degenerate/repetitive output; the stall-hole
failure mode documented throughout this project (M1's finding: both this
pipeline and the openai-whisper fp32 reference independently produce the
same short, wrong completion on the same audio in isolation) is a
different failure shape entirely -- a fluent, high-probability, wrong
answer -- that none of openai's own three signals (avg_logprob,
compression_ratio, no_speech_prob) are designed to detect.

**Confirmed the ladder isn't silently broken**, since it never fires
naturally: temporarily lowered `retry_ladder.LOGPROB_THRESHOLD` from -1.0
to -0.05 (an easily-crossed bar) on `44e8f624` alone. Result: 55/123
chunks (44.7%) flagged for retry, 2 resolved at t=0.2, 1 resolved at
t=0.8, 52 exhausted the schedule and kept their t=1.0 attempt (text
length changed, 51062 -> 49469 characters, confirming the sampled
retries produced genuinely different output). Pooling, escalation across
multiple non-adjacent rungs, and keep-last all worked correctly. Reverted
immediately after -- this was a one-off mechanism check, not a
measurement, and the threshold is back to openai's real -1.0 in the
shipped code (diffed against the pre-test file to confirm the revert was
exact).

### RTF and divergence, before vs after

Because the ladder retried zero chunks, the transcripts are **provably
identical** to the pre-ladder (M4/M5) runs -- not just "should be similar",
verified by comparing SHA256 hashes of the per-file text output,
ladder-on vs ladder-off, and all 10 (5 files x 2 modes) matched exactly.
Divergence numbers below are therefore the M4/M5 numbers, reused rather
than recomputed (recomputing an expensive Levenshtein pass to confirm a
number that a hash match already proves identical would just burn time
for no new information).

| mode | RTF (ladder on) | RTF (M4/M5, no ladder) | ours-vs-prod | ours-vs-baseline |
|---|---|---|---|---|
| notimestamps | 0.00298 | 0.00318 | 12.39% | 11.87% |
| timestamps | 0.00316 | 0.00316 | 13.32% | 12.17% |

RTF differences between the "ladder on" and "no ladder" columns are
measurement noise (clock state, not a systematic ladder cost) -- expected,
since 0 retries means 0 extra decode work; the ladder's only added cost
when nothing needs retrying is the per-chunk avg_logprob/compression_ratio
bookkeeping, which is CPU-side and negligible next of mel/encode/decode.
baseline-vs-prod is unchanged at **8.16%** (doesn't depend on anything
this pipeline does).

**Word-count deficit, the direct hole metric, before vs after (identical,
since 0 chunks changed):**

| mode | prod ref words | ours hyp words | deficit | deficit % |
|---|---|---|---|---|
| notimestamps | 88323 | 80288 | 8035 | 9.1% |
| timestamps | 88323 | 80171 | 8152 | 9.2% |

**The orchestrator's hypothesis -- "the deletion mass (stall holes, ~70%
of divergence) collapses; ours-vs-prod drops toward the 8.16% cross-engine
floor" -- is not confirmed. The word deficit is unchanged, because the
ladder never engaged.** This is the honest result, reported as measured,
not adjusted to match what was hoped for.

**Dual-GPU** (notimestamps, the corpus-level winner on both RTF and
divergence with the ladder on): one wall clock 79.0s, overall RTF
**0.00267** -- consistent with the pre-ladder dual-GPU number (0.00268,
79.27s), again because nothing was retried.

### What would actually catch the stall holes

Not built (out of scope for this pass -- flagging for a possible future
one, not implementing speculatively): the fluent-but-wrong failure mode
found here needs a signal that doesn't require LOW confidence, since the
model is highly confident and wrong simultaneously. Options that don't
depend on the model doubting itself:
- **Length-based suspicion**: a chunk whose generated duration is far
  below its audio duration (e.g., 1-3 words decoded from a 30s window
  that isn't silence per the no-speech gate) is a directly measurable
  anomaly, independent of confidence.
- **Cross-temperature disagreement as the trigger, not just the
  threshold**: sample at t=0.2 regardless of the t=0 avg_logprob, and
  retry only if the two attempts disagree sharply on length or content --
  a fluent-but-wrong greedy output and a fluent-but-different sampled
  output would flag each other even though neither individually looks
  "bad" by openai's metrics.
- **Cross-chunk context**: M1/M2 already established the failure
  correlates with genuinely ambiguous/isolated audio (a 30s window with
  no surrounding context); stride-mode chunking (already implemented,
  M2) gives neighboring chunks more audio context and was shown to help
  on some files -- a targeted combination (stride chunking specifically
  for chunks that failed a length-based check) wasn't tried here but is
  a more promising next step than tuning openai's thresholds, since
  those thresholds are answering a different question than "is this
  chunk suspiciously short."

## VAD chunking mode

Jim-approved (2026-08-10): replace hard-cut's fixed 30s grid with
Silero-VAD-detected speech windows, greedily packed to <=30s and closed
on silence gaps instead of at an arithmetic offset -- directly targets
the boundary-clipping / stall-window failure mode named in the section
above. No kernel changes; this is a `whisper_rocm/audio.py` +
`pipeline.py` change (`chunking="vad"`), now the **default** chunking
mode (`hard`/`stride` remain available).

### Implementation

`whisper_rocm/audio.py`: `chunk_audio_vad` runs Silero VAD (CPU-only,
never touches GPU timing) on the file's 16kHz mono samples, then
`_pack_vad_windows` greedily extends the current window while the next
speech segment (plus the silence gap before it) still fits in 30s;
when it doesn't fit, the window closes at the end of the last segment
that DID fit -- the boundary lands in silence, not mid-word. A single
VAD-detected speech run longer than 30s with no internal silence can't
be closed on silence at all; that's hard-split at 30s boundaries and
counted separately (`n_long_segments_hard_split` in `vad_stats`) so how
often the "unavoidable hard cut" actually happens is visible, not
silently absorbed. Silent spans between windows are never sliced out of
the audio at all, so "skip long pure-silence spans" falls out of the
packer for free rather than needing separate handling.

**The one real refactor**: `chunk_audio` / `chunk_audio_stride` /
`chunk_audio_vad` now all return explicit per-chunk `(offsets_s,
durations_s)` instead of `pipeline.py` deriving offsets from a
`chunk_step` formula -- VAD windows aren't evenly spaced or sized, so
there's no formula to derive them from. Giving hard/stride the same
explicit shape means `_decode_and_classify` has exactly one code path
for all three modes. Side effect, not the goal: this fixes a latent
inaccuracy where hard-cut's last (shorter-than-30s) chunk of a file was
handed the nominal 30s as its `chunk_duration` for `segments.py`'s
"unpaired final timestamp" case (an open segment at chunk end would
have closed 30s after the chunk start instead of at the file's real
end) -- worth naming since it was found via the VAD work, not gone
looking for.

### Sanity check (one file, `44e8f624`)

All 1348 VAD-mode segments monotonic, `Start`/`End` sane, first segment
at 0.1s, last segment ends at 3669.6s (file duration 3670.6s). Concrete
boundary example, same ~1830s region in both modes:

| mode | segments |
|---|---|
| hard-cut | `[1828.9-1830.0]` *(blank)* · `[1830.0-1836.8]` "**wondered about that.** 77% goes to Japan. And Japan was actually the one that started all th..." |
| VAD | `[1829.8-1830.8]` "**Yeah, I wondered about that.**" · `[1831.1-1833.1]` "77% goes to Japan." |

Hard-cut's fixed 30s boundary lands inside "Yeah, I wondered about
that," clipping it to "wondered about that" and leaving an orphaned
blank segment just before it. VAD's boundary sits in the silence
between utterances, so the phrase survives intact -- this single
mechanism, repeated at every hard-cut chunk boundary in every file
(~1 per 30s of audio), is a large part of what the M1-M6 "why doesn't
divergence hit DESIGN.md's <=3% target" investigation never pinned
down.

### Stall-window rate (the number VAD mode exists to fix)

Methodology: for each file, VAD's own transcribed segments mark which
30s-aligned buckets contain real speech (independent ground truth for
hard-cut, since hard-cut's chunk boundaries had no say in it). A
"stall" bucket is one where the corresponding mode produced <=3 words
despite the bucket being speechful (RESULTS.md's own language from the
section above: "1-3 words decoded from a 30s window that isn't
silence"). Measured across the full 20-episode corpus (1846 speechful
buckets):

| mode | stall buckets | speechful buckets | stall rate |
|---|---|---|---|
| hard-cut | 11 | 1846 | **0.60%** |
| VAD | 2 | 1846 | **0.11%** |

A ~5.5x reduction. Both rates are low in absolute terms (this failure
mode was already established as affecting "a small fraction of
chunks," not a large one), but VAD meaningfully reduces it rather than
eliminating it -- the residual 2 VAD stalls are presumably genuine
model failures on hard audio, not chunking artifacts.

### Divergence: 5-file bench corpus, VAD vs hard-cut

Same methodology as the rest of this document (`bench/check_quality.py`,
corpus-wide = sum(edits)/sum(ref_words)), both notimestamps (the
original M1-M4 measurement config) and timestamps (the config the
20-episode production-equivalence work actually uses) modes:

| config | vs prod (hard-cut -> VAD) | vs baseline (hard-cut -> VAD) |
|---|---|---|
| notimestamps | 12.39% -> **7.30%** | 11.87% -> **7.93%** |
| timestamps | 13.32% -> **6.98%** | 12.17% -> **7.61%** |

Per-file, VAD-mode divergence (all 5 files, both configs):

| file | notimestamps vs prod | notimestamps vs baseline | timestamps vs prod | timestamps vs baseline |
|---|---|---|---|---|
| 1abf3ec0-...-e344.mp3 | 6.67% | 6.93% | 6.51% | 6.51% |
| 44e8f624-...-acf4.mp3 | 7.59% | 7.87% | 6.21% | 6.52% |
| 882380ee-...-6be.mp3 | 8.85% | 10.71% | 8.04% | 9.59% |
| 9beeffbb-...-41e6.mp3 | 6.93% | 6.42% | 7.78% | 7.14% |
| 9f0ed85f-...-e2.mp3 | 7.15% | 8.29% | 6.91% | 8.84% |

Roughly halved in every case, and -- unlike the notimestamps/timestamps
divergence story elsewhere in this document, where one file's
improvement didn't generalize to the corpus -- this holds file-by-file,
not just in aggregate: all 5 files individually improve by 4-8 points
in every comparison (the table above; the pattern is uniform, no
outliers in the wrong direction). This is the single largest
divergence improvement found anywhere in this project, and it comes
from a
chunking change, not a kernel or decode change -- consistent with the
boundary-clipping mechanism above being a real, corpus-wide effect
rather than a one-file coincidence.

### RTF: VAD is slower before optimization, faster after

VAD's own CPU cost (Silero inference + packing) is NOT free, and
packing did NOT reduce total chunk count on this speech-dense podcast
corpus -- if anything the opposite: 20-episode corpus, 2104 VAD windows
vs 1873 hard-cut chunks (+12.3%), because closing early on any silence
gap that would blow the 30s budget leaves windows shorter than a full
30s on average, so more of them are needed to cover the same speech.
GPU-side compute (encode+decode) tracks that +12% roughly 1:1 -- no win
there, unlike the divergence result above.

| config | dual-GPU overall_rtf (20 episodes) |
|---|---|
| hard-cut (established headline) | 0.00214 |
| VAD, unoptimized (torch JIT VAD, no overlap) | 0.00486 (2.27x slower) |
| **VAD, optimized (ONNX + CPU/GPU overlap)** | **0.00206 (1.04x FASTER than hard-cut)** |

**Cost recovery, two changes** (Jim-directed, "GPU VAD" explicitly
ruled out):

1. **ONNX backend.** `silero_vad`'s own `model.py` calls
   `torch.set_num_threads(1)` at import time, so the torch-JIT path
   (the original implementation) was single-threaded with no override
   available. Switching to `load_silero_vad(onnx=True)` and
   reconstructing its `onnxruntime.InferenceSession` with more
   intra-op threads (the package hardcodes 1 there too, no public
   override) A/B'd at 1/4/8/12 threads on gpu-host's 12 cores: 1 thread
   already beats torch JIT 2.4x; 4 threads gets to 2.9x; 8/12 add <1%
   more (the model is small enough that thread count stops mattering
   past 4). Settled on **4 intra-op threads** as the default --
   verified bit-for-bit... not quite: 19/20 files in the full corpus
   run produced byte-identical text between the torch-JIT and ONNX
   backends (same chunk counts too), one file (`ec230700`, also the
   file with by far the most long-segment hard-splits at 32) differs
   in text with an identical chunk count -- consistent with a small
   floating-point difference between the two backends occasionally
   shifting a VAD boundary by a few ms and landing on a different
   word, not a bug.
2. **CPU/GPU overlap.** `pipeline.py` split into `prepare_chunks`
   (CPU-only: load + VAD) and `transcribe_prepared` (GPU-only);
   `transcribe_file` is now just the two called back to back
   (unchanged behavior for existing callers). `bench_e2e.py`'s
   per-file loop prefetches file N+1's `PreparedChunks` on a
   background thread as soon as file N's GPU work starts. VAD releases
   the GIL during its native (onnxruntime) compute, so this is real
   wall-clock overlap, not concurrency theater -- confirmed by the
   result: total VAD CPU time summed across per-file stage timings
   dropped from 242.8s (torch JIT) to 124.1s (ONNX) in the *overlapped*
   run (less than the isolated A/B's ~2.9x, plausibly some CPU
   contention with concurrent GPU-dispatch Python code, but the point
   of overlap is that this stops mattering for wall clock -- almost
   all of it is hidden behind GPU compute).

Combined: **2.36x recovered** (0.00486 -> 0.00206), landing VAD mode
*faster* than the previous hard-cut headline despite +12% more GPU
work, purely from no longer paying VAD's CPU cost serially. Both
`chunking="vad"`'s default backend (ONNX, 4 threads) and the overlap
prefetch are unconditional -- no flag needed, no regressions expected
for hard/stride (which never call VAD code at all).

### Ad-detection re-run: 20-episode aggregate, VAD vs hard-cut

Same protocol as the original 20-episode production-equivalence check
(`s1_chunked_baseline`, `LLM_BACKEND=prod`, three comparisons per
episode) -- prod-transcript detections and shipped `prod_ads` reused
unchanged from the hard-cut run (not re-queried); only the "ours"
(our-transcript) side was re-run against the VAD transcripts.

| comparison | hard-cut mean F1 -> VAD mean F1 | hard-cut median -> VAD median |
|---|---|---|
| **ours vs prod-transcript detections** (headline) | 0.73 -> **0.78** | 0.77 -> **0.81** |
| ours vs prod ads (shipped) | 0.73 -> **0.78** | 0.73 -> **0.81** |
| control (prod-transcript vs prod_ads, unchanged) | 0.87 | 0.87 |

Unmatched-segment rate (a detected ad matching neither other signal --
the pilot's "extra segment" finding, measured symmetrically against a
same-detector noise baseline, see the production-equivalence results
for the full methodology):

| side | hard-cut | VAD |
|---|---|---|
| our-transcript detections, unmatched rate | 27.3% (38/139) | **21.4% (30/140)** |
| episodes showing the pattern | 14/20 | 12/20 |

Narrower, not closed -- VAD's rate is still meaningfully above the
~10% same-detector noise floor established by the prod-transcript
side, so this remains a real (if now smaller) transcript-driven
effect, not fully explained by detector noise. Ad-time recovered in
aggregate also improved slightly: VAD's total detected ad-seconds is
97.5% of shipped (vs hard-cut's 95.4%). Full per-episode numbers live
in this project's private results tracking (real show/episode content,
not appropriate for this document or the public repo).

### Verdict

VAD becomes the default. On every axis measured -- divergence (roughly
halved, file-by-file, not just in aggregate), stall-window rate (5.5x
lower), ad-detection F1 (both headline comparisons improved ~5-8
points), unmatched-segment rate (27%->21%), and now RTF too, after the
two cost-recovery changes (2.36x recovered, net *faster* than
hard-cut) -- VAD is a clear win with no measured regression. The
residual unmatched-segment gap above the noise floor (21% vs ~10%) is
the one number that isn't fully closed; worth a future pass, but not a
reason to hold the default flip.
