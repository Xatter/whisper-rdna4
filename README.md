# whisper-rdna4

Fast Whisper **large-v3-turbo** inference for AMD RDNA4 GPUs (gfx1201 — Radeon
AI PRO R9700, RX 9070 XT), built on ROCm + PyTorch with custom fused HIP
decode kernels.

Measured on 2x Radeon AI PRO R9700 over an 8.2-hour podcast corpus:

| Configuration | RTF | Real-time multiple |
|---|---|---|
| HF `transformers` pipeline, fp16 (same machine, reference) | 0.0488 | 20x |
| **This repo, one R9700, batch 16, hard-cut chunking** | **0.00316** | **316x** |
| **This repo, two R9700s, hard-cut chunking** | **0.00267** | **374x** |

RTF = wall seconds / audio seconds, measured after model load, including audio
decode, mel, encoder, decoder, and detokenization. Output includes
word-timestamped segments (`{Start, End, Text}`); timestamps cost nothing (the
timestamp rules run inside the logits kernel, +48 µs/step).

**VAD chunking is the default as of the numbers below** (`chunking="vad"`,
see "Chunking modes"). Measured on a larger, 15.5-hour real-podcast corpus,
two R9700s, both configs otherwise identical (kernels on, batch 16,
timestamps on):

| Configuration | RTF |
|---|---|
| Hard-cut chunking (`--chunking hard`, still available) | 0.00214 |
| **VAD chunking (default), unoptimized (no ONNX/overlap)** | 0.00486 |
| **VAD chunking (default), with ONNX backend + CPU/GPU overlap** | **0.00206** |

VAD's own CPU cost (Silero VAD inference + packing) made it 2.3x *slower*
end to end at first — packing didn't reduce chunk count on this speech-dense
content (+12% more chunks than hard-cut, if anything), so the whole win had
to come from not paying VAD's CPU cost serially. Two changes recovered it
2.36x, landing VAD chunking net *faster* than hard-cut despite the extra
chunks: switching the VAD backend from Silero's torch-JIT path (hardcoded to
1 CPU thread by the package itself) to onnxruntime with 4 intra-op threads
(2.9x on its own), and overlapping file N+1's CPU-side VAD/audio-load with
file N's GPU work via a background-thread prefetch in the per-file loop.
See `RESULTS.md`'s "VAD chunking mode" section for the full breakdown.

## How it's fast

1. **Batch tiling.** 30 s chunks are processed 16 at a time. The encoder's
   linear layers become 24,000-row GEMMs that saturate the matrix cores
   through hipBLASLt. The decoder decodes 16 chunks in lock-step, sharing
   every weight read (~290 MB/token fp16) across the batch — batch-1 decode is
   memory-bandwidth-bound at ~2 ops/byte, so sharing weight reads is worth
   ~an order of magnitude on its own.
2. **Custom fused decode kernels** (`kernels/`, HIP, wave32, full occupancy):
   - `attention_decode.hip` — KV-cache append + softmax(qKᵀ/8)·V in one
     launch per attention (6 torch launches → 1; 2.1–2.4x self-attn,
     1.35–1.41x cross-attn vs the torch op sequence).
   - `logits.hip` — suppress mask + optional 3-gram ban + optional OpenAI
     timestamp rules + argmax + per-sequence log-prob accumulation +
     finished-flag maintenance in one launch. The 51,866-wide logit tensor
     never leaves the GPU.
3. **GPU-resident decode loop.** No per-step host syncs; the host polls a
   lagged "all finished" flag every 8 steps.

An honest note: hand-written WMMA GEMM and flash-attention kernels for the
*encoder* are also in `kernels/` — measured, they **lose** to hipBLASLt and
PyTorch SDPA on ROCm 7.2 (0.2–0.7x) and are disabled by default. The shipped
wins are all decode-side fusion. See `RESULTS.md` for every measurement,
including the negative results.

## Chunking modes

Three chunking modes are available (`chunking=` in `pipeline.py` /
`--chunking` in `bench_e2e.py`); **VAD is the default**.

- **`vad` (default).** Silero VAD (CPU-only, ONNX backend) finds real speech
  segments; a greedy packer merges consecutive segments into ≤30s windows,
  closing each window on a silence gap instead of at a fixed offset. Long
  uninterrupted speech runs with no silence get hard-split at 30s (rare,
  counted separately).
- **`hard`.** Fixed, non-overlapping 30s chunks — simple, still available,
  no CPU-side VAD cost.
- **`stride`.** 30s windows, 5s overlap each side, word-level LCS merge.

**Why VAD is the default.** Hard-cut's fixed 30s boundary doesn't know where
words end. On a real chunk boundary from this project's bench corpus, hard
cutting split "*Yeah, I wondered about that*" into an orphaned blank segment
followed by "*wondered about that*" — the words before the cut are just
gone. VAD's boundary falls in the silence between utterances instead, so the
phrase survives intact. That's not a cherry-picked example — measured across
a real podcast corpus, this single mechanism (repeated at every ~30s
boundary, in every file) tracks with a **~5.5x drop in near-empty output on
speechful audio** (0.60% → 0.11% of 30s windows) and **transcript divergence
against two independent reference transcripts roughly halved** (both
externally-measured references improved by 4–8 points each, consistently,
not just in aggregate — see `RESULTS.md`'s "VAD chunking mode" section for
the full numbers and methodology). VAD's own CPU cost is real (see the RTF
table above) but fully recovered — and then some — by an ONNX backend switch
and CPU/GPU prefetch overlap, both on by default.

## Requirements

- An RDNA4 GPU (gfx1201): Radeon AI PRO R9700 or RX 9070 XT
- Linux with **ROCm 7.2+** at `/opt/rocm` (hipcc must support
  `--offload-arch=gfx1201`)
- Python 3.12
- **PyTorch built for the same ROCm major version as your system toolchain.**
  This matters: the custom kernels are loaded via ctypes into the torch
  process and must share its HIP runtime. Mixed versions fail with HSA symbol
  errors.

## Setup

```bash
python3.12 -m venv .venv
# torch MUST come from the matching ROCm index (and only that index —
# adding PyPI as an extra index can silently resolve the CUDA build):
.venv/bin/pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
.venv/bin/pip install openai-whisper soundfile scipy tiktoken numpy
.venv/bin/pip install silero-vad onnxruntime  # VAD chunking (default mode)
```

`silero-vad` pulls in `torchaudio` as a transitive dependency; on some ROCm
torch builds the pip-resolved `torchaudio` wheel's compiled extension is
ABI-incompatible and fails to import (`OSError: Could not load this
library`). `whisper_rocm/audio.py` works around this automatically (Silero's
own code only needs `torchaudio` for helpers this project never calls) — no
action needed, but it's worth knowing that error is expected and harmless if
you see it while experimenting directly with `silero_vad`.

Verify the GPU is visible (mask any iGPU — device enumeration includes it):

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Get the checkpoint (OpenAI format):

```bash
.venv/bin/python -c "import whisper; whisper.load_model('large-v3-turbo')"
# downloads to ~/.cache/whisper/large-v3-turbo.pt
```

Build the kernels:

```bash
cd kernels && bash build.sh   # hipcc → libwhisper_kernels.so
```

Run the kernel test suites (each prints PASS/FAIL + a microbenchmark table):

```bash
HIP_VISIBLE_DEVICES=0 ../.venv/bin/python test_attention_decode.py
HIP_VISIBLE_DEVICES=0 ../.venv/bin/python test_logits.py
```

## Transcribe

```python
import os, torch
from whisper_rocm import weights, tokenizer, pipeline, ops, kernel_bridge

kernel_bridge.register()          # register the fused decode kernels...
ops.USE_KERNELS = True            # ...and switch them on

device = torch.device("cuda:0")
w = weights.load_weights(
    os.path.expanduser("~/.cache/whisper/large-v3-turbo.pt"), device=device)
tok = tokenizer.Tokenizer(n_vocab=w.dims.n_vocab)

result = pipeline.transcribe_file(
    "episode.mp3", w, tok, device,
    batch_size=16,
    timestamps=True,              # segment output; costs nothing
)
print(result.text)
for seg in result.segments:
    print(f"[{seg['Start']:.2f} - {seg['End']:.2f}] {seg['Text']}")
```

(See `whisper_rocm/pipeline.py` for the full option list: chunking mode,
3-gram ban, temperature-retry ladder; the no-speech gate and repetition brake
are on by default.)

## Benchmark

```bash
HIP_VISIBLE_DEVICES=0,1 .venv/bin/python bench/bench_e2e.py \
    --mode dual \
    --corpus-dir /path/to/audio \
    --checkpoint ~/.cache/whisper/large-v3-turbo.pt \
    --batch-size 16 --use-kernels
```

`--mode single` runs one GPU. The output JSON has per-file and overall RTF
plus per-stage timings (audio/mel/encode/decode/detokenize).

## Repo layout

```
kernels/          HIP kernels + per-kernel test/microbench suites
  attention_decode.hip   fused decode self/cross attention (ships, on)
  logits.hip             fused logits epilogue: suppress/ngram/timestamps/
                         argmax/logprob/finished (ships, on)
  gemm_wmma.hip          WMMA GEMM w/ fused epilogues (loses to hipBLASLt; off)
  attention_encoder.hip  encoder flash attention (loses to SDPA; off)
  layernorm.hip          fused LayerNorm (ties torch; off)
  DECODE_INTEGRATION.md  exact kernel API contract
whisper_rocm/     model, batched pipeline, GPU-resident decode loop
bench/            end-to-end benchmark + transcript-divergence tooling
DESIGN.md         architecture rationale (roofline analysis, fusion plan)
RESULTS.md        every measurement, including what didn't work
```

## Accuracy

Greedy decoding, fp16. Token output matches the fp32 openai-whisper reference
exactly on direct comparisons. On real podcast audio with hard-cut chunking,
transcript divergence from other Whisper engines is ~12% word-level (the
noise floor between any two independent fp16 Whisper engines on this corpus
measures ~8%). **With VAD chunking (the default), that divergence is roughly
halved against both references** — see "Chunking modes" above and
RESULTS.md's "VAD chunking mode" section for the full quality analysis and
its caveats — divergence is not an error rate, and no side is ground truth.

## License

MIT — see LICENSE.
