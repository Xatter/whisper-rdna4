# whisper-rdna4

Fast Whisper **large-v3-turbo** inference for AMD RDNA4 GPUs (gfx1201 — Radeon
AI PRO R9700, RX 9070 XT), built on ROCm + PyTorch with custom fused HIP
decode kernels.

Measured on 2x Radeon AI PRO R9700 over an 8.2-hour podcast corpus:

| Configuration | RTF | Real-time multiple |
|---|---|---|
| HF `transformers` pipeline, fp16 (same machine, reference) | 0.0488 | 20x |
| **This repo, one R9700, batch 16** | **0.00316** | **316x** |
| **This repo, two R9700s** | **0.00267** | **374x** |

RTF = wall seconds / audio seconds, measured after model load, including audio
decode, mel, encoder, decoder, and detokenization. Output includes
word-timestamped segments (`{Start, End, Text}`); timestamps cost nothing (the
timestamp rules run inside the logits kernel, +48 µs/step).

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
```

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
exactly on direct comparisons. On real podcast audio, transcript divergence
from other Whisper engines is ~12% word-level (the noise floor between any two
independent fp16 Whisper engines on this corpus measures ~8%; see RESULTS.md
for the full quality analysis and its caveats — divergence is not an error
rate, and no side is ground truth).

## License

MIT — see LICENSE.
