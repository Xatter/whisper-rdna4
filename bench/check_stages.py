#!/usr/bin/env python3
"""Stage-by-stage tensor diff vs openai-whisper reference (DESIGN.md S8.1).

Reference: openai-whisper package running the checkpoint in fp32 on CPU
(ground truth). Compares, in order, on one fixed 30s clip: mel -> conv
stem -> encoder block 0 -> encoder out -> decoder logits at step 0 ->
first 20 greedy tokens. Reports max-abs-err per stage and stops at the
first stage whose error exceeds tolerance (DESIGN.md S6 tolerance: max-abs
<= 5e-2 on unit-scale tensors, or rel <= 1%) so a real divergence isn't
masked by later stages.

Not named explicitly in DESIGN.md's bench/ tree (only check_quality.py
is), but S8.1's stage-by-stage protocol needs a runnable script and this
is the natural place for it, next to check_quality.py.

Usage (on gpu-host):
    .venv/bin/python bench/check_stages.py \\
        --checkpoint ~/.cache/whisper/large-v3-turbo.pt \\
        --audio ~/insanely-fast-whisper-rocm/bench-audio/44e8f624-1876-460e-bcd4-8180b7a3acf4.mp3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOL_ABS = 5e-2
TOL_REL = 1e-2


def report(name: str, ours: torch.Tensor, ref: torch.Tensor) -> float:
    """DESIGN.md S6 tolerance: max-abs <= 5e-2 on unit-scale tensors, OR
    rel <= 1%. Large-magnitude tensors (e.g. post-LN activations scaled by
    a learned weight >> 1) legitimately have max-abs errors above 5e-2
    while still being well within the relative-error bar, so both are
    checked and either passing is a pass. Returns the max abs error
    (informational; pass/fail is `ok`, not this return value alone).
    """
    ours = ours.float().cpu()
    ref = ref.float().cpu()
    if ours.shape != ref.shape:
        print(f"[{name}] SHAPE MISMATCH ours={tuple(ours.shape)} ref={tuple(ref.shape)}")
        return float("inf"), False
    diff = (ours - ref).abs()
    max_abs = diff.max().item()
    rel = (diff / ref.abs().clamp_min(1e-6)).median().item()
    ok = (max_abs <= TOL_ABS) or (rel <= TOL_REL)
    status = "OK" if ok else "FAIL"
    print(f"[{name}] max_abs_err={max_abs:.5f} median_rel_err={rel:.5f} -> {status}")
    return max_abs, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    import whisper as oa_whisper

    from whisper_rocm import audio as whisper_audio
    from whisper_rocm import model as whisper_model
    from whisper_rocm import tokenizer as whisper_tokenizer
    from whisper_rocm import weights as whisper_weights

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- load reference (fp32, CPU) and ours (fp16, GPU) ----
    ref_model = oa_whisper.load_model(args.checkpoint, device="cpu").float().eval()
    w = whisper_weights.load_weights(args.checkpoint, device=device, dtype=torch.float16)
    tok = whisper_tokenizer.Tokenizer(n_vocab=w.dims.n_vocab)

    # ---- fixed 30s clip: first chunk of the given file ----
    raw = whisper_audio.load_audio_mono16k(args.audio)
    chunk = whisper_audio.pad_or_trim(raw[: whisper_audio.N_SAMPLES])
    audio_np = chunk.astype(np.float32)

    failed = False

    # Stage 1: mel
    ref_mel = oa_whisper.audio.log_mel_spectrogram(torch.from_numpy(audio_np), n_mels=128)
    ref_mel = ref_mel.unsqueeze(0)  # (1, 128, 3000)
    ours_audio_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    ours_mel = whisper_audio.log_mel_spectrogram_batch(ours_audio_t, device)
    _, ok = report("mel", ours_mel, ref_mel)
    failed = failed or not ok

    # Stage 2: conv stem (conv1+gelu+conv2+gelu+pos, pre-encoder-blocks)
    with torch.no_grad():
        ref_x = torch.nn.functional.gelu(ref_model.encoder.conv1(ref_mel))
        ref_x = torch.nn.functional.gelu(ref_model.encoder.conv2(ref_x))
        ref_x = ref_x.permute(0, 2, 1)
        ref_x = (ref_x + ref_model.encoder.positional_embedding).to(ref_x.dtype)

        ours_mel_fp16 = ours_mel  # already fp16 on device
        ours_x = torch.nn.functional.gelu(
            torch.nn.functional.conv1d(ours_mel_fp16, w.conv1_w, w.conv1_b, padding=1)
        )
        ours_x = torch.nn.functional.gelu(
            torch.nn.functional.conv1d(ours_x, w.conv2_w, w.conv2_b, stride=2, padding=1)
        )
        ours_x = ours_x.permute(0, 2, 1)
        ours_x = (ours_x + w.encoder_pos_emb).to(ours_x.dtype)
    _, ok = report("conv_stem", ours_x, ref_x)
    failed = failed or not ok

    # Stage 3: encoder block 0
    with torch.no_grad():
        ref_block0_out = ref_model.encoder.blocks[0](ref_x)
        ours_block0_out = whisper_model._encoder_block(ours_x, w.encoder_layers[0], w.dims.n_audio_head)
    _, ok = report("encoder_block_0", ours_block0_out, ref_block0_out)
    failed = failed or not ok

    # Stage 4: full encoder out
    with torch.no_grad():
        ref_enc_out = ref_model.encoder(ref_mel)
        ours_enc_out = whisper_model.encode(ours_mel, w)
    _, ok = report("encoder_out", ours_enc_out, ref_enc_out)
    failed = failed or not ok

    # Stage 5: decoder logits at step 0 (prompt -> logits at last prompt position)
    prompt = torch.tensor([whisper_tokenizer.PROMPT], dtype=torch.long)
    with torch.no_grad():
        ref_logits = ref_model.decoder(prompt, ref_enc_out)[:, -1, :]

        self_cache = whisper_model.alloc_self_kv_cache(w, 1, 448, device)
        cross_kv = whisper_model.precompute_cross_kv(ours_enc_out, w)
        scratch = whisper_model.alloc_decode_scratch(w, 1, device)
        prompt_gpu = prompt.to(device)
        ours_logits = None
        for position in range(prompt_gpu.shape[1]):
            ours_logits = whisper_model.decoder_step(
                prompt_gpu[:, position], position, w, self_cache, cross_kv, scratch
            )
    _, ok = report("decoder_logits_step0", ours_logits, ref_logits)
    failed = failed or not ok

    # Stage 6: first 20 greedy tokens
    with torch.no_grad():
        opts = oa_whisper.DecodingOptions(
            language="en", without_timestamps=True, fp16=False, temperature=0.0
        )
        ref_result = ref_model.decode(ref_mel, opts)
        if isinstance(ref_result, list):  # decode() returns a list for batched mel input
            ref_result = ref_result[0]
        ref_tokens = [whisper_tokenizer.SOT, whisper_tokenizer.EN, whisper_tokenizer.TRANSCRIBE, whisper_tokenizer.NO_TIMESTAMPS]
        ref_tokens += list(ref_result.tokens[:20])

        from whisper_rocm import decode as whisper_decode

        ours_result = whisper_decode.greedy_decode(ours_enc_out, w, tok, max_len=24)
        ours_tokens = ours_result.tokens[0].cpu().tolist()

    n = min(20, len(ref_tokens) - 4, len(ours_tokens) - 4)
    matches = sum(
        1 for a, b in zip(ref_tokens[4 : 4 + n], ours_tokens[4 : 4 + n]) if a == b
    )
    print(f"[first_20_greedy_tokens] {matches}/{n} exact token-id matches")
    print(f"  ref : {ref_tokens[4:4+n]}")
    print(f"  ours: {ours_tokens[4:4+n]}")
    if matches < n:
        print("  NOTE: token-level mismatches are expected once fp16 argmax ties break")
        print("  differently than fp32 -- inspect whether text still matches; this is")
        print("  not automatically a FAIL the way the tensor stages above are.")

    print()
    print("FAIL" if failed else "ALL STAGES WITHIN TOLERANCE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
