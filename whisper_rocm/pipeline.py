"""Per-file execution flow (DESIGN.md S5): load+resample -> chunks ->
batched mel/encode/decode -> detokenize -> concatenate. Not a file DESIGN.md
names explicitly in S5's tree, but S5's "Execution flow per audio file"
section has to live somewhere; bench/bench_e2e.py and the correctness
harness both need it, so it belongs in whisper_rocm/ rather than being
duplicated in bench/.

M2 addition: `chunking="stride"` (orchestrator directive 2026-08-09) uses
30s windows with 5s overlap on each side (audio.chunk_audio_stride) instead
of the DESIGN.md-specified hard non-overlapping cut, then stitches
consecutive chunks' text with a word-level longest-common-subsequence
merge (_lcs_merge below) instead of transformers' timestamp-based merge --
this was root-caused during M1 as the dominant explanation for the WER gap
vs baseline (baseline's HF pipeline defaults to exactly this 5s stride,
verified in transformers' automatic_speech_recognition.py source).

M5 addition: `timestamps=True` (orchestrator directive 2026-08-09) decodes
with openai-whisper's timestamp grammar (see decode.py, whisper_rocm.ops's
apply_timestamp_rules torch fallback / K5's SS3c extension) and additionally
produces a `segments` list ({"Start", "End", "Text"} dicts, absolute
whole-file seconds -- same shape as Castria's own
`data/episodes/{uuid}.json` `transcript` field, so this is drop-in for
ad-detection input). Segment extraction (whisper_rocm.segments) is only
exercised for chunking="hard" in practice (the M5 headline config); with
chunking="stride" segments from overlapping windows are concatenated
as-is, without the overlap-aware merge `_lcs_merge` does for plain text --
not specially handled, since nothing in scope needs it yet.

M6 addition (orchestrator directive, 2026-08-09): temperature-retry
ladder, scoped at THIS level per explicit instruction (not inside
decode.py's step loop). After the main greedy (t=0) pass, each chunk gets
classified ok/retry/silent (whisper_rocm.retry_ladder.classify, an exact
port of openai-whisper's decode_with_fallback rule). Chunks needing retry
are pooled across ALL batches of the file and re-decoded together at
escalating temperatures (0.2, 0.4, 0.6, 0.8, 1.0 -- openai's default
schedule); the first temperature that resolves a chunk wins, and a chunk
that never resolves keeps its t=1.0 attempt (openai's keep-last behavior).
Guard interactions (orchestrator-specified): repetition brake ON for the
t=0 pass, OFF during sampled retries (t>0) -- the compression-ratio check
already catches degenerate repetition, and the brake's period-1..8 cycle
detector was tuned against greedy's failure mode, not sampled output, so
turning it off during retries avoids it fighting the sampler for a
guard whose job the compression check already covers. No-speech gate is
unchanged in both passes (it's part of decode.py, orthogonal to
temperature). Ngram ban stays at whatever the caller passes (default
OFF, per the M4/M5 decision) in both passes.

VAD addition (Jim-approved, orchestrator directive 2026-08-10):
`chunking="vad"` (whisper_rocm.audio.chunk_audio_vad) replaces hard-cut's
fixed 30s grid / stride's fixed-step overlap with Silero-VAD-detected
speech windows, greedily packed to <=30s and closed on silence gaps
instead of mid-word. The one real refactor this required: chunk_audio /
chunk_audio_stride / chunk_audio_vad all now return explicit per-chunk
(offsets_s, durations_s) instead of this file deriving offsets from a
`chunk_step` formula -- VAD windows aren't evenly spaced or sized, so
there's no formula to derive them from, and giving hard/stride the same
explicit shape means _decode_and_classify has exactly one code path
(and, as a side effect, fixes a latent inaccuracy where the last,
shorter-than-30s chunk of a hard-cut file was handed the nominal 30s as
its chunk_duration for segments.py's "unpaired final timestamp" case).
Everything downstream of chunking (batching, kernels, timestamps,
guards, retry ladder) is unchanged -- VAD windows are still independent
30s-mel-padded chunks to the model, only the windows' boundaries differ.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from . import audio as whisper_audio
from . import decode as whisper_decode
from . import model as whisper_model
from . import retry_ladder
from . import segments as whisper_segments
from . import tokenizer as whisper_tokenizer
from .weights import WhisperWeights

ChunkingMode = Literal["hard", "stride", "vad"]


@dataclass
class StageTimings:
    load_s: float = 0.0
    vad_s: float = 0.0  # CPU-only Silero VAD + packing time; 0 for hard/stride
    mel_s: float = 0.0
    encode_s: float = 0.0
    decode_s: float = 0.0
    detok_s: float = 0.0


@dataclass
class TranscribeResult:
    text: str
    chunk_texts: list[str]
    num_chunks: int
    duration_s: float
    timings: StageTimings = field(default_factory=StageTimings)
    segments: list[dict] = field(default_factory=list)
    retry_telemetry: dict = field(default_factory=dict)
    vad_stats: dict = field(default_factory=dict)  # populated only for chunking="vad"


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _lcs_merge(chunk_texts: list[str], overlap_word_guess: int = 15) -> str:
    """Stitch overlapping chunks' text via word-level LCS (orchestrator's
    "simpler timestamp-free heuristic" option, not transformers' exact
    timestamp-based algorithm -- we have no per-token timestamps to work
    with since DESIGN.md decodes without-timestamps throughout).

    For each consecutive pair, find the longest matching run of words
    between chunk i's tail and chunk i+1's head (difflib.SequenceMatcher,
    same algorithm `diff`/`git diff` use). Cut at the MIDDLE of that
    matching run -- "keeping the mid-chunk side on disagreements" means:
    where the two chunks' independent transcriptions of the same overlap
    audio actually agree, split the difference; where they don't overlap
    into an agreeing run at all (chunk mismatch/hallucination), fall back
    to skipping a fixed guessed number of words from the next chunk's
    head, on the assumption the tail of the previous chunk already
    covered that content (crude, but bounded -- avoids unbounded
    duplication when LCS finds nothing).
    """
    if not chunk_texts:
        return ""
    words_per_chunk = [c.split() for c in chunk_texts]
    merged: list[str] = list(words_per_chunk[0])
    for i in range(1, len(words_per_chunk)):
        tail = merged[-40:]
        tail_offset = len(merged) - len(tail)
        head = words_per_chunk[i][:40]

        sm = difflib.SequenceMatcher(None, tail, head, autojunk=False)
        blocks = [blk for blk in sm.get_matching_blocks() if blk.size >= 2]
        if blocks:
            best = max(blocks, key=lambda blk: blk.size)
            cut_tail_local = best.a + best.size // 2
            cut_head = best.b + best.size // 2
            merged = merged[: tail_offset + cut_tail_local] + words_per_chunk[i][cut_head:]
        else:
            skip = min(overlap_word_guess, len(words_per_chunk[i]))
            merged = merged + words_per_chunk[i][skip:]
    return " ".join(merged)


def _decode_and_classify(
    audio_chunks: list[np.ndarray],
    chunk_offsets: list[float],
    weights: WhisperWeights,
    tok: whisper_tokenizer.Tokenizer,
    device: torch.device,
    n_prompt: int,
    timestamps: bool,
    enable_ngram_ban: bool,
    repetition_brake: bool,
    temperature: float,
    chunk_durations: list[float],
    timings: StageTimings,
) -> list["retry_ladder.ChunkResult"]:
    """Runs mel -> encode -> decode -> classify for one batch's worth of
    RAW audio chunks (already the right length for pad_or_trim), at a
    single temperature. Used for both the main t=0 pass and every retry
    rung -- the only difference between them is `temperature` and
    `repetition_brake`, both passed straight through to decode.py.

    chunk_durations is per-chunk (not a single shared value) since VAD
    windows vary in real (pre-pad) length; extract_segments uses each
    chunk's own true duration for the "unpaired final timestamp" case
    (open segment ends at the chunk's real end, not the 30s pad boundary).
    """
    t0 = time.perf_counter()
    audio_t = torch.from_numpy(np.stack(audio_chunks, axis=0)).to(device=device, dtype=torch.float32)
    mel = whisper_audio.log_mel_spectrogram_batch(audio_t, device)
    _sync(device)
    timings.mel_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    audio_features = whisper_model.encode(mel, weights)
    _sync(device)
    timings.encode_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    decode_result = whisper_decode.greedy_decode(
        audio_features, weights, tok,
        enable_ngram_ban=enable_ngram_ban, timestamps=timestamps,
        repetition_brake=repetition_brake, temperature=temperature,
    )
    _sync(device)
    timings.decode_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    tokens_cpu = decode_result.tokens.cpu().tolist()
    sum_logprob_cpu = decode_result.sum_logprob.cpu().tolist()
    no_speech_prob_cpu = decode_result.no_speech_prob.cpu().tolist()

    out: list[retry_ladder.ChunkResult] = []
    for i, row in enumerate(tokens_cpu):
        text = tok.decode(row).strip()
        gen_len = retry_ladder.generated_length_excl_eot(row, n_prompt, whisper_tokenizer.EOT)
        avg_lp = retry_ladder.avg_logprob(sum_logprob_cpu[i], gen_len)
        comp_ratio = retry_ladder.compression_ratio(text)
        nsp = no_speech_prob_cpu[i]
        status = retry_ladder.classify(avg_lp, comp_ratio, nsp)
        segs = (
            whisper_segments.extract_segments(row[n_prompt:], tok, chunk_offsets[i], chunk_durations[i])
            if timestamps
            else []
        )
        if status == "silent":
            # orchestrator's exact spec: "mark silent, emit empty, NO retry"
            text = ""
            segs = []
        out.append(
            retry_ladder.ChunkResult(
                text=text, segments=segs, tokens_row=row,
                avg_logprob=avg_lp, compression_ratio=comp_ratio, no_speech_prob=nsp,
                status=status, temperature=temperature,
            )
        )
    timings.detok_s += time.perf_counter() - t0
    return out


@dataclass
class PreparedChunks:
    """Output of the CPU-only phase of transcribing a file (load + chunk,
    including VAD if that's the mode) -- everything transcribe_prepared
    needs to go straight to GPU work. Splitting this out from transcribe_file
    (which still does both phases back to back, unchanged behavior) is what
    lets a caller overlap file N+1's CPU prep with file N's GPU work: see
    bench_e2e.py's per-file loop, which prefetches the next file's
    PreparedChunks on a background thread as soon as the current file's GPU
    phase starts (orchestrator directive 2026-08-10, "VAD cost recovery").
    """

    chunks: list[np.ndarray]
    chunk_offsets: list[float]
    chunk_durations: list[float]
    duration_s: float
    chunking: ChunkingMode
    vad_stats: dict = field(default_factory=dict)
    load_s: float = 0.0
    vad_s: float = 0.0


def prepare_chunks(path: str, chunking: ChunkingMode = "vad") -> PreparedChunks:
    """CPU-only: load audio + chunk it (VAD inference happens here for
    chunking="vad"). No GPU access -- safe to call from a background thread
    while another file's GPU work is in flight."""
    t0 = time.perf_counter()
    raw = whisper_audio.load_audio_mono16k(path)
    duration_s = len(raw) / whisper_audio.SAMPLE_RATE
    load_s = time.perf_counter() - t0

    # VAD (CPU-only) is timed separately from load_s -- it's real wall-clock
    # cost on the total pipeline even though it never touches the GPU, and
    # lumping it into "load_s" would hide exactly the number the orchestrator
    # asked to see reported (RTF delta, GPU time untouched).
    vad_stats: dict = {}
    t0 = time.perf_counter()
    if chunking == "hard":
        chunks, chunk_offsets, chunk_durations = whisper_audio.chunk_audio(raw)
    elif chunking == "stride":
        chunks, chunk_offsets, chunk_durations = whisper_audio.chunk_audio_stride(raw)
    elif chunking == "vad":
        (chunks, chunk_offsets, chunk_durations), vad_stats = whisper_audio.chunk_audio_vad(raw)
    else:
        raise ValueError(f"unknown chunking mode {chunking!r}")
    chunks = [whisper_audio.pad_or_trim(c) for c in chunks]
    vad_s = time.perf_counter() - t0

    return PreparedChunks(
        chunks=chunks, chunk_offsets=chunk_offsets, chunk_durations=chunk_durations,
        duration_s=duration_s, chunking=chunking, vad_stats=vad_stats,
        load_s=load_s, vad_s=vad_s,
    )


@torch.no_grad()
def transcribe_prepared(
    prepared: PreparedChunks,
    weights: WhisperWeights,
    tok: whisper_tokenizer.Tokenizer,
    device: torch.device,
    batch_size: int = 16,
    enable_ngram_ban: bool = False,  # see decode.py's module docstring for why
    timestamps: bool = False,
    retry_ladder_enabled: bool = True,
) -> TranscribeResult:
    """GPU phase: mel -> encode -> decode -> classify -> retry ladder, given
    already-prepared chunks (see prepare_chunks). transcribe_file below is
    just prepare_chunks + this, back to back, for callers that don't need
    the two phases split."""
    timings = StageTimings(load_s=prepared.load_s, vad_s=prepared.vad_s)
    chunks, chunk_offsets, chunk_durations = (
        prepared.chunks, prepared.chunk_offsets, prepared.chunk_durations
    )
    duration_s = prepared.duration_s
    chunking = prepared.chunking
    vad_stats = prepared.vad_stats

    n_prompt = len(whisper_tokenizer.PROMPT_TIMESTAMPS if timestamps else whisper_tokenizer.PROMPT)

    # ---- main pass: batched greedy (t=0), same as before M6 ----
    results: list[retry_ladder.ChunkResult] = [None] * len(chunks)  # type: ignore[list-item]
    for start in range(0, len(chunks), batch_size):
        idxs = list(range(start, min(start + batch_size, len(chunks))))
        batch_results = _decode_and_classify(
            [chunks[i] for i in idxs], [chunk_offsets[i] for i in idxs],
            weights, tok, device, n_prompt, timestamps, enable_ngram_ban,
            repetition_brake=True, temperature=0.0,
            chunk_durations=[chunk_durations[i] for i in idxs], timings=timings,
        )
        for i, r in zip(idxs, batch_results):
            results[i] = r

    # ---- retry ladder: pool failing chunks, escalate temperature ----
    telemetry = retry_ladder.RetryTelemetry(total_chunks=len(chunks))
    if retry_ladder_enabled:
        pool = [i for i, r in enumerate(results) if r.status == "retry"]
        telemetry.retried = len(pool)
        for t in retry_ladder.TEMPERATURE_SCHEDULE:
            if not pool:
                break
            still_failing = []
            for start in range(0, len(pool), batch_size):
                idxs = pool[start : start + batch_size]
                batch_results = _decode_and_classify(
                    [chunks[i] for i in idxs], [chunk_offsets[i] for i in idxs],
                    weights, tok, device, n_prompt, timestamps, enable_ngram_ban,
                    repetition_brake=False, temperature=t,
                    chunk_durations=[chunk_durations[i] for i in idxs], timings=timings,
                )
                for i, r in zip(idxs, batch_results):
                    results[i] = r  # keep-last: always update, regardless of status
                    if r.status == "silent":
                        telemetry.silent += 1
                    elif r.status == "retry":
                        still_failing.append(i)
                    else:
                        telemetry.resolved_by_temp[t] = telemetry.resolved_by_temp.get(t, 0) + 1
            pool = still_failing
        telemetry.unresolved_kept_last = len(pool)  # kept whatever the t=1.0 (or last-tried) attempt produced

    chunk_texts = [r.text for r in results]
    all_segments: list[dict] = []
    if timestamps:
        for r in results:
            all_segments.extend(r.segments)

    if chunking == "stride":
        text = _lcs_merge(chunk_texts)
    else:
        # hard and vad windows are both non-overlapping (vad closes on
        # silence gaps instead of a fixed grid, but never overlaps) -- plain
        # join, no LCS stitch needed.
        text = " ".join(t for t in chunk_texts if t)

    return TranscribeResult(
        text=text,
        chunk_texts=chunk_texts,
        num_chunks=len(chunks),
        duration_s=duration_s,
        segments=all_segments,
        timings=timings,
        retry_telemetry=telemetry.as_dict(),
        vad_stats=vad_stats,
    )


def transcribe_file(
    path: str,
    weights: WhisperWeights,
    tok: whisper_tokenizer.Tokenizer,
    device: torch.device,
    batch_size: int = 16,
    chunking: ChunkingMode = "vad",  # default since 2026-08-10 (Jim's decision) -- see RESULTS.md's
    # "VAD chunking mode" section: divergence roughly halved vs hard-cut at every measured config,
    # stall-window rate 0.60% -> 0.11%; hard/stride remain available for comparison/fallback.
    enable_ngram_ban: bool = False,
    timestamps: bool = False,
    retry_ladder_enabled: bool = True,
) -> TranscribeResult:
    """prepare_chunks + transcribe_prepared, back to back -- unchanged
    single-call behavior for callers that don't need the CPU/GPU phases
    split (bench_e2e.py's prefetch loop calls the two phases directly
    instead, to overlap file N+1's CPU prep with file N's GPU work)."""
    prepared = prepare_chunks(path, chunking=chunking)
    return transcribe_prepared(
        prepared, weights, tok, device, batch_size=batch_size,
        enable_ngram_ban=enable_ngram_ban, timestamps=timestamps,
        retry_ladder_enabled=retry_ladder_enabled,
    )
