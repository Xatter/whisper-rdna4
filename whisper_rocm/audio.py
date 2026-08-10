"""Audio loading, chunking, and batched GPU mel spectrogram (DESIGN.md S3, S5).

Mel: 128 bins, n_fft 400, hop 160, 16 kHz; 30s chunk -> 3000 frames.
Mel filterbank constants are ported from the openai-whisper package's
packaged `assets/mel_filters.npz` (librosa-derived, shipped as a binary
asset -- reading it via whisper's own loader is the faithful "port" DESIGN
asks for; hand-transcribing a 128x201 float matrix would be error-prone and
strictly worse). No other openai-whisper code runs on the hot path: the
STFT + framing below is our own batched implementation.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000
N_MELS = 128


@lru_cache(maxsize=None)
def _mel_filters(device: str, n_mels: int = N_MELS) -> torch.Tensor:
    """Load the (n_mels, n_fft//2+1) mel filterbank matrix.

    Ported from openai-whisper's whisper/audio.py:mel_filters -- same
    packaged npz asset, same lookup key ("mel_128"). Falls back to scipy
    if the whisper package isn't importable in this environment.
    """
    try:
        import whisper.audio as _oa_audio

        return _oa_audio.mel_filters(device, n_mels).to(torch.float32)
    except ImportError:
        raise RuntimeError(
            "openai-whisper package not importable; needed at least once to "
            "source the mel filterbank asset. `uv pip install openai-whisper`."
        )


def load_audio_mono16k(path: str) -> np.ndarray:
    """Load an audio file and return mono float32 samples at 16 kHz.

    Uses soundfile for the common formats; for compressed formats (mp3)
    soundfile relies on libsndfile with mp3 support (present on gpu-host).
    Resamples with scipy.signal.resample_poly when the source rate differs.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1)  # downmix to mono
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, SAMPLE_RATE)
        up, down = SAMPLE_RATE // g, sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    return audio


ChunkSpec = tuple[list[np.ndarray], list[float], list[float]]  # (chunks, offsets_s, durations_s)


def chunk_audio(audio: np.ndarray, chunk_length_s: float = CHUNK_LENGTH) -> ChunkSpec:
    """Split into non-overlapping chunks of `chunk_length_s` seconds
    (DESIGN.md S5: "same as baseline: 375 chunks for 11235s", no stride).
    The last chunk is short; callers pad it (pad_or_trim) before batching.

    Returns (chunks, offsets_s, durations_s) -- offsets/durations are each
    chunk's TRUE (pre-pad) position and length in the file's timeline. VAD
    mode (chunk_audio_vad) needs genuinely per-chunk offsets/durations since
    its windows aren't evenly spaced or sized; hard/stride mode's are a
    trivial arithmetic sequence, but returning them in the same explicit
    per-chunk shape (rather than a step formula the caller reconstructs)
    means pipeline.py has exactly one code path for all three modes -- see
    pipeline.py's transcribe_file.
    """
    n = int(chunk_length_s * SAMPLE_RATE)
    chunks = [audio[i : i + n] for i in range(0, len(audio), n)]
    if len(chunks) == 0:
        chunks = [np.zeros(n, dtype=np.float32)]
    offsets = [i * chunk_length_s for i in range(len(chunks))]
    durations = [len(c) / SAMPLE_RATE for c in chunks]
    return chunks, offsets, durations


def chunk_audio_stride(
    audio: np.ndarray, window_s: float = CHUNK_LENGTH, stride_s: float = 5.0
) -> ChunkSpec:
    """HF-pipeline-style overlapping windows (M2 quality guard, orchestrator
    directive 2026-08-09): 30s window, 5s stride on each side (transformers'
    AutomaticSpeechRecognitionPipeline default is stride_length_s =
    chunk_length_s / 6 = 5s for chunk_length_s=30 -- verified directly in
    that package's source; DESIGN.md's assumption that the baseline uses
    simple non-overlapping chunks does not hold in practice, and this mode
    exists to close the resulting WER gap). Step between window starts is
    window_s - 2*stride_s = 20s of new content per window. The last window
    is short; callers pad it (pad_or_trim) before batching. Overlap is
    resolved at merge time by pipeline.py's LCS-based text stitch, not here.

    Returns (chunks, offsets_s, durations_s), see chunk_audio's docstring.
    """
    window = int(window_s * SAMPLE_RATE)
    stride = int(stride_s * SAMPLE_RATE)
    step = window - 2 * stride
    assert step > 0, "stride_s too large relative to window_s"
    n = len(audio)
    chunks = []
    offsets = []
    start = 0
    while True:
        chunks.append(audio[start : start + window])
        offsets.append(start / SAMPLE_RATE)
        if start + window >= n:
            break
        start += step
    if not chunks:
        chunks = [np.zeros(window, dtype=np.float32)]
        offsets = [0.0]
    durations = [len(c) / SAMPLE_RATE for c in chunks]
    return chunks, offsets, durations


# ---------------------------------------------------------------------------
# VAD chunking (Jim-approved, orchestrator directive 2026-08-10). Silero VAD
# on CPU picks real speech boundaries; a faster-whisper/WhisperX-style greedy
# packer merges consecutive speech segments into <=30s windows so encoder
# calls spend less time on silence and chunk boundaries fall in silence
# instead of mid-word. GPU timing is untouched -- VAD runs entirely on CPU
# before any GPU work starts.
# ---------------------------------------------------------------------------

VAD_MAX_WINDOW_S = CHUNK_LENGTH  # 30s, same encoder/mel budget as hard/stride


def _stub_torchaudio_if_broken() -> None:
    """silero_vad.utils_vad does an unconditional `import torchaudio` even
    though we never call its read_audio/save_audio helpers (we always pass
    an already-loaded tensor straight in). On gpu-host's ROCm build
    (torch 2.13.0+rocm7.2), the torchaudio wheel pip pulls in as a dependency
    is a stock CUDA/CPU build with an incompatible compiled extension --
    `import torchaudio` raises OSError ("Could not load this library:
    .../_torchaudio.abi3.so") from its C-extension loader, not the more
    easily-caught ImportError. Since nothing we use needs torchaudio's
    actual functionality, pre-registering an empty stub module (only if a
    real import genuinely fails) lets silero_vad's own `import torchaudio`
    line succeed as a no-op instead of taking down the whole VAD path over
    a dependency we don't exercise.
    """
    import sys

    if "torchaudio" in sys.modules:
        return
    try:
        import torchaudio  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        import types

        stub = types.ModuleType("torchaudio")
        stub.__silero_vad_stub_reason__ = repr(exc)
        sys.modules["torchaudio"] = stub


VAD_ONNX_INTRA_OP_THREADS = 4  # A/B'd 1/4/8/12 (2026-08-10): 1 thread already
# beats the torch JIT path 2.4x (silero_vad's own model.py hardcodes
# torch.set_num_threads(1) at import time, so the JIT path never had a
# chance); 4 threads gets the rest of the win (2.9x total). 8/12 add <1%
# more -- this model is small enough that thread count past 4 stops
# mattering, so 4 is the practical choice rather than reserving all cores.


def _load_vad_model_onnx(intra_op_threads: int = VAD_ONNX_INTRA_OP_THREADS):
    """Loads Silero VAD via onnxruntime instead of the torch JIT path --
    2.4-2.9x faster in direct A/B (see VAD_ONNX_INTRA_OP_THREADS' comment),
    the default backend since 2026-08-10. silero_vad's own
    load_silero_vad(onnx=True) (silero_vad/utils_vad.py's OnnxWrapper)
    hardcodes both inter- and intra-op thread counts to 1 regardless of
    what's asked for, so this reconstructs the onnxruntime session
    afterward with more intra-op threads when requested -- the model path
    is re-derived the same way load_silero_vad(onnx=True) does internally
    (there's no public accessor for it on the returned wrapper).
    """
    _stub_torchaudio_if_broken()
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=True)
    if intra_op_threads and intra_op_threads > 1:
        import onnxruntime

        try:
            import importlib_resources as impresources
        except ImportError:
            from importlib import resources as impresources
        model_path = str(impresources.files("silero_vad.data").joinpath("silero_vad.onnx"))
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        model.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
    return model, get_speech_timestamps


@lru_cache(maxsize=1)
def _load_vad_model():
    """Loads Silero VAD once per process, CPU only. Tries onnxruntime first
    (default backend since 2026-08-10, see _load_vad_model_onnx); falls back
    to the `silero-vad` pip package's torch JIT path if onnxruntime isn't
    installed, then to torch.hub if the package itself isn't installed
    (torch.hub.load('snakers4/silero-vad', ...) is the older but still-
    supported path and some environments may have it cached already).
    """
    _stub_torchaudio_if_broken()
    try:
        import onnxruntime  # noqa: F401

        return _load_vad_model_onnx()
    except ImportError:
        pass
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()
        return model, get_speech_timestamps
    except ImportError:
        pass

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def _vad_speech_segments(audio: np.ndarray) -> list[tuple[float, float]]:
    """Runs Silero VAD on CPU, returns [(start_s, end_s), ...] speech
    segments, sorted, non-overlapping (Silero's own contract)."""
    model, get_speech_timestamps = _load_vad_model()
    wav = torch.from_numpy(audio.astype(np.float32))
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=SAMPLE_RATE, return_seconds=True
    )
    return [(float(t["start"]), float(t["end"])) for t in timestamps]


def _pack_vad_windows(
    speech_segments: list[tuple[float, float]], max_window_s: float = VAD_MAX_WINDOW_S
) -> tuple[list[tuple[float, float]], dict]:
    """faster-whisper/WhisperX-style greedy packer: extend the current
    window while the next speech segment (plus the silence gap before it)
    still fits within max_window_s; when it doesn't fit, close the window
    at the end of the last segment that DID fit -- the boundary lands in
    the silence gap before the next segment, not mid-speech. A single VAD
    segment longer than max_window_s on its own (an uninterrupted speech
    run with no detected silence) can't be closed on silence at all; that
    case is hard-split at max_window_s boundaries, counted separately so
    it's visible how often the "unavoidable hard cut" actually happens.

    Returns (windows, stats) where windows is [(start_s, end_s), ...] and
    stats has n_speech_segments, n_windows, n_long_segments_hard_split,
    n_hard_split_subwindows.
    """
    windows: list[tuple[float, float]] = []
    n_long_segments_hard_split = 0
    n_hard_split_subwindows = 0

    i = 0
    n = len(speech_segments)
    while i < n:
        win_start, win_end = speech_segments[i]
        j = i + 1
        while j < n:
            next_start, next_end = speech_segments[j]
            if next_end - win_start <= max_window_s:
                win_end = next_end
                j += 1
            else:
                break

        if win_end - win_start > max_window_s:
            # Only possible when j == i+1: segment i alone already exceeds
            # the window budget (a long uninterrupted speech run). Hard-split
            # it -- there's no silence anywhere in this span to close on.
            n_long_segments_hard_split += 1
            seg_dur = win_end - win_start
            n_splits = math.ceil(seg_dur / max_window_s)
            n_hard_split_subwindows += n_splits
            for k in range(n_splits):
                s = win_start + k * max_window_s
                e = min(win_start + (k + 1) * max_window_s, win_end)
                windows.append((s, e))
        else:
            windows.append((win_start, win_end))

        i = j

    stats = {
        "n_speech_segments": n,
        "n_windows": len(windows),
        "n_long_segments_hard_split": n_long_segments_hard_split,
        "n_hard_split_subwindows": n_hard_split_subwindows,
    }
    return windows, stats


def chunk_audio_vad(audio: np.ndarray, max_window_s: float = VAD_MAX_WINDOW_S) -> tuple[ChunkSpec, dict]:
    """VAD-packed chunking: Silero VAD (CPU) finds speech segments, then
    _pack_vad_windows greedily merges them into <=30s windows, closing on
    silence gaps rather than at a fixed arithmetic offset. Silent spans
    between windows are never sliced out of `audio` at all, so they never
    reach the encoder -- "skip long pure-silence spans" falls out of this
    for free rather than needing separate handling.

    A file with NO detected speech at all (e.g. dead air) falls back to one
    zero-length-safe window covering the whole file, matching hard/stride's
    "never return an empty chunk list" contract.

    Returns ((chunks, offsets_s, durations_s), vad_stats).
    """
    speech_segments = _vad_speech_segments(audio)
    if not speech_segments:
        chunks, offsets, durations = chunk_audio(audio, chunk_length_s=max_window_s)
        stats = {
            "n_speech_segments": 0, "n_windows": len(chunks),
            "n_long_segments_hard_split": 0, "n_hard_split_subwindows": 0,
            "fell_back_to_hard_cut_no_speech": True,
        }
        return (chunks, offsets, durations), stats

    windows, stats = _pack_vad_windows(speech_segments, max_window_s)
    chunks = [audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)] for s, e in windows]
    offsets = [s for s, _e in windows]
    durations = [e - s for s, e in windows]
    return (chunks, offsets, durations), stats


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio


def log_mel_spectrogram_batch(
    audio_batch: torch.Tensor, device: torch.device, n_mels: int = N_MELS
) -> torch.Tensor:
    """Batched log-mel spectrogram on GPU.

    audio_batch: (B, N_SAMPLES) float32, already on `device` (or moved here).
    returns: (B, n_mels, N_FRAMES) float16.

    torch.stft natively batches over a leading dimension, so this is the
    same computation as openai-whisper's log_mel_spectrogram, just with a
    batch dim carried through and the filterbank matmul done as one
    batched GEMM instead of B separate calls.
    """
    audio_batch = audio_batch.to(device=device, dtype=torch.float32)
    window = torch.hann_window(N_FFT, device=device)
    stft = torch.stft(
        audio_batch, N_FFT, HOP_LENGTH, window=window, return_complex=True
    )  # (B, n_fft//2+1, n_frames)
    magnitudes = stft[..., :-1].abs() ** 2  # drop last frame, matches reference

    filters = _mel_filters(str(device), n_mels)  # (n_mels, n_fft//2+1)
    mel_spec = torch.einsum("mf,bft->bmt", filters, magnitudes)  # (B, n_mels, n_frames)

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # max over freq+time per-example, matching reference's per-call .max()
    per_example_max = log_spec.amax(dim=(1, 2), keepdim=True)
    log_spec = torch.maximum(log_spec, per_example_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.to(torch.float16)
