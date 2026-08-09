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


def chunk_audio(audio: np.ndarray, chunk_length_s: float = CHUNK_LENGTH) -> list[np.ndarray]:
    """Split into non-overlapping chunks of `chunk_length_s` seconds
    (DESIGN.md S5: "same as baseline: 375 chunks for 11235s", no stride).
    The last chunk is short; callers pad it (pad_or_trim) before batching.
    """
    n = int(chunk_length_s * SAMPLE_RATE)
    chunks = [audio[i : i + n] for i in range(0, len(audio), n)]
    if len(chunks) == 0:
        chunks = [np.zeros(n, dtype=np.float32)]
    return chunks


def chunk_audio_stride(
    audio: np.ndarray, window_s: float = CHUNK_LENGTH, stride_s: float = 5.0
) -> list[np.ndarray]:
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
    """
    window = int(window_s * SAMPLE_RATE)
    stride = int(stride_s * SAMPLE_RATE)
    step = window - 2 * stride
    assert step > 0, "stride_s too large relative to window_s"
    n = len(audio)
    chunks = []
    start = 0
    while True:
        chunks.append(audio[start : start + window])
        if start + window >= n:
            break
        start += step
    if not chunks:
        chunks = [np.zeros(window, dtype=np.float32)]
    return chunks


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
