"""Temperature-retry ladder building blocks (M6, orchestrator directive
2026-08-09). Mirrors openai-whisper's `transcribe.py:decode_with_fallback`
and `whisper/utils.py:compression_ratio` exactly -- read directly from the
venv on gpu-host as the reference, not from memory. The orchestration
(pooling failed chunks across batches, escalating temperature, keep-last)
lives in `pipeline.py`; this module only holds the per-chunk math and
decision rule, kept separate so that logic is unit-testable/readable on
its own.

Denominator convention (openai-whisper `decoding.py`, `DecodingTask.run`):
    tokens = t[sample_begin : (t == eot).nonzero()[0,0]]   # EXCLUDES eot
    avg_logprob = sum_logprob / (len(tokens) + 1)          # +1 compensates
So `generated_length_excl_eot` below deliberately does NOT count the EOT
token itself, matching that slice exactly -- the "+1" in avg_logprob is
not a fudge factor, it's accounting for EOT's own logprob contribution
(which IS included in sum_logprob, since a row's EOT-producing step is
never excluded -- see decode.py's M6 section) against a token count that
excludes EOT from the list.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Literal

LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
NO_SPEECH_THRESHOLD = 0.6  # matches decode.py's NO_SPEECH_THRESHOLD default
TEMPERATURE_SCHEDULE = (0.2, 0.4, 0.6, 0.8, 1.0)

ChunkStatus = Literal["ok", "retry", "silent"]


def compression_ratio(text: str) -> float:
    """openai-whisper `whisper/utils.py:compression_ratio` exactly:
    len(utf-8 bytes) / len(zlib-compressed bytes). Empty text -> 0.0 (zlib
    of an empty string still produces a nonzero-length header, so this
    needs an explicit guard to avoid a division that would otherwise
    silently work but isn't what the reference does for len(text)==0 --
    openai's transcribe.py never calls this on empty text because
    should_skip/continue happens first; we can reach it, so guard it).
    """
    text_bytes = text.encode("utf-8")
    if not text_bytes:
        return 0.0
    return len(text_bytes) / len(zlib.compress(text_bytes))


def generated_length_excl_eot(tokens_row: list[int], n_prompt: int, eot: int) -> int:
    """Number of generated tokens BEFORE the first EOT (EOT itself
    excluded), matching openai's `t[sample_begin : eot_index]` slice.
    If no EOT appears in range (hit the step cap without finishing),
    the whole generated span counts -- there's no EOT to exclude."""
    for i in range(n_prompt, len(tokens_row)):
        if tokens_row[i] == eot:
            return i - n_prompt
    return len(tokens_row) - n_prompt


def avg_logprob(sum_logprob: float, gen_len_excl_eot: int) -> float:
    return sum_logprob / (gen_len_excl_eot + 1)


def classify(
    avg_lp: float,
    comp_ratio: float,
    no_speech_prob: float,
    logprob_threshold: float = LOGPROB_THRESHOLD,
    compression_ratio_threshold: float = COMPRESSION_RATIO_THRESHOLD,
    no_speech_threshold: float = NO_SPEECH_THRESHOLD,
) -> ChunkStatus:
    """openai's `decode_with_fallback` decision logic, collapsed to a
    single per-attempt classification (orchestrator's exact wording):
    - no_speech_prob > 0.6 AND avg_logprob < -1.0 -> "silent" (mark empty,
      stop retrying -- mirrors the no_speech override in
      decode_with_fallback's loop, which prevents pointless retries on
      probable-silence segments, combined with the outer should_skip
      check in transcribe()'s main loop that actually blanks the segment).
    - else compression_ratio > 2.4 OR avg_logprob < -1.0 -> "retry".
    - else -> "ok".

    Architectural note (measured, not assumed -- see RESULTS.md): this
    pipeline's no-speech GATE (decode.py, M2) already force-finishes a
    sequence with zero generated tokens the moment position-0's
    p(no_speech) exceeds the SAME 0.6 threshold used here. A gated row's
    sum_logprob stays exactly 0.0 (no step contributes, `forced` is True
    from the start), so its avg_logprob is 0.0 -- which fails this
    function's "avg_logprob < -1.0" half of the "silent" condition. Net
    effect: under this architecture, "silent" is expected to fire rarely
    or never in the retry ladder specifically, because the upstream gate
    already caught the same signal earlier and more cheaply (before
    spending any decode steps at all, vs. after a full attempt here).
    Implemented faithfully anyway (matching the orchestrator's literal
    spec, and as a defensive backstop if the two thresholds are ever
    tuned independently) -- and telemetry reports the actual fire rate
    rather than assuming it's zero.
    """
    if no_speech_prob > no_speech_threshold and avg_lp < logprob_threshold:
        return "silent"
    if comp_ratio > compression_ratio_threshold or avg_lp < logprob_threshold:
        return "retry"
    return "ok"


@dataclass
class ChunkResult:
    text: str
    segments: list[dict] = field(default_factory=list)
    tokens_row: list[int] = field(default_factory=list)
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    status: ChunkStatus = "ok"
    temperature: float = 0.0


@dataclass
class RetryTelemetry:
    total_chunks: int = 0
    retried: int = 0
    resolved_by_temp: dict[float, int] = field(default_factory=dict)
    silent: int = 0
    unresolved_kept_last: int = 0

    def as_dict(self) -> dict:
        return {
            "total_chunks": self.total_chunks,
            "retried": self.retried,
            "pct_retried": (self.retried / self.total_chunks * 100) if self.total_chunks else 0.0,
            "resolved_by_temp": dict(self.resolved_by_temp),
            "silent": self.silent,
            "pct_silent": (self.silent / self.total_chunks * 100) if self.total_chunks else 0.0,
            "unresolved_kept_last": self.unresolved_kept_last,
        }
