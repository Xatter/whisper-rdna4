"""Timestamp-token -> {Start, End, Text} segment extraction (M5, orchestrator
directive 2026-08-09). Turns a decoded token sequence (timestamps mode: an
alternating stream of timestamp tokens and text tokens) into the same shape
Castria's ad-detection input expects (`data/episodes/{uuid}.json`'s
`transcript` field: a list of {"Start": float, "End": float, "Text": str}).

Standard Whisper segment grammar (matches openai-whisper's own
`decoding.py:ApplyTimestampRules`, which this decodes the *output* of):
timestamp tokens alternate role by encounter order within a chunk --
1st/3rd/5th/... is an "open" (segment start), 2nd/4th/6th/... is a "close"
(segment end). Text tokens between an open and its close are that
segment's content. `ApplyTimestampRules` guarantees opens/closes alternate
correctly and are non-decreasing, so no lookahead or validation is needed
here -- just walk the sequence once.
"""

from __future__ import annotations

from . import tokenizer as whisper_tokenizer

SECONDS_PER_TIMESTAMP_TOKEN = 0.02  # 30s chunk / 1500 encoder frames


def extract_segments(
    generated_ids: list[int],
    tok: "whisper_tokenizer.Tokenizer",
    chunk_offset: float,
    chunk_duration: float,
) -> list[dict]:
    """generated_ids: tokens AFTER the prompt (decode.py's token_ids[n_prompt:],
    as returned by greedy_decode with timestamps=True) for ONE chunk/sequence
    -- stop at (not including) EOT and any trailing eot-padding.

    chunk_offset: this chunk's start time in the full-file timeline (hard-cut:
    chunk_index * chunk_duration). chunk_duration: this chunk's length in
    seconds (30.0 for a full hard-cut chunk; the last chunk of a file may be
    shorter after padding is accounted for by the caller).

    Returns a list of {"Start": float, "End": float, "Text": str} dicts,
    Start/End in absolute (whole-file) seconds, in chronological order.
    Handles two edge cases explicitly (M5 spec):
    - an unpaired final (opening) timestamp before EOT: decode stopped
      mid-segment (e.g. hit max_len, or finished by the no-speech gate /
      repetition brake after opening a segment) -- treated as an open
      segment that ends at the chunk boundary.
    - a degenerate chunk with text but no timestamp tokens at all (should
      not happen under ApplyTimestampRules' invariant that the first
      generated token is always a timestamp, but handled defensively
      rather than assumed impossible): the whole chunk becomes one segment.
    """
    segments: list[dict] = []
    cur_start: float | None = None
    cur_text_ids: list[int] = []

    for t in generated_ids:
        if t == whisper_tokenizer.EOT:
            break
        if t >= whisper_tokenizer.TIMESTAMP_BEGIN:
            ts = (t - whisper_tokenizer.TIMESTAMP_BEGIN) * SECONDS_PER_TIMESTAMP_TOKEN
            if cur_start is None:
                cur_start = ts  # opening timestamp
            else:
                # closing timestamp -- emit the segment, reset for the next one
                segments.append(
                    {
                        "Start": chunk_offset + cur_start,
                        "End": chunk_offset + ts,
                        "Text": tok.decode(cur_text_ids).strip(),
                    }
                )
                cur_start = None
                cur_text_ids = []
        else:
            cur_text_ids.append(t)

    if cur_start is not None:
        # unpaired final (opening) timestamp -- open segment, ends at chunk end
        segments.append(
            {
                "Start": chunk_offset + cur_start,
                "End": chunk_offset + chunk_duration,
                "Text": tok.decode(cur_text_ids).strip(),
            }
        )
    elif cur_text_ids:
        # no-timestamp degenerate chunk -- whole chunk is one segment
        segments.append(
            {
                "Start": chunk_offset,
                "End": chunk_offset + chunk_duration,
                "Text": tok.decode(cur_text_ids).strip(),
            }
        )

    return segments
