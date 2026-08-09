"""Batched greedy decode loop, GPU-resident (DESIGN.md S7).

Per DESIGN.md S7: preallocated KV cache tensors, no per-step .item()/.cpu()
calls except the lagged all-finished check every 8 steps; suppress-token
mask; prompt [50258, 50259, 50360, 50364]; stop at eot=50257; cap 448.

The 4 prompt tokens are run through the exact same per-step loop as every
generated token (model.decoder_step, position 0..3) rather than a special
batched "prefill" call -- this keeps every decode step at the single-token
granularity Agent C's K4 kernel contract (kernels/decode_api.py) requires.

M2 additions (orchestrator quality-guard directive, 2026-08-09):
  - no-speech gate: openai-whisper's own heuristic (decoding.py
    DecodingTask._main_loop, i==0 branch) -- softmax the position-0 logits
    (the ones produced from just the SOT token, before EN/TRANSCRIBE/
    NOTIMESTAMPS are fed) and read off p(no_speech token). > 0.6 forces
    that sequence to finish immediately with zero generated content, which
    detokenizes to "" (tokenizer.decode drops every id >= eot).
  - repetition brake: piggybacked on the existing every-8-steps host sync
    (DECODE_INTEGRATION.md SS6's queue-then-check pattern). Pulls the last
    24 generated tokens per sequence to host and force-finishes any
    sequence whose tail is a period-1..8 cycle repeated >=3 times --
    a crude but effective brake on greedy-decode hallucination loops (see
    bench-out/ investigation: chunk 39/123 of 44e8f624... repeated
    "to New York" ~90x under plain greedy decoding).

M4 addition (orchestrator directive, 2026-08-09): Agent C's GPU-side
no_repeat_ngram_size=3 ban (K5, kernels/DECODE_INTEGRATION.md SS3b) mirrors
HF's NoRepeatNGramLogitsProcessor exactly. Unlike the repetition brake
above (host-side, every 8 steps, catches period-1..8 cycles), the ngram
ban runs on GPU every single step and specifically prevents a 3-gram from
ever repeating. Agent A owns `token_history`: seeded with the 4 prompt
tokens at history_len=4, one token appended per step in the same lock-step
convention as `step`/K4.

Default is now OFF (`enable_ngram_ban=False`), reversing the M4-directive
default. Measured on file 44e8f624 (M4 quality re-measure): turning it ON
moved hard-cut divergence from 15.46% to 18.40% -- a regression, not the
fix expected. Root cause (orchestrator decision, 2026-08-09): baseline
decodes WITH timestamp tokens interleaved in its output stream: those
interleaved timestamps break up repeated word trigrams in TOKEN space, so
HF's ngram ban rarely fires on natural speech in baseline's own pipeline.
Our notimestamps stream makes the same repeated phrases contiguous
trigrams, so a config-exact ban fires on legitimate speech, and greedy
decode has no graceful recovery once its natural continuation is banned --
it falls through to EOT and truncates mid-sentence (confirmed per-chunk:
20 of 24 changed chunks got shorter with the ban on, only 4 got longer;
e.g. chunk 4 of that file: 87 words -> 17, cut off mid-sentence). Matching
baseline's flag literally does not match its behavior in a timestamp-free
pipeline. The flag stays implemented and available for a future
timestamps-mode pipeline, where the rationale above would no longer apply
-- see M5 below, which is exactly that pipeline.

M5 addition (orchestrator directive, 2026-08-09): `timestamps=True` mode.
Prompt drops NOTIMESTAMPS (3 tokens: sot, en, transcribe -- `sample_begin`
is 3, not 4), the suppress mask leaves the timestamp range open (see
tokenizer.suppress_mask(timestamps=True)), and every step additionally
runs ops.apply_timestamp_rules-equivalent logic (via
logits_argmax(..., enable_timestamp_rules=True, sample_begin=n_prompt)) --
a faithful port of openai-whisper's decoding.py:ApplyTimestampRules. The
no-speech gate and repetition brake are unchanged: the gate reads
position-0 logits (SOT-only forward), which doesn't depend on prompt
length; the brake operates on the raw generated-token tail regardless of
what's mixed into it. Per-chunk ngram-ban default is decided empirically
per file/mode (see bench-out/ M5 ngram A/B) rather than assumed -- the
M4 rationale above (interleaved timestamps break up trigrams) predicts the
ban should stop hurting once timestamps are on, but that's a hypothesis to
verify, not a given.

M6 addition (orchestrator directive, 2026-08-09): temperature-retry
ladder support. `greedy_decode` now accepts `temperature` (0.0 = argmax,
matching every mode above unchanged; >0 = sample from
Categorical(logits=masked/temperature), openai's GreedyDecoder.update
semantics exactly) and always collects `sum_logprob` per sequence via
Agent C's K5 SS3d kernel arg (or the torch fallback's equivalent -- see
ops.py). The actual retry LOGIC (per-chunk avg_logprob/compression_ratio
checks, pooling failed chunks, escalating temperature, keep-last-on-total-
failure) lives in pipeline.py, not here, per the orchestrator's explicit
scoping instruction -- this function's only M6-facing job is: decode once
at a given temperature and hand back enough per-sequence telemetry
(tokens, sum_logprob, no_speech_prob) for the caller to make that
decision. Returns a `GreedyDecodeResult`, not a bare tensor, as of this
change -- see below.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import model as whisper_model
from . import ops
from . import tokenizer as whisper_tokenizer
from .weights import WhisperWeights


@dataclass
class GreedyDecodeResult:
    tokens: torch.Tensor  # (B, T<=max_len) long, prompt + generated
    sum_logprob: torch.Tensor  # (B,) fp32, openai's GreedyDecoder.update convention
    no_speech_prob: torch.Tensor  # (B,) fp32, softmax(position-0 logits)[NO_SPEECH_TOKEN]

CHECK_INTERVAL = 8
NO_SPEECH_THRESHOLD = 0.6
NO_SPEECH_TOKEN = 50363  # verified via whisper.tokenizer.get_tokenizer(...).no_speech
                          # for this checkpoint's vocab (51866, 100 languages);
                          # NOTE: differs from the 50362 in the M2 task message --
                          # double-checked directly against the tokenizer, see report.
REP_WINDOW = 24
REP_MAX_PERIOD = 8
REP_MIN_REPEATS = 3


def _find_repetition_loops(tail: list[list[int]]) -> list[int]:
    """tail: per-sequence list of up to REP_WINDOW most-recent generated
    token ids (host ints). Returns indices of sequences whose tail is a
    period-p (1<=p<=REP_MAX_PERIOD) cycle repeated >=REP_MIN_REPEATS times
    back-to-back, e.g. [..., "to","New","York","to","New","York","to",
    "New","York"] (p=3, 3 repeats) -- exactly the pattern found in the
    hallucinating chunk during M1 WER analysis.
    """
    bad = []
    for b, seq in enumerate(tail):
        n = len(seq)
        for p in range(1, REP_MAX_PERIOD + 1):
            need = p * REP_MIN_REPEATS
            if n < need:
                continue
            window = seq[-need:]
            blocks = [window[i * p : (i + 1) * p] for i in range(REP_MIN_REPEATS)]
            if all(block == blocks[0] for block in blocks[1:]):
                bad.append(b)
                break
    return bad


@torch.no_grad()
def greedy_decode(
    audio_features: torch.Tensor,
    weights: WhisperWeights,
    tok: whisper_tokenizer.Tokenizer,
    max_len: int = 448,
    check_interval: int = CHECK_INTERVAL,
    no_speech_threshold: float = NO_SPEECH_THRESHOLD,
    repetition_brake: bool = True,
    enable_ngram_ban: bool = False,
    timestamps: bool = False,
    temperature: float = 0.0,
) -> GreedyDecodeResult:
    """audio_features: (B, 1500, D) fp16, encoder output for B chunks.

    Returns a GreedyDecodeResult. `.tokens`: (B, T<=max_len) long tensor
    holding the prompt followed by generated tokens; positions past a
    sequence's own eot (or a no-speech/repetition-forced finish) are left
    as eot (safe to detokenize -- tokenizer.decode() drops every id >=
    eot; timestamp tokens are also >= eot's id and get dropped the same
    way, so plain-text output is unaffected by this flag -- see
    whisper_rocm.segments for the separate {Start,End,Text} extraction
    timestamps mode enables). `.sum_logprob` / `.no_speech_prob`: (B,)
    fp32, for the caller's (pipeline.py's) temperature-retry decision --
    see this module's M6 docstring section.
    """
    device = audio_features.device
    b = audio_features.shape[0]
    d = weights.dims

    prompt_ids = whisper_tokenizer.PROMPT_TIMESTAMPS if timestamps else whisper_tokenizer.PROMPT
    prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long)
    n_prompt = prompt.shape[0]

    tokens = torch.full(
        (b, max_len), whisper_tokenizer.EOT, dtype=torch.long, device=device
    )
    tokens[:, :n_prompt] = prompt.unsqueeze(0).expand(b, -1)

    # finished is int32 to match Agent C's K5 contract exactly (in-place
    # mutation by ops.logits_argmax, kernel or torch fallback alike).
    finished = torch.zeros(b, dtype=torch.int32, device=device)
    suppress_mask = tok.suppress_mask(device, timestamps=timestamps)
    blank_mask = torch.zeros(d.n_vocab, dtype=torch.float32, device=device)
    blank_mask[tok.blank_suppress_ids()] = float("-inf")

    self_cache = whisper_model.alloc_self_kv_cache(weights, b, max_len, device)
    cross_kv = whisper_model.precompute_cross_kv(audio_features, weights)
    scratch = whisper_model.alloc_decode_scratch(weights, b, device)

    # K4/K5-facing buffers, preallocated once per chunk batch and reused
    # every step (DECODE_INTEGRATION.md SS4) -- avoids a caching-allocator
    # round trip on the hottest path, 4 layers x up to 448 steps.
    next_tokens_buf = torch.empty(b, dtype=torch.int32, device=device)
    all_finished_buf = torch.empty(1, dtype=torch.uint8, device=device)
    pinned_host = torch.zeros(1, dtype=torch.uint8, pin_memory=True)

    # M6: [B] fp32, zeroed at decode start, caller (this function) never
    # resets it again -- DECODE_INTEGRATION.md SS3d's "caller zeroes,
    # kernel never resets" pattern, same as `finished`.
    sum_logprob = torch.zeros(b, dtype=torch.float32, device=device)

    # token_history for the ngram ban (DECODE_INTEGRATION.md SS3b): [B, 448]
    # int32, seeded with the prompt, history_len starts at n_prompt -- HF's
    # processor (and this kernel) operate over the full input_ids, prompt
    # included, not just generated tokens.
    token_history = torch.zeros(b, max_len, dtype=torch.int32, device=device)
    token_history[:, :n_prompt] = prompt.unsqueeze(0).expand(b, -1).to(torch.int32)
    history_len = n_prompt

    last_len = n_prompt
    no_speech_gate = torch.zeros(b, dtype=torch.bool, device=device)
    no_speech_prob = torch.zeros(b, dtype=torch.float32, device=device)

    # Positions 0..n_prompt-1: the fixed prompt, run through the same
    # per-step loop (populates the KV cache; logits are discarded except
    # at position 0 (no-speech gate) and the last prompt position (predicts
    # the first generated token)).
    for position in range(n_prompt):
        logits = whisper_model.decoder_step(
            tokens[:, position], position, weights, self_cache, cross_kv, scratch
        )
        if position == 0:
            # openai-whisper's no-speech heuristic (decoding.py SS_main_loop,
            # i==0): softmax the position-0 logits (SOT-only forward),
            # read off p(no_speech). This happens once per chunk batch.
            # M6: exposed to the caller (pipeline.py's retry-ladder decision)
            # regardless of whether the gate itself fires here.
            probs_at_sot = logits.float().softmax(dim=-1)
            p_no_speech = probs_at_sot[:, NO_SPEECH_TOKEN]
            no_speech_prob = p_no_speech
            no_speech_gate = p_no_speech > no_speech_threshold
            if no_speech_gate.any():
                finished = torch.where(
                    no_speech_gate, torch.ones_like(finished), finished
                )
        if position == n_prompt - 1:
            # openai-whisper's SuppressBlank only fires "if tokens.shape[1]
            # == sample_begin", i.e. exactly this first generated position.
            next_token, _ = ops.logits_argmax(
                logits, suppress_mask + blank_mask, finished, next_tokens_buf, all_finished_buf,
                token_history, history_len, enable_ngram_ban,
                sample_begin=n_prompt, enable_timestamp_rules=timestamps,
                sum_logprob=sum_logprob, temperature=temperature,
            )
            tokens[:, n_prompt] = next_token.long()
            token_history[:, history_len] = next_token
            history_len += 1
            last_len = n_prompt + 1

    # Positions n_prompt..max_len-1: generated tokens. Check cadence follows
    # DECODE_INTEGRATION.md SS6 exactly: queue the async D2H copy at
    # step%8==0, only inspect it at step%8==7 -- 7 steps of real GPU work
    # give the 1-byte copy time to land before the sync, so the sync is
    # "cheap" (work already done) rather than a fresh per-step stall. The
    # repetition-loop check piggybacks on the same %8==7 sync point.
    for position in range(n_prompt, max_len - 1):
        logits = whisper_model.decoder_step(
            tokens[:, position], position, weights, self_cache, cross_kv, scratch
        )
        next_token, _ = ops.logits_argmax(
            logits, suppress_mask, finished, next_tokens_buf, all_finished_buf,
            token_history, history_len, enable_ngram_ban,
            sample_begin=n_prompt, enable_timestamp_rules=timestamps,
            sum_logprob=sum_logprob, temperature=temperature,
        )
        tokens[:, position + 1] = next_token.long()
        token_history[:, history_len] = next_token
        history_len += 1
        last_len = position + 2

        step = position - n_prompt + 1  # 1-indexed count of generated tokens so far

        if step % check_interval == 0:
            pinned_host.copy_(all_finished_buf, non_blocking=True)
        elif step % check_interval == check_interval - 1:
            if repetition_brake:
                # .tolist() forces a D2H sync on its own -- this IS the
                # cheap/lagged sync point (the async copy above queued 7
                # steps ago has had that long to land), no separate
                # explicit synchronize() call is needed on this branch.
                gen_end = position + 2
                window_start = max(n_prompt, gen_end - REP_WINDOW)
                tail = tokens[:, window_start:gen_end].tolist()
                bad_rows = _find_repetition_loops(tail)
                if bad_rows:
                    idx = torch.tensor(bad_rows, dtype=torch.long, device=device)
                    finished.index_fill_(0, idx, 1)
            else:
                torch.cuda.current_stream().synchronize()
            if bool(pinned_host.item()):
                break

    return GreedyDecodeResult(
        tokens=tokens[:, :last_len], sum_logprob=sum_logprob, no_speech_prob=no_speech_prob
    )
