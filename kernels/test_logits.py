"""Correctness + microbenchmark for K5 (logits.hip), Agent C.

Run on gpu-host:
  HIP_VISIBLE_DEVICES=1 ~/r9700-whisper/.venv/bin/python \
      ~/r9700-whisper/rocm-whisper-src/kernels/test_logits.py

Correctness: exact match required for argmax token ids and finished
flags (these are integer/index outputs, not fp16 numeric ones -- no
tolerance is appropriate). Covers: normal argmax, a forced-EOT row
(finished already set), a row whose natural argmax lands on EOT itself,
and the all_finished AND-reduce in both the "some still running" and
"all done" cases.

Microbench: kernel (2 launches: argmax+finished-update, then
all-finished-reduce) vs the torch op sequence it replaces (mask add,
argmax, where-forced, finished update, all-finished reduce) -- 5 ops,
each with a device round-trip if any host `.item()` were needed (this
comparison keeps everything as GPU tensors so the torch side is not
unfairly penalized by a sync that a naive implementation would also need
to pay once per step).

Also covers the optional 3-gram-repeat ban pre-pass (benchmark-validity
review task): a torch reference mirroring HF's
NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3), constructed cases at
B in {8, 32} for each specified edge case, a regression check that the
feature being ON with nothing to ban is byte-identical to it being OFF,
and a microbench of the added ban-kernel cost in isolation.

Also covers the optional OpenAI Whisper timestamp rules pre-pass (Jim's
timestamps decode mode): a torch reference that is a verbatim port of
ApplyTimestampRules.apply from openai-whisper's whisper/decoding.py (read
directly from the venv on gpu-host as the source of truth, not from
memory), constructed cases at B in {8, 32} covering the first-position
rule (including the max_initial_timestamp cutoff boundary at exactly
index 50), both branches of the pair rule, both branches of the
monotonicity threshold, the logsumexp force-timestamp rule, EOT surviving
a post-timestamp text mask, finished rows being provably untouched, and a
microbench of the added cost.
"""

import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_api  # noqa: E402

DEVICE = "cuda:0"
VOCAB = 51866
EOT = decode_api.EOT_TOKEN


def _seed(n):
    torch.manual_seed(n)


def torch_logits_reference(logits, suppress_mask, finished):
    masked = logits.float() + suppress_mask.float()
    best_idx = masked.argmax(dim=-1).to(torch.int32)
    forced = finished.to(torch.bool)
    next_tokens = torch.where(forced, torch.full_like(best_idx, EOT), best_idx)
    new_finished = (forced | (next_tokens == EOT)).to(torch.int32)
    all_finished = new_finished.to(torch.bool).all()
    return next_tokens, new_finished, all_finished


def make_suppress_mask(vocab, n_suppressed=200, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.zeros(vocab, dtype=torch.float32)
    suppressed_ids = torch.randperm(vocab, generator=g)[:n_suppressed]
    mask[suppressed_ids] = float("-inf")
    return mask.to(DEVICE)


def check_case(B, label, finished_pattern, force_eot_argmax_rows=()):
    _seed(hash((B, label)) & 0xFFFFFFFF)
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE)
    suppress_mask = make_suppress_mask(VOCAB, seed=B)
    # keep argmax comfortably clear of suppressed ids by construction (random
    # suppression over 51866 ids vs. a randn row's single max is astronomically
    # unlikely to collide, so no special-casing needed there)

    for row in force_eot_argmax_rows:
        logits[row] = -100.0
        logits[row, EOT] = 100.0  # force this row's natural argmax to be EOT

    finished = torch.tensor(finished_pattern, dtype=torch.int32, device=DEVICE)
    assert finished.shape[0] == B

    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(logits, suppress_mask, finished)

    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(logits, suppress_mask, finished_kernel)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    finished_match = torch.equal(finished_kernel, ref_finished)
    all_finished_match = bool(all_finished.item()) == bool(ref_all_finished.item())

    ok = tokens_match and finished_match and all_finished_match
    print(f"[logits {label}] B={B}: tokens_match={tokens_match} finished_match={finished_match} "
          f"all_finished(kernel={bool(all_finished.item())}, ref={bool(ref_all_finished.item())}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        if not tokens_match:
            mism = (next_tokens != ref_tokens).nonzero().flatten().tolist()
            print(f"    token mismatches at rows: {mism[:10]}")
        if not finished_match:
            mism = (finished_kernel != ref_finished).nonzero().flatten().tolist()
            print(f"    finished mismatches at rows: {mism[:10]}")
    return ok


def check_tie_case(B, label, build_row_fn, finished_pattern=None):
    """Constructed-tie correctness check: `build_row_fn(logits, row)` mutates
    `logits[row]` in place to plant an EXACT multi-way tie at known indices.
    Background values are tiny noise, far below the tie value, so the tie
    is unambiguous. No suppression mask is applied here (zeros) -- these
    tests isolate the tie-break rule itself from the mask-add step, which
    is already covered separately by check_case().

    This targets the bug found in adversarial review: stage 1 used strict
    '>' (keeps whichever tied value a thread's strided walk happens to see
    first, not the lowest vocab index), and stage 2's tree reduction used
    strict '>' too (tree position does not correspond to original index
    order, so "keep the left slot" != "keep the lower index"). torch.argmax
    guarantees the lowest index wins on an exact tie; the kernel must match.
    """
    _seed(4242)
    logits = (torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * 0.01)
    suppress_mask = torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    for row in range(B):
        build_row_fn(logits, row)

    if finished_pattern is None:
        finished_pattern = [0] * B
    finished = torch.tensor(finished_pattern, dtype=torch.int32, device=DEVICE)

    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(logits, suppress_mask, finished)

    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(logits, suppress_mask, finished_kernel)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    finished_match = torch.equal(finished_kernel, ref_finished)
    all_finished_match = bool(all_finished.item()) == bool(ref_all_finished.item())

    ok = tokens_match and finished_match and all_finished_match
    print(f"[logits TIE {label}] B={B}: tokens_match={tokens_match} finished_match={finished_match} "
          f"all_finished(kernel={bool(all_finished.item())}, ref={bool(ref_all_finished.item())}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        if not tokens_match:
            mism = (next_tokens != ref_tokens).nonzero().flatten().tolist()
            print(f"    token mismatches at rows: {mism[:10]} "
                  f"(kernel={next_tokens[mism[:5]].tolist() if mism else []}, "
                  f"ref={ref_tokens[mism[:5]].tolist() if mism else []})")
        if not finished_match:
            mism = (finished_kernel != ref_finished).nonzero().flatten().tolist()
            print(f"    finished mismatches at rows: {mism[:10]}")
    return ok


def _tie_multiway_lowest_wins(logits, row):
    # scattered across different ARGMAX_BLOCK=256 strides AND different
    # stage-2 tree distances from each other -- exercises both stages'
    # tie-break, not just one. Lowest index (17) must win.
    tie_val = 50.0
    for idx in (17, 300, 5000, 12345, 40000, 51000):
        logits[row, idx] = tie_val


def _tie_zero_must_win(logits, row):
    # index 0 tied with two higher indices -- winner must be exactly 0.
    tie_val = 70.0
    for idx in (0, 999, 50000):
        logits[row, idx] = tie_val


def _tie_eot_vs_lower_index_lower_wins(logits, row):
    # EOT (50257) ties with a LOWER index (42) -- 42 must win, so the
    # sequence must NOT be forced to EOT and finished must stay 0. This is
    # the exact shape of bug the reviewer flagged: a tie touching EOT that
    # picks the wrong side terminates a sequence early (transcript
    # corruption).
    tie_val = 60.0
    logits[row, 42] = tie_val
    logits[row, EOT] = tie_val


def _tie_eot_vs_higher_index_eot_wins(logits, row):
    # EOT (50257) ties with a HIGHER index (50300) -- EOT is the lower
    # index here, so EOT must legitimately win and finished must become 1.
    # Opposite direction of the case above, to confirm the fix didn't just
    # bias every EOT tie toward "never terminate".
    tie_val = 65.0
    logits[row, EOT] = tie_val
    logits[row, 50300] = tie_val


T_HIST_CAP = 448
NGRAM_SIZE = 3


def torch_ngram_ban_reference(logits, token_history, history_len, finished, ngram_size=NGRAM_SIZE):
    """Mirrors HF's NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3)
    exactly (see transformers' generation/logits_process.py): for every
    row not already finished, find (t1, t2) = the last two tokens, scan
    every earlier position i (i+2 < history_len) for a prior (t1, t2, X)
    trigram, and set that row's logit for every such X to -inf. NO EOS/EOT
    exemption -- HF's processor does not have one, so neither does this.
    Returns a NEW tensor; does not mutate `logits`.
    """
    B, vocab = logits.shape
    out = logits.clone()
    if history_len < ngram_size:
        return out
    hist = token_history[:, :history_len].tolist()
    fin = finished.tolist()
    for b in range(B):
        if fin[b]:
            continue
        seq = hist[b]
        t1, t2 = seq[history_len - 2], seq[history_len - 1]
        banned = set()
        for i in range(history_len - 2):
            if seq[i] == t1 and seq[i + 1] == t2:
                banned.add(seq[i + 2])
        for tok in banned:
            out[b, tok] = float("-inf")
    return out


def make_token_history(B, seq, cap=T_HIST_CAP):
    """Build a [B, cap] int32 CUDA token-history tensor with the same
    `seq` (a python list[int]) written into every row -- matches the
    uniform-per-scenario style already used by check_tie_case above.
    Returns (token_history, history_len)."""
    hist = torch.zeros(B, cap, dtype=torch.int32, device=DEVICE)
    seq_t = torch.tensor(seq, dtype=torch.int32, device=DEVICE)
    hist[:, :len(seq)] = seq_t.unsqueeze(0).expand(B, -1)
    return hist, len(seq)


def check_ngram_case(B, label, seq, logit_overrides, finished_pattern=None):
    """seq: shared token_history sequence (list[int]) applied to every row.
    logit_overrides: {token_id: value} applied on top of tiny background
    noise to every row's logits, so the expected post-ban winner is known
    by construction (background noise can never beat an override value).
    """
    _seed(9191)
    logits = (torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * 0.01)
    for tok, val in logit_overrides.items():
        logits[:, tok] = val
    suppress_mask = torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    token_history, history_len = make_token_history(B, seq)

    if finished_pattern is None:
        finished_pattern = [0] * B
    finished = torch.tensor(finished_pattern, dtype=torch.int32, device=DEVICE)

    banned_logits_ref = torch_ngram_ban_reference(logits, token_history, history_len, finished)
    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(banned_logits_ref, suppress_mask, finished)

    logits_kernel = logits.clone()
    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits_kernel, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len, enable_ngram_ban=True)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    finished_match = torch.equal(finished_kernel, ref_finished)
    all_finished_match = bool(all_finished.item()) == bool(ref_all_finished.item())

    ok = tokens_match and finished_match and all_finished_match
    print(f"[logits NGRAM {label}] B={B} history_len={history_len}: tokens_match={tokens_match} "
          f"finished_match={finished_match} all_finished(kernel={bool(all_finished.item())}, "
          f"ref={bool(ref_all_finished.item())}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        if not tokens_match:
            mism = (next_tokens != ref_tokens).nonzero().flatten().tolist()
            print(f"    token mismatches at rows: {mism[:10]} "
                  f"(kernel={next_tokens[mism[:5]].tolist() if mism else []}, "
                  f"ref={ref_tokens[mism[:5]].tolist() if mism else []})")
        if not finished_match:
            mism = (finished_kernel != ref_finished).nonzero().flatten().tolist()
            print(f"    finished mismatches at rows: {mism[:10]}")
    return ok


def check_ngram_ban_inert(B, label="ban-enabled-inert"):
    """Regression check: enable_ngram_ban=True with a token history that
    has NO repeated bigram anywhere (strictly increasing token ids) must
    ban nothing and produce byte-identical results to the feature being
    off entirely -- i.e. turning the feature on does not change behavior
    by itself, only actual repeats do."""
    _seed(hash((B, "ban-inert")) & 0xFFFFFFFF)
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE)
    suppress_mask = make_suppress_mask(VOCAB, seed=B)
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    hist_len = 50
    seq = list(range(hist_len))  # strictly increasing -> no bigram ever repeats
    token_history, history_len = make_token_history(B, seq)

    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(logits, suppress_mask, finished)

    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len, enable_ngram_ban=True)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    finished_match = torch.equal(finished_kernel, ref_finished)
    all_finished_match = bool(all_finished.item()) == bool(ref_all_finished.item())
    ok = tokens_match and finished_match and all_finished_match
    print(f"[logits NGRAM {label}] B={B}: tokens_match={tokens_match} finished_match={finished_match} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ---- the 5 required ngram-ban scenarios -------------------------------
# Each seq/override pair is constructed so the expected post-ban winner is
# unambiguous: the override values are far above the ~N(0, 0.01) background
# noise, and each candidate's rank among the overrides is chosen by hand.

def _ngram_case_would_have_been_argmax():
    # history: ...,10,20,30,10,20 -- (10,20) previously completed with 30,
    # so 30 is banned this step even though it would have won outright.
    seq = [10, 20, 30, 10, 20]
    overrides = {30: 100.0, 31: 90.0}  # 30 banned -> 31 must win
    return seq, overrides


def _ngram_case_bans_eot():
    # history: ...,100,200,EOT,100,200 -- (100,200) previously completed
    # with EOT, so EOT is banned this step (matching HF exactly: no eos
    # exemption) even though it would otherwise win and end the sequence.
    seq = [100, 200, EOT, 100, 200]
    overrides = {EOT: 100.0, 123: 90.0}  # EOT banned -> 123 must win, finished stays 0
    return seq, overrides


def _ngram_case_multiple_matches():
    # (5,6) completed with 7 once and with 8 once earlier -- both banned.
    seq = [5, 6, 7, 1, 2, 3, 5, 6, 8, 5, 6]
    overrides = {7: 100.0, 8: 95.0, 9: 90.0}  # 7 and 8 banned -> 9 must win
    return seq, overrides


def _ngram_case_len_exactly_3():
    # smallest possible history where a ban is even possible: [X,X,X].
    seq = [42, 42, 42]
    overrides = {42: 100.0, 43: 90.0}  # 42 banned -> 43 must win
    return seq, overrides


def _ngram_case_alternating_loop_breaks():
    # A,B,A,B,A,B -- greedy decoding left alone would repeat this forever.
    # Only A (the exact completion of the (A,B) trigram) is banned per HF's
    # literal semantics -- B itself is not banned, so B must win here.
    seq = [11, 22, 11, 22, 11, 22]
    overrides = {11: 100.0, 22: 95.0, 33: 90.0}  # 11(A) banned -> 22(B) must win
    return seq, overrides


# =========================================================================
# OpenAI Whisper timestamp rules (Jim-approved timestamps decode mode).
# =========================================================================

TIMESTAMP_BEGIN = 50365
MAX_INITIAL_TS_INDEX = 50
# [sot, en, transcribe] -- 3-token prompt, no notimestamps token, per the
# timestamps-mode spec (distinct from the 4-token prompt used elsewhere).
PROMPT_TOKENS_TS_MODE = [50258, 50259, 50360]


def torch_apply_timestamp_rules_reference(logits, token_history, prompt_len, history_len,
                                           finished, suppress_mask,
                                           timestamp_begin=TIMESTAMP_BEGIN, eot=EOT,
                                           max_initial_timestamp_index=MAX_INITIAL_TS_INDEX):
    """Verbatim port of openai-whisper's ApplyTimestampRules.apply (see
    whisper/decoding.py in the venv on gpu-host -- read as the reference
    source for this port, not from memory). Adapted to this codebase's
    tensor/host-int conventions:
      - `tokens` -> token_history[:, :history_len] (int32)
      - `self.sample_begin` -> prompt_len
      - `self.tokenizer.timestamp_begin` -> timestamp_begin
      - `self.tokenizer.eot` -> eot
      - `self.max_initial_timestamp_index` -> max_initial_timestamp_index
      - `self.tokenizer.no_timestamps` suppression is intentionally OMITTED
        here -- Agent A's suppress_mask already covers it in the real
        pipeline (see DECODE_INTEGRATION.md SS3c); this port only covers
        the four numbered rules the kernel implements.
      - the rule-4 log_softmax step includes `suppress_mask` (added, not
        physically baked into the returned tensor) because that mirrors
        exactly what the kernel computes -- this codebase's suppress mask
        is register-only/virtual (see logits_argmax_kernel's header), never
        physically written into `logits`, so a faithful reference for what
        the KERNEL decides must add it the same way, not assume openai's
        "one fully-mutated shared tensor" model.
      - rules apply only to rows with finished[b]==0 (the task's explicit
        requirement; openai's own code has no notion of `finished` since
        it doesn't do batched lock-step decoding the same way).
    Returns a NEW fp16 tensor -- feed it straight into the existing
    torch_logits_reference() for the full expected (next_tokens, finished,
    all_finished), exactly mirroring how the real pipeline composes
    launch_timestamp_rules then the unchanged launch_logits_argmax.
    """
    out = logits.clone()
    B, vocab = out.shape
    tokens = token_history[:, :history_len]
    fin = finished.tolist()

    for k in range(B):
        if fin[k]:
            continue
        sampled_tokens = tokens[k, prompt_len:]
        seq = sampled_tokens.tolist()
        last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
        penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin

        if last_was_timestamp:
            if penultimate_was_timestamp:
                out[k, timestamp_begin:] = float("-inf")
            else:
                out[k, :eot] = float("-inf")

        timestamps = sampled_tokens[sampled_tokens.ge(timestamp_begin)]
        if timestamps.numel() > 0:
            if last_was_timestamp and not penultimate_was_timestamp:
                timestamp_last = int(timestamps[-1])
            else:
                timestamp_last = int(timestamps[-1]) + 1
            out[k, timestamp_begin:timestamp_last] = float("-inf")

    if history_len == prompt_len:  # tokens.shape[1] == sample_begin, batch-wide under lock-step
        for k in range(B):
            if fin[k]:
                continue
            out[k, :timestamp_begin] = float("-inf")
            if max_initial_timestamp_index is not None:
                last_allowed = timestamp_begin + max_initial_timestamp_index
                out[k, last_allowed + 1:] = float("-inf")

    logprobs = torch.log_softmax(out.float() + suppress_mask.float(), dim=-1)
    for k in range(B):
        if fin[k]:
            continue
        timestamp_logprob = logprobs[k, timestamp_begin:].logsumexp(dim=-1)
        max_text_token_logprob = logprobs[k, :timestamp_begin].max()
        if timestamp_logprob > max_text_token_logprob:
            out[k, :timestamp_begin] = float("-inf")

    return out


def make_token_history_ts(B, prompt, seq, cap=T_HIST_CAP):
    """Builds a [B, cap] int32 token_history with `prompt + seq` written
    into every row. Returns (token_history, prompt_len, history_len)."""
    full = list(prompt) + list(seq)
    hist = torch.zeros(B, cap, dtype=torch.int32, device=DEVICE)
    full_t = torch.tensor(full, dtype=torch.int32, device=DEVICE)
    hist[:, :len(full)] = full_t.unsqueeze(0).expand(B, -1)
    return hist, len(prompt), len(full)


def check_timestamp_case(B, label, seq, logit_overrides, finished_pattern=None,
                          background_scale=0.01, prompt=PROMPT_TOKENS_TS_MODE):
    """seq: the shared post-prompt token sequence (list[int]) applied to
    every row. logit_overrides: {token_id: value} applied on top of tiny
    background noise, so the expected post-rules winner is known by
    construction (background noise can never beat an override value, and
    override margins are chosen wide enough to be robust to fp16/fp32
    rounding, not knife-edge)."""
    _seed(31337)
    logits = (torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * background_scale)
    for tok, val in logit_overrides.items():
        logits[:, tok] = val
    suppress_mask = torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    token_history, prompt_len, history_len = make_token_history_ts(B, prompt, seq)

    if finished_pattern is None:
        finished_pattern = [0] * B
    finished = torch.tensor(finished_pattern, dtype=torch.int32, device=DEVICE)

    ref_masked = torch_apply_timestamp_rules_reference(
        logits, token_history, prompt_len, history_len, finished, suppress_mask)
    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(ref_masked, suppress_mask, finished)

    logits_kernel = logits.clone()
    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits_kernel, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len,
        enable_timestamp_rules=True, prompt_len=prompt_len)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    finished_match = torch.equal(finished_kernel, ref_finished)
    all_finished_match = bool(all_finished.item()) == bool(ref_all_finished.item())

    ok = tokens_match and finished_match and all_finished_match
    print(f"[logits TS {label}] B={B} prompt_len={prompt_len} history_len={history_len}: "
          f"tokens_match={tokens_match} finished_match={finished_match} "
          f"all_finished(kernel={bool(all_finished.item())}, ref={bool(ref_all_finished.item())}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        if not tokens_match:
            mism = (next_tokens != ref_tokens).nonzero().flatten().tolist()
            print(f"    token mismatches at rows: {mism[:10]} "
                  f"(kernel={next_tokens[mism[:5]].tolist() if mism else []}, "
                  f"ref={ref_tokens[mism[:5]].tolist() if mism else []})")
        if not finished_match:
            mism = (finished_kernel != ref_finished).nonzero().flatten().tolist()
            print(f"    finished mismatches at rows: {mism[:10]}")
    return ok


def check_timestamp_finished_rows_untouched(B, label="finished-rows-untouched"):
    """Rows with finished[b]!=0 must not be mutated by the timestamp-rules
    kernel AT ALL -- not "the observable next_token happens to still be
    EOT" (that would be true even if the finished-row skip were broken,
    since logits_argmax_kernel forces EOT for finished rows regardless of
    what their logits say), but a literal byte-for-byte unchanged logits
    row. Uses the most aggressive rule available (first-position,
    seq=[] -- masks almost the entire row) specifically so a broken skip
    would be impossible to miss."""
    _seed(2025)
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE)
    suppress_mask = torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    token_history, prompt_len, history_len = make_token_history_ts(B, PROMPT_TOKENS_TS_MODE, [])
    finished_pattern = [1 if i % 2 == 0 else 0 for i in range(B)]
    finished = torch.tensor(finished_pattern, dtype=torch.int32, device=DEVICE)

    logits_before = logits.clone()
    finished_kernel = finished.clone()
    decode_api.logits_argmax_step(
        logits, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len,
        enable_timestamp_rules=True, prompt_len=prompt_len)
    torch.cuda.synchronize()

    finished_rows = [i for i in range(B) if finished_pattern[i] == 1]
    unfinished_rows = [i for i in range(B) if finished_pattern[i] == 0]
    finished_untouched = torch.equal(logits[finished_rows], logits_before[finished_rows])
    unfinished_changed = not torch.equal(logits[unfinished_rows], logits_before[unfinished_rows])

    ok = finished_untouched and unfinished_changed
    print(f"[logits TS {label}] B={B}: finished_rows_untouched={finished_untouched} "
          f"unfinished_rows_did_change={unfinished_changed} -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---- the required timestamp-rules scenarios ----------------------------

def _ts_case_first_position():
    # seq=[] -> rule 3: all text masked, only [TIMESTAMP_BEGIN, +50] allowed.
    seq = []
    overrides = {
        1234: 100.0,           # text token, must be masked
        TIMESTAMP_BEGIN + 51: 95.0,  # one past the cutoff, must be masked
        TIMESTAMP_BEGIN + 50: 50.0,  # exactly at the cutoff, must survive and win
    }
    return seq, overrides


def _ts_case_first_position_boundary_exact():
    # isolates the index-50 boundary: nothing competes with it at all, so
    # a winner other than exactly TIMESTAMP_BEGIN+50 proves an off-by-one.
    seq = []
    overrides = {TIMESTAMP_BEGIN + 50: 100.0}
    return seq, overrides


def _ts_case_pair_rule_mask_all_timestamps():
    # last two tokens both timestamps -> penultimate_was_timestamp=True ->
    # ALL timestamp tokens banned this step.
    seq = [50370, 50380]
    overrides = {50390: 100.0, 999: 90.0}  # 50390(ts) banned -> 999(text) must win
    return seq, overrides


def _ts_case_pair_rule_mask_all_text():
    # last token is a timestamp, the one before is not -> penultimate_was_timestamp=False
    # -> ALL non-EOT text tokens banned (timestamp range untouched by this branch).
    seq = [111, 50370]
    overrides = {2222: 100.0, 50410: 90.0}  # 2222(text) banned -> 50410(ts) must win
    return seq, overrides


def _ts_case_monotonicity_equal_allowed():
    # same seq as pair_rule_mask_all_text: last_was_timestamp and not
    # penultimate_was_timestamp -> threshold = last_ts (EQUAL allowed).
    seq = [111, 50370]
    overrides = {
        50370: 100.0,  # == threshold, must remain ALLOWED (equal-allowed semantics)
        50367: 95.0,   # < threshold, must be banned despite the higher score
    }
    return seq, overrides


def _ts_case_monotonicity_strictly_greater():
    # ends in plain text (last_was_timestamp=False) but an earlier timestamp
    # exists -> threshold = last_ts + 1 (the timestamp itself now banned too).
    seq = [50370, 111, 222]
    overrides = {
        50370: 100.0,  # == last_ts, now BANNED (strictly-greater branch)
        50371: 90.0,   # first value not banned, must win
    }
    return seq, overrides


def _ts_case_eot_survives_text_mask():
    # same trigger as pair_rule_mask_all_text (rule 1's "mask all text"
    # branch), but this test's whole point is EOT specifically: EOT must
    # remain selectable even though "all text" got masked, because the
    # masked range is [0, EOT) -- EOT itself is excluded from it.
    seq = [111, 50370]
    overrides = {EOT: 100.0, 2222: 90.0}  # 2222 masked -> EOT must win
    return seq, overrides


def check_timestamp_logsumexp_force(B, label="logsumexp-forces-timestamp"):
    """Rule 4: construct a row with one clear best text token, but with
    enough moderate-value timestamp tokens that their combined
    (logsumexp) mass exceeds it -- must force a timestamp winner, matching
    the literal log_softmax reference bit-for-bit (this is also the test
    that proves the kernel's raw-logit logsumexp/max simplification
    produces the identical decision to openai's log_softmax-based one --
    see the derivation in logits.hip's file header)."""
    _seed(31337)
    logits = (torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * 0.01)
    logits[:, 999] = 10.0  # best single text token
    logits[:, TIMESTAMP_BEGIN:] = 8.0  # every timestamp token: individually well below 10.0...
    logits[:, 51000] = 8.5  # ...but logsumexp(~1500 * exp(8.0)) ~= 8+ln(1501) ~= 15.3 >> 10.0
    suppress_mask = torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    # no prior timestamps in history -> rules 1-3 are no-ops, isolating rule 4
    token_history, prompt_len, history_len = make_token_history_ts(B, PROMPT_TOKENS_TS_MODE, [111, 222])
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    ref_masked = torch_apply_timestamp_rules_reference(
        logits, token_history, prompt_len, history_len, finished, suppress_mask)
    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(ref_masked, suppress_mask, finished)

    logits_kernel = logits.clone()
    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits_kernel, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len,
        enable_timestamp_rules=True, prompt_len=prompt_len)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    ok = tokens_match and torch.equal(finished_kernel, ref_finished)
    print(f"[logits TS {label}] B={B}: tokens_match={tokens_match} "
          f"(kernel picked {next_tokens[0].item()}, ref picked {ref_tokens[0].item()}, "
          f"expected a timestamp token, i.e. >= {TIMESTAMP_BEGIN}) -> {'PASS' if ok else 'FAIL'}")
    return ok


# =========================================================================
# Rule-4 NaN-poisoning regression (production bug, found via Agent A's
# integration cross-check on real audio -- see the final report for the
# full root-cause writeup).
#
# Root cause: timestamp_rules_kernel's online-logsumexp accumulator
# started each thread at the sentinel (ts_local_m=-INFINITY, ts_local_s=0).
# When rule 2's monotonicity mask had already banned enough of the LOW end
# of the timestamp range -- which happens routinely a few timestamps into
# real decoding, not some exotic edge case -- a thread's FIRST
# timestamp-range element in its strided walk could itself be an
# already-masked -inf value. The old code's accumulation branch computed
# `expf(val - ts_local_m)` = `expf(-inf - (-inf))` = `expf(NaN)` = NaN in
# that case, permanently poisoning that thread's partial sum (NaN +
# anything = NaN), which the block-merge then propagated into the whole
# row's ts_lse. Any comparison against NaN is false in IEEE754, so
# `ts_lse > text_max` silently evaluated False forever whenever this
# happened, regardless of how decisive the true margin was -- exactly
# Agent A's reported symptom (a ~1.76-log-prob-margin case where EOT
# should have been banned and wasn't).
#
# Fix (in logits.hip): skip -inf elements in the timestamp-range
# accumulation outright. This is mathematically exact (a -inf logit
# contributes exp(-inf)=0 to the sum either way) and avoids the inf-inf
# subtraction rather than special-casing it after the fact. The
# max-based text_local_max side never had this problem (max(-inf,-inf)
# is well-defined) and is unchanged.
# =========================================================================

def check_timestamp_logsumexp_with_masked_low_range(B, threshold_offset, label=None):
    """The permanent regression test for the bug above, generalized as a
    sweep parameter: `threshold_offset` controls how many low timestamp
    entries rule 2's monotonicity masks (via an earlier timestamp token at
    TIMESTAMP_BEGIN + threshold_offset, ending in plain text so rule 1
    stays out of the way). Same collective-mass-vs-single-peak setup as
    check_timestamp_logsumexp_force above, but now with that masking
    layered on top -- this is exactly Agent A's real-audio shape:
    monotonicity has already advanced, and rule 4 must still correctly
    force a timestamp winner over a single strong text candidate.
    """
    label = label or f"logsumexp-masked-low-range-offset{threshold_offset}"
    _seed(31337)
    logits = (torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * 0.01)
    logits[:, 999] = 10.0        # single best text token -- must lose if rule 4 fires
    logits[:, TIMESTAMP_BEGIN:] = 8.0   # collectively-dominant timestamp mass
    logits[:, TIMESTAMP_BEGIN + 1499] = 8.5  # clear top-of-range peak, survives any offset tested below
    suppress_mask = torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    if threshold_offset > 0:
        # ends in plain text (last_was_ts=False) -> rule 1 doesn't fire;
        # rule 2 alone masks [TIMESTAMP_BEGIN, TIMESTAMP_BEGIN+threshold_offset+1).
        seq = [111, TIMESTAMP_BEGIN + threshold_offset, 222, 333]
    else:
        seq = [111, 222]  # offset=0 control: no masking at all
    token_history, prompt_len, history_len = make_token_history_ts(B, PROMPT_TOKENS_TS_MODE, seq)
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    ref_masked = torch_apply_timestamp_rules_reference(
        logits, token_history, prompt_len, history_len, finished, suppress_mask)
    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(ref_masked, suppress_mask, finished)

    logits_kernel = logits.clone()
    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits_kernel, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len,
        enable_timestamp_rules=True, prompt_len=prompt_len)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    ok = tokens_match and torch.equal(finished_kernel, ref_finished)
    print(f"[logits TS {label}] B={B} threshold_offset={threshold_offset}: tokens_match={tokens_match} "
          f"(kernel={next_tokens[0].item()}, ref={ref_tokens[0].item()}, "
          f"text token 999 survived={logits_kernel[0, 999].item() > -1e30}) -> {'PASS' if ok else 'FAIL'}")
    return ok


# Sweep offsets chosen to exercise every TS_RULES_BLOCK=256-wide reduction
# stride boundary: 0 (no masking, control), 1 (exactly one thread's first
# hit affected), 127/128 (half a stride window), 255/256/257 (exactly one
# full window, then one past it -- this is where 100% of threads first hit
# a masked element, the worst case for the old bug), and deep into the
# range (900, 1300) matching realistic mid-to-late-utterance decode
# positions like Agent A's real repro.
_TS_STRIDE_SWEEP_OFFSETS = [0, 1, 127, 128, 255, 256, 257, 500, 900, 1300]


def check_timestamp_near_tie(B, margin, label):
    """Rule 4's boundary-sign regression: construct a row where
    logsumexp(ts) is only `margin` nats away from the best text token (in
    either direction), isolated from rule 2's masking (seq has no prior
    timestamps) so this specifically probes the comparison itself, not the
    NaN-poisoning interaction covered above. `margin` is deliberately kept
    well above fp16's rounding floor (~1e-3 at this magnitude) so the test
    is stable, not a coin flip on quantization noise."""
    _seed(4711)
    logits = (torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * 0.001)
    text_val = 10.0
    logits[:, 999] = text_val
    n_ts = 1501
    ts_val = text_val - math.log(n_ts) + margin
    logits[:, TIMESTAMP_BEGIN:] = ts_val
    suppress_mask = torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    token_history, prompt_len, history_len = make_token_history_ts(B, PROMPT_TOKENS_TS_MODE, [111, 222])
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    ref_masked = torch_apply_timestamp_rules_reference(
        logits, token_history, prompt_len, history_len, finished, suppress_mask)
    ref_tokens, ref_finished, ref_all_finished = torch_logits_reference(ref_masked, suppress_mask, finished)

    logits_kernel = logits.clone()
    finished_kernel = finished.clone()
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits_kernel, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len,
        enable_timestamp_rules=True, prompt_len=prompt_len)
    torch.cuda.synchronize()

    tokens_match = torch.equal(next_tokens, ref_tokens)
    ok = tokens_match and torch.equal(finished_kernel, ref_finished)
    print(f"[logits TS {label}] B={B} target_margin={margin:+.3f}nats: tokens_match={tokens_match} "
          f"(kernel={next_tokens[0].item()}, ref={ref_tokens[0].item()}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def _median_us(fn, warmup=10, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)


def bench(B):
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE)
    suppress_mask = make_suppress_mask(VOCAB, seed=B)
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    next_tokens = torch.empty(B, dtype=torch.int32, device=DEVICE)
    all_finished = torch.empty(1, dtype=torch.uint8, device=DEVICE)

    def kernel_call():
        decode_api.logits_argmax_step(logits, suppress_mask, finished, next_tokens, all_finished)

    def torch_call():
        masked = logits.float() + suppress_mask
        best_idx = masked.argmax(dim=-1).to(torch.int32)
        forced = finished.to(torch.bool)
        tok = torch.where(forced, torch.full_like(best_idx, EOT), best_idx)
        new_fin = (forced | (tok == EOT)).to(torch.int32)
        finished.copy_(new_fin)
        _ = new_fin.to(torch.bool).all()  # stays a GPU tensor, no .item() sync

    k_us = _median_us(kernel_call)
    t_us = _median_us(torch_call)
    print(f"[bench logits] B={B} vocab={VOCAB}: kernel(2 launches)={k_us:.1f}us "
          f"torch_seq(5 ops)={t_us:.1f}us speedup={t_us / k_us:.2f}x")
    return k_us, t_us


def bench_ngram_ban_delta(B, history_len=100):
    """Isolates the added cost of the ngram-ban pre-pass: same step, once
    with enable_ngram_ban=False and once with it True (against a
    realistic mid-decode history length), holding everything else fixed."""
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE)
    suppress_mask = make_suppress_mask(VOCAB, seed=B)
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    next_tokens = torch.empty(B, dtype=torch.int32, device=DEVICE)
    all_finished = torch.empty(1, dtype=torch.uint8, device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(B)
    seq = torch.randint(0, VOCAB, (history_len,), generator=g).tolist()
    token_history, hlen = make_token_history(B, seq)

    def call_without_ban():
        decode_api.logits_argmax_step(logits, suppress_mask, finished, next_tokens, all_finished)

    def call_with_ban():
        decode_api.logits_argmax_step(logits, suppress_mask, finished, next_tokens, all_finished,
                                       token_history=token_history, history_len=hlen, enable_ngram_ban=True)

    us_off = _median_us(call_without_ban)
    us_on = _median_us(call_with_ban)
    delta = us_on - us_off
    print(f"[bench ngram-ban delta] B={B} history_len={hlen}: without_ban={us_off:.1f}us "
          f"with_ban={us_on:.1f}us delta={delta:+.1f}us")
    return us_off, us_on


def bench_timestamp_rules_delta(B, seq_len=100):
    """Isolates the added cost of the timestamp-rules pre-pass: same step,
    once with enable_timestamp_rules=False and once with it True (against
    a realistic mid-decode sequence length), holding everything else
    fixed."""
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE)
    suppress_mask = make_suppress_mask(VOCAB, seed=B)
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    next_tokens = torch.empty(B, dtype=torch.int32, device=DEVICE)
    all_finished = torch.empty(1, dtype=torch.uint8, device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(B + 777)
    seq = torch.randint(0, VOCAB, (seq_len,), generator=g).tolist()
    token_history, prompt_len, history_len = make_token_history_ts(B, PROMPT_TOKENS_TS_MODE, seq)

    def call_without():
        decode_api.logits_argmax_step(logits, suppress_mask, finished, next_tokens, all_finished)

    def call_with():
        decode_api.logits_argmax_step(logits, suppress_mask, finished, next_tokens, all_finished,
                                       token_history=token_history, history_len=history_len,
                                       enable_timestamp_rules=True, prompt_len=prompt_len)

    us_off = _median_us(call_without)
    us_on = _median_us(call_with)
    delta = us_on - us_off
    print(f"[bench timestamp-rules delta] B={B} seq_len={seq_len}: without={us_off:.1f}us "
          f"with={us_on:.1f}us delta={delta:+.1f}us")
    return us_off, us_on


# =========================================================================
# Optional per-step chosen-token sum_logprob accumulation
# (temperature-retry ladder, Jim-approved).
#
# Reference: ported from openai-whisper's whisper/decoding.py
# GreedyDecoder.update (read directly from the venv on gpu-host):
#   logprobs = F.log_softmax(logits.float(), dim=-1)
#   current_logprobs = logprobs[arange(B), next_tokens]
#   sum_logprobs += current_logprobs * (tokens[:, -1] != eot)
# i.e. logprob(chosen) = logit[chosen] - logsumexp(row), accumulated iff
# the row was NOT already finished BEFORE this step (openai's check is on
# the token before this step, so the step a sequence first emits EOT is
# still counted -- "up to and including the first EOT").
# =========================================================================

def _reference_masked_logits(logits, suppress_mask, finished, token_history=None,
                              history_len=0, prompt_len=0, use_ngram=False, use_timestamp=False):
    """Applies the same sequence of mask MUTATIONS decode_api.logits_argmax_step
    would apply (ban, then timestamp rules -- both real -inf writes into a
    clone), matching kernel call order exactly. Returns the fp16 tensor the
    real argmax/logprob kernel would have as `logits` at read time; caller
    still needs to add suppress_mask (virtual/register-only in the real
    kernel, so not baked in here) to get the final masked view."""
    out = logits.clone()
    if use_ngram:
        out = torch_ngram_ban_reference(out, token_history, history_len, finished)
    if use_timestamp:
        out = torch_apply_timestamp_rules_reference(
            out, token_history, prompt_len, history_len, finished, suppress_mask)
    return out


def check_sum_logprob_step(B, label, use_suppress=False, use_ngram=False, use_timestamp=False,
                            finished_pattern=None):
    """Single-step correctness: kernel's sum_logprob delta (from a
    freshly-zeroed accumulator) vs. log_softmax+gather on the SAME masked
    logits the kernel would see, across suppress/ngram/timestamp-rules
    active and inactive."""
    _seed(hash((B, label)) & 0xFFFFFFFF)
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * 2.0  # realistic-ish scale
    suppress_mask = make_suppress_mask(VOCAB, seed=B) if use_suppress else \
        torch.zeros(VOCAB, dtype=torch.float32, device=DEVICE)
    finished_pattern = finished_pattern if finished_pattern is not None else [0] * B
    finished = torch.tensor(finished_pattern, dtype=torch.int32, device=DEVICE)

    token_history = None
    prompt_len = 0
    history_len = 0
    if use_timestamp:
        seq = [5, 6, 7, 5, 6, 8, 111, 222]  # has an actual ngram repeat AND no timestamps
        token_history, prompt_len, history_len = make_token_history_ts(B, PROMPT_TOKENS_TS_MODE, seq)
    elif use_ngram:
        seq = [5, 6, 7, 5, 6, 8, 111, 222]
        token_history, history_len = make_token_history(B, seq)

    ref_masked_fp16 = _reference_masked_logits(
        logits, suppress_mask, finished, token_history, history_len, prompt_len, use_ngram, use_timestamp)
    ref_masked = ref_masked_fp16.float() + suppress_mask.float()
    ref_best = ref_masked.argmax(dim=-1)
    ref_lp = torch.log_softmax(ref_masked, dim=-1)
    ref_chosen_lp = ref_lp.gather(1, ref_best.unsqueeze(1)).squeeze(1)
    was_finished_bool = finished.to(torch.bool)
    ref_contribution = torch.where(was_finished_bool, torch.zeros_like(ref_chosen_lp), ref_chosen_lp)
    ref_next_tokens = torch.where(was_finished_bool, torch.full_like(ref_best, EOT), ref_best).to(torch.int32)

    logits_kernel = logits.clone()
    finished_kernel = finished.clone()
    sum_logprob = torch.zeros(B, dtype=torch.float32, device=DEVICE)
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits_kernel, suppress_mask, finished_kernel,
        token_history=token_history, history_len=history_len, prompt_len=prompt_len,
        enable_ngram_ban=use_ngram, enable_timestamp_rules=use_timestamp,
        sum_logprob=sum_logprob)
    torch.cuda.synchronize()

    logprob_match = torch.allclose(sum_logprob, ref_contribution, rtol=1e-4, atol=1e-4)
    tokens_match = torch.equal(next_tokens, ref_next_tokens)
    max_diff = (sum_logprob - ref_contribution).abs().max().item()
    ok = logprob_match and tokens_match
    print(f"[logits LOGPROB {label}] B={B} suppress={use_suppress} ngram={use_ngram} "
          f"timestamp={use_timestamp}: logprob_match={logprob_match} tokens_match={tokens_match} "
          f"max_abs_diff={max_diff:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_sum_logprob_chained(B, n_steps=20, label="chained-20-steps"):
    """Runs n_steps of fresh per-step random logits through both the
    kernel and an independent step-by-step torch reference sharing the
    SAME per-step logits (but otherwise evolving its own finished/
    sum_logprob state in parallel) -- proves fp32 accumulation matches a
    running torch sum within tolerance over a realistic multi-step run,
    not just a single call. Forces one row to EOT mid-run so the
    finished-row exclusion gets exercised partway through, not only at
    the very end."""
    _seed(2024)
    suppress_mask = make_suppress_mask(VOCAB, seed=99)
    finished_kernel = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    finished_ref = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    sum_logprob_kernel = torch.zeros(B, dtype=torch.float32, device=DEVICE)
    sum_logprob_ref = torch.zeros(B, dtype=torch.float32, device=DEVICE)

    all_ok = True
    for step in range(n_steps):
        logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE) * 2.0
        if step == n_steps // 2:
            logits[0] = -50.0
            logits[0, EOT] = 50.0  # force row 0 to EOT mid-run

        masked = logits.float() + suppress_mask.float()
        best = masked.argmax(dim=-1)
        lp = torch.log_softmax(masked, dim=-1)
        chosen_lp = lp.gather(1, best.unsqueeze(1)).squeeze(1)
        was_finished_bool = finished_ref.to(torch.bool)
        contribution = torch.where(was_finished_bool, torch.zeros_like(chosen_lp), chosen_lp)
        sum_logprob_ref = sum_logprob_ref + contribution
        tok = torch.where(was_finished_bool, torch.full_like(best, EOT), best)
        finished_ref = (was_finished_bool | (tok == EOT)).to(torch.int32)

        logits_kernel_step = logits.clone()
        next_tokens, all_finished = decode_api.logits_argmax_step(
            logits_kernel_step, suppress_mask, finished_kernel,
            sum_logprob=sum_logprob_kernel)
        torch.cuda.synchronize()

        step_tokens_match = torch.equal(next_tokens, tok.to(torch.int32))
        step_finished_match = torch.equal(finished_kernel, finished_ref)
        if not (step_tokens_match and step_finished_match):
            print(f"    step {step}: tokens_match={step_tokens_match} finished_match={step_finished_match}")
        all_ok = all_ok and step_tokens_match and step_finished_match

    logprob_match = torch.allclose(sum_logprob_kernel, sum_logprob_ref, rtol=1e-4, atol=1e-4)
    all_ok = all_ok and logprob_match
    max_diff = (sum_logprob_kernel - sum_logprob_ref).abs().max().item()
    print(f"[logits LOGPROB {label}] B={B} n_steps={n_steps}: logprob_match={logprob_match} "
          f"max_abs_diff={max_diff:.3e} (rtol=1e-4,atol=1e-4) -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def bench_sum_logprob_delta(B):
    """Isolates the added cost of logprob accumulation: same step, once
    via launch_logits_argmax (sum_logprob=None) and once via
    launch_logits_argmax_lp (sum_logprob set), holding everything else
    fixed."""
    logits = torch.randn(B, VOCAB, dtype=torch.float16, device=DEVICE)
    suppress_mask = make_suppress_mask(VOCAB, seed=B)
    finished = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    next_tokens = torch.empty(B, dtype=torch.int32, device=DEVICE)
    all_finished = torch.empty(1, dtype=torch.uint8, device=DEVICE)
    sum_logprob = torch.zeros(B, dtype=torch.float32, device=DEVICE)

    def call_without():
        decode_api.logits_argmax_step(logits, suppress_mask, finished, next_tokens, all_finished)

    def call_with():
        decode_api.logits_argmax_step(logits, suppress_mask, finished, next_tokens, all_finished,
                                       sum_logprob=sum_logprob)

    us_off = _median_us(call_without)
    us_on = _median_us(call_with)
    delta = us_on - us_off
    print(f"[bench sum-logprob delta] B={B}: without={us_off:.1f}us with={us_on:.1f}us delta={delta:+.1f}us")
    return us_off, us_on


def main():
    assert torch.cuda.is_available(), "no HIP/CUDA device visible"
    all_ok = True
    for B in (8, 32):
        # normal case: nobody finished yet
        all_ok &= check_case(B, "normal", finished_pattern=[0] * B)
        # some already finished (forced-eot path) mixed with running ones
        pattern = [1 if i % 3 == 0 else 0 for i in range(B)]
        all_ok &= check_case(B, "mixed-finished", finished_pattern=pattern)
        # a natural argmax that lands on EOT itself (row 0), everyone else running
        pattern2 = [0] * B
        all_ok &= check_case(B, "natural-eot-argmax", finished_pattern=pattern2,
                              force_eot_argmax_rows=(0,))
        # everybody already finished -> all_finished must be True
        all_ok &= check_case(B, "all-finished", finished_pattern=[1] * B)

        # --- constructed exact-tie tests (adversarial review fix) ---
        all_ok &= check_tie_case(B, "multiway-lowest-wins", _tie_multiway_lowest_wins)
        all_ok &= check_tie_case(B, "zero-must-win", _tie_zero_must_win)
        all_ok &= check_tie_case(B, "eot-vs-lower-index", _tie_eot_vs_lower_index_lower_wins)
        all_ok &= check_tie_case(B, "eot-vs-higher-index", _tie_eot_vs_higher_index_eot_wins)

        # --- 3-gram-repeat ban tests (benchmark-validity review) ---
        seq, ov = _ngram_case_would_have_been_argmax()
        all_ok &= check_ngram_case(B, "would-have-been-argmax", seq, ov)
        seq, ov = _ngram_case_bans_eot()
        all_ok &= check_ngram_case(B, "bans-eot", seq, ov)
        seq, ov = _ngram_case_multiple_matches()
        all_ok &= check_ngram_case(B, "multiple-matches", seq, ov)
        seq, ov = _ngram_case_len_exactly_3()
        all_ok &= check_ngram_case(B, "len-exactly-3", seq, ov)
        seq, ov = _ngram_case_alternating_loop_breaks()
        all_ok &= check_ngram_case(B, "alternating-loop-breaks", seq, ov)
        # regression: feature ON but nothing to ban == feature OFF
        all_ok &= check_ngram_ban_inert(B)

        # --- timestamp rules tests (Jim-approved timestamps decode mode) ---
        seq, ov = _ts_case_first_position()
        all_ok &= check_timestamp_case(B, "first-position", seq, ov)
        seq, ov = _ts_case_first_position_boundary_exact()
        all_ok &= check_timestamp_case(B, "first-position-boundary-exact-50", seq, ov)
        seq, ov = _ts_case_pair_rule_mask_all_timestamps()
        all_ok &= check_timestamp_case(B, "pair-rule-mask-all-timestamps", seq, ov)
        seq, ov = _ts_case_pair_rule_mask_all_text()
        all_ok &= check_timestamp_case(B, "pair-rule-mask-all-text", seq, ov)
        seq, ov = _ts_case_monotonicity_equal_allowed()
        all_ok &= check_timestamp_case(B, "monotonicity-equal-allowed", seq, ov)
        seq, ov = _ts_case_monotonicity_strictly_greater()
        all_ok &= check_timestamp_case(B, "monotonicity-strictly-greater", seq, ov)
        seq, ov = _ts_case_eot_survives_text_mask()
        all_ok &= check_timestamp_case(B, "eot-survives-text-mask", seq, ov)
        all_ok &= check_timestamp_logsumexp_force(B)
        all_ok &= check_timestamp_finished_rows_untouched(B)

        # --- Rule-4 NaN-poisoning bug: permanent regression (see report) ---
        for offset in _TS_STRIDE_SWEEP_OFFSETS:
            all_ok &= check_timestamp_logsumexp_with_masked_low_range(B, offset)
        all_ok &= check_timestamp_near_tie(B, +0.05, "near-tie-should-mask")
        all_ok &= check_timestamp_near_tie(B, -0.05, "near-tie-should-not-mask")

        # --- sum_logprob accumulation (temperature-retry ladder) ---
        all_ok &= check_sum_logprob_step(B, "baseline")
        all_ok &= check_sum_logprob_step(B, "suppress-only", use_suppress=True)
        all_ok &= check_sum_logprob_step(B, "ngram-only", use_ngram=True)
        all_ok &= check_sum_logprob_step(B, "timestamp-only", use_timestamp=True)
        all_ok &= check_sum_logprob_step(B, "suppress+ngram+timestamp", use_suppress=True,
                                          use_ngram=True, use_timestamp=True)
        mixed_finished = [1 if i % 3 == 0 else 0 for i in range(B)]
        all_ok &= check_sum_logprob_step(B, "finished-row-exclusion", use_suppress=True,
                                          finished_pattern=mixed_finished)
        all_ok &= check_sum_logprob_chained(B)

    print("\n=== correctness summary:", "ALL PASS" if all_ok else "FAILURES ABOVE", "===\n")

    print("=== microbenchmarks (median of 10, 10 warmup) ===")
    for B in (8, 32):
        bench(B)
    print("=== ngram-ban added-cost delta (median of 10, 10 warmup) ===")
    for B in (8, 32):
        bench_ngram_ban_delta(B)
    print("=== timestamp-rules added-cost delta (median of 10, 10 warmup) ===")
    for B in (8, 32):
        bench_timestamp_rules_delta(B)
    print("=== sum-logprob added-cost delta (median of 10, 10 warmup) ===")
    for B in (8, 32):
        bench_sum_logprob_delta(B)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
