# DECODE_INTEGRATION.md — Agent C kernels (K4 attention_decode, K5 logits)

For Agent A's `whisper_rocm/decode.py` seam. Everything below is exposed from
`kernels/decode_api.py`. Import it as a normal Python module (it lives next
to your other kernel wrappers):

```python
import sys
sys.path.insert(0, "<repo>/kernels")   # or add kernels/ to your package path
import decode_api
```

All four kernel launches take the current torch stream
(`torch.cuda.current_stream().cuda_stream`) internally — you do not pass a
stream yourself, and there is no host synchronization inside any of these
calls. Every tensor must already be on the decode GPU (`HIP_VISIBLE_DEVICES`
pins this) and contiguous in the layout described below.

## 1. `self_attn_decode_step` — K4a

```python
out = decode_api.self_attn_decode_step(q, k_new, v_new, k_cache, v_cache, step, out=None)
```

| arg | shape | dtype | notes |
|---|---|---|---|
| `q` | `[B, 20, 64]` | fp16 | this step's query projection |
| `k_new`, `v_new` | `[B, 20, 64]` | fp16 | this step's new K/V projection |
| `k_cache`, `v_cache` | `[B, 20, T_max, 64]` | fp16 | **mutated in place** — the kernel writes `k_new`/`v_new` into cache position `step` before computing attention |
| `step` | python `int` | — | 0-indexed decode position, see SS3 |
| `out` (optional) | `[B, 20, 64]` | fp16 | preallocate and pass this in the hot loop (see SS5) |

Returns `out`, `[B, 20, 64]` fp16. `H*64 == 1280 == d_model` and the layout
is already contiguous in that order, so `out.view(B, 1280)` is a free
reshape straight into the next K2 projection — no copy.

Replaces (per call): 2 cache writes + QK^T matmul + softmax + PV matmul —
**~6 torch launches → 1**.

## 2. `cross_attn_decode_step` — K4b

```python
out = decode_api.cross_attn_decode_step(q, k_cache_x, v_cache_x, out=None)
```

| arg | shape | dtype | notes |
|---|---|---|---|
| `q` | `[B, 20, 64]` | fp16 | |
| `k_cache_x`, `v_cache_x` | `[B, 20, 1500, 64]` | fp16 | precomputed once per chunk from the encoder output (one GEMM each, per DESIGN.md SS7 step c) — **read-only** here, never mutated |
| `out` (optional) | `[B, 20, 64]` | fp16 | |

Returns `out`, `[B, 20, 64]` fp16, same reshape story as above. `T_cross` is
read from `k_cache_x.shape[2]` — if you ever chunk shorter than 1500 frames
of encoder output, pass a correspondingly shorter cache and it just works
(kernel LDS is sized for a 1500 cap, not fixed to exactly 1500).

Replaces (per call): QK^T matmul + softmax + PV matmul — **~4 torch
launches → 1**.

## 3. `logits_argmax_step` — K5

```python
next_tokens, all_finished = decode_api.logits_argmax_step(
    logits, suppress_mask, finished, next_tokens=None, all_finished=None,
    token_history=None, history_len=0, enable_ngram_ban=False)
```

| arg | shape | dtype | notes |
|---|---|---|---|
| `logits` | `[B, 51866]` | fp16 | output of the (torch/hipBLASLt) final GEMM. **Not modified** when `enable_ngram_ban=False` (the default). When `True`, banned `(row, token_id)` positions are set to `-inf` in place before the argmax runs — see SS3b. |
| `suppress_mask` | `[51866]` | fp32 | `0.0` at allowed ids, `-inf` at suppressed ids (see SS4) |
| `finished` | `[B]` | int32 | **in/out** — allocate once per chunk batch, init to `0`, this call updates it every step |
| `next_tokens` (optional) | `[B]` | int32 | preallocate for the hot loop |
| `all_finished` (optional) | `[1]` | uint8 | preallocate for the hot loop; GPU-resident |
| `token_history` (optional) | `[B, T_hist_cap]` | int32 | required when `enable_ngram_ban=True` — see SS3b |
| `history_len` (optional) | python `int` | — | required when `enable_ngram_ban=True` — see SS3b |
| `enable_ngram_ban` (optional) | python `bool` | — | default `False`. Existing calls that omit the last three args are **byte-identical in behavior and cost** to before this feature existed — the ban kernel isn't even launched. |

Returns `(next_tokens, all_finished)`. With the ban feature off, this is
**2 kernel launches** (the argmax+finished-update kernel, then a tiny
AND-reduce kernel over `finished[0..B)`); with it on and `history_len >= 3`,
**3 launches** (ban pre-pass, then the same two) — no host-side reset step
needed between calls (the AND-reduce is a fresh reduction every call, not
an atomic that requires zeroing first).

Replaces (per call, ban off): mask-add + argmax + forced-eot `where` +
finished update + all-finished reduce — **5 torch ops → 2 launches**.

`EOT_TOKEN = 50257` is exposed as `decode_api.EOT_TOKEN`. Argmax ties
break toward the **lowest** vocab index, matching `torch.argmax`'s
documented tie-break guarantee exactly (verified with constructed
multi-way ties in `test_logits.py`, including ties that touch `EOT_TOKEN`
in both directions).

### 3b. Optional 3-gram-repeat ban (benchmark-validity review)

Mirrors HF's `NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3)`
**exactly**, including the fact that HF does not exempt EOS from banning:
if a trigram `(t1, t2, EOT_TOKEN)` already occurred earlier in a row's
history, `EOT_TOKEN` is banned as this step's candidate too, even though
that means a row that would otherwise have terminated keeps generating
instead. This is intentional — it is what "match HF exactly" requires —
not a bug to guard against.

**You (Agent A) own and must maintain `token_history`:**
- Allocate `token_history` as `[B, 448]` int32 once per chunk batch (448
  matches the `T_max` you already use for `k_cache`/`v_cache` — see SS1).
  Zero-init is fine; only `token_history[:, :history_len]` is ever read.
- After every step's `next_tokens` comes back (from this same call, or
  from a previous one — order doesn't matter as long as it happens before
  the *next* call with the ban enabled), write `next_tokens` into
  `token_history[:, history_len]` and increment your `history_len`
  counter. This is the same lock-step bookkeeping you already do for
  `step` in `self_attn_decode_step` — `history_len` should track 1:1 with
  however you're counting generated positions, prompt tokens included
  (HF's processor operates over the full `input_ids`, prompt and all, and
  this kernel matches that convention). If your prompt is
  `[sot, en, transcribe, notimestamps]`, seed `token_history[:, 0:4]` with
  those 4 ids and start `history_len` at 4, not 0.
- `history_len` is a plain host `int`, same lock-step convention as `step`
  (DESIGN.md SS7: every sequence in the batch advances together, so one
  host integer describes the valid length for the whole batch — see SS5
  above for the full reasoning, it applies identically here).
- Pass `enable_ngram_ban=True` to turn it on. It is safe to pass it every
  step (the kernel itself no-ops in <1us when `history_len < 3`, and skips
  each already-finished row internally at zero cost).

**What gets banned, precisely:** for row `b`, let `t1, t2` be the last two
entries in `token_history[b, :history_len]`. For every earlier position
`i` (with `i+2 < history_len`) where `token_history[b,i] == t1` and
`token_history[b,i+1] == t2`, `token_history[b,i+2]` is banned this step
(logit forced to `-inf`, before the suppress-mask add and argmax). Rows
already `finished` are skipped entirely (their output is forced to EOT
downstream regardless of what their logits say, so banning would be
wasted work). Added cost measured in `test_logits.py`: **+2.6 to +3.4us**
per call at `history_len=100`, B in {8, 32} — does not materially change
step latency (~68us baseline at B=32).

### 3c. Optional OpenAI Whisper timestamp rules (Jim-approved timestamps decode mode)

```python
next_tokens, all_finished = decode_api.logits_argmax_step(
    logits, suppress_mask, finished,
    token_history=token_history, history_len=history_len,
    enable_timestamp_rules=True, prompt_len=prompt_len)
```

This is a **verbatim port** of `ApplyTimestampRules.apply` from
openai-whisper's `whisper/decoding.py` (read directly from the venv on
gpu-host as the reference source for this port — not written from memory).
It exists to make our decode mode drop-in-compatible with the
baseline/production transcript format (timestamps), per Jim's approval.
Off by default — omitting `enable_timestamp_rules`/`prompt_len` is
byte-identical to before this feature existed.

**This is a different, separate decode mode from SS3b's ban feature**, with
its own prompt convention: `[sot, lang, transcribe]`, **3 tokens, no
`notimestamps`** (50364) in the prompt at all — that's what makes timestamp
tokens legal output in the first place. Do not reuse a 4-token
`notimestamps`-prompt `history_len` bookkeeping scheme for this mode; seed
`token_history[:, 0:3]` with those 3 ids and start `history_len` at 3.
`token_history` itself is the **same buffer type/shape** as SS3b (`[B,
448]` int32) and can be the same physical buffer if you're only running
one mode at a time — `prompt_len` is what tells this feature where your
prompt ends and the generated sequence begins; it does not have to match
whatever prompt length SS3b's ban feature might separately assume.

**Your suppress_mask for this mode must NOT suppress the timestamp range
`[50365, 51866)` and MUST suppress `notimestamps` (50364)** — that mask
construction is your job (Agent A's), not this kernel's; the kernel just
applies whatever mask it's given, plus the four rules below, in openai's
exact order (suppress/ban first, then these rules — see SS3b's "ban runs
before timestamp rules" note, which still holds: when both features are
enabled, the ban's `-inf` writes are already reflected in `logits` by the
time these rules run).

**The four rules, applied in order (see `logits.hip`'s
`timestamp_rules_kernel` header for the full derivation/rationale, and
`test_logits.py`'s `torch_apply_timestamp_rules_reference` for a literal
Python port used as the correctness oracle):**
1. **Pair rule** — timestamps must appear in pairs, except directly before
   EOT. If the last generated token was a timestamp: and the one before it
   was too → the next token can't be a timestamp (masks the whole
   timestamp range). Otherwise → the next token can't be ordinary text
   (masks `[0, EOT_TOKEN)` — **EOT itself stays allowed**, verified
   explicitly in `test_logits.py`'s `eot-survives-text-mask` case).
2. **Monotonicity** — timestamps can't decrease and every segment needs
   nonzero length. Finds the most recent timestamp anywhere in the row's
   history (not just the last position) and bans everything below it —
   *below or equal* (strictly-greater required) unless rule 1's "can't be
   text" branch just fired, in which case *below only* (the timestamp
   itself may repeat). Both branches are separately tested.
3. **First position** — at the very first generated token (`history_len ==
   prompt_len`), no text at all is allowed, and the initial timestamp is
   capped at `max_initial_timestamp` = 1.0s = index 50 (hardcoded, not a
   parameter — see `logits.hip` if this vocab/timing ever changes):
   allowed range is `[TIMESTAMP_BEGIN, TIMESTAMP_BEGIN+50]` **inclusive**.
   `test_logits.py` checks the boundary at exactly index 50 from both
   sides (50 survives, 51 doesn't).
4. **Probability rule** — after rules 1-3 (and the suppress mask) are
   accounted for, if the combined probability mass of every timestamp
   token exceeds the single best text token's probability, force a
   timestamp by masking all text. The kernel computes this as
   `logsumexp(masked_logit[timestamp_range]) > max(masked_logit[text_range])`
   directly on the (suppress-mask-adjusted) logits, **not** via openai's
   literal `log_softmax` — these are algebraically the identical decision,
   because `log_softmax` subtracts the same row-wide constant from every
   element and that constant cancels in the comparison (full derivation in
   `logits.hip`). This is verified empirically, not just argued: the test
   reference does perform the literal `log_softmax`, and both agree.

**Rules apply only to unfinished rows.** A finished row's `logits` are
**provably untouched** by this kernel (`test_logits.py`'s
`finished-rows-untouched` case checks the actual bytes, not just the
downstream token — a broken skip would still "look" fine there since
`logits_argmax_kernel` forces EOT for finished rows regardless).

**`no_timestamps` (50364) suppression** — openai's reference also does
`logits[:, no_timestamps] = -inf` inside `ApplyTimestampRules`. This
kernel does **not** duplicate that; it's already your suppress_mask's job
(see above). Doing it twice would be harmless but redundant.

**Cost:** measured in `test_logits.py` at `seq_len=100`, B in {8,
32}: **+43.8 to +44.3us** per call — substantially more than SS3b's ban
(+3us), because rule 4 needs its own full-row reduction pass (max over
~50365 text logits, online logsumexp over ~1501 timestamp logits) on top
of rules 1-3's range-clear writes. This is honest, not optimized past
"correctness first" per the task's explicit instruction — a future pass
could look at fusing this reduction into `logits_argmax_kernel`'s own scan
if the added ~44us (against a baseline step of a few hundred us across the
full per-layer kernel chain) ever shows up as a real bottleneck in
end-to-end benchmarking. It is opt-in per chunk batch (only paid when
Jim's timestamps mode is actually selected), not paid by the default path.

### 3d. Optional per-step chosen-token logprob accumulation (temperature-retry ladder)

```python
next_tokens, all_finished = decode_api.logits_argmax_step(
    logits, suppress_mask, finished,
    sum_logprob=sum_logprob)   # combine freely with any SS3a/3b/3c args too
```

For Jim's temperature-retry ladder (OpenAI `transcribe()`-style fallback for
stalled chunks): the main greedy pass needs each sequence's average
logprob, and this computes the SUM half of that (you divide by token count
yourself, same as openai does) essentially for free, fused into the same
row-reduction pass `logits_argmax_kernel` already does for argmax — not a
second pass over the row.

**Semantics — ported from openai-whisper's `whisper/decoding.py`
`GreedyDecoder.update`** (read directly from the venv on gpu-host as the
reference, not from memory):
```python
logprobs = F.log_softmax(logits.float(), dim=-1)
current_logprobs = logprobs[arange(B), next_tokens]
sum_logprobs += current_logprobs * (tokens[:, -1] != eot)
```
i.e. `logprob(chosen) = logit[chosen] - logsumexp(full row)`, accumulated
**if and only if the row was NOT already finished BEFORE this step**
(openai's check is on the token generated by the *previous* step, not the
one just chosen this step) — this is exactly this codebase's `was_finished`
flag, already computed every call. A row's own EOT-producing step **is**
counted (openai's check is on the token before this step, so the step a
sequence first emits EOT still contributes) — "sum_logprob covers tokens
up to and including the first EOT," matching openai exactly. Once a row IS
finished, every later step contributes exactly 0, forever.

The logsumexp is computed over the SAME masked-logits state
`logits_argmax_kernel` already argmaxes over: suppress_mask added in
registers (never physically written to `logits`, per that kernel's
existing contract), plus whatever real `-inf` writes SS3b's ban and/or
SS3c's timestamp rules already made to `logits` if either ran first this
step. You don't need to do anything differently to get this right — just
pass `sum_logprob` to the same call you're already making.

**Off by default, separate kernel, zero cost when unused:** omitting
`sum_logprob` (the default, `None`) calls the exact same
`launch_logits_argmax` path as before this feature existed — byte-for-byte
unchanged code, not just "equivalent" — because this is implemented as a
SEPARATE kernel (`logits_argmax_kernel_lp` / `launch_logits_argmax_lp`),
not a runtime branch bolted onto the existing, extensively-tested argmax
kernel. Passing `sum_logprob` switches to that second kernel for the call,
which does everything the default one does plus the logprob accumulation
in one launch (never both kernels in the same step).

**Who allocates/zeroes `sum_logprob`, and when it's valid to read:**
- **Agent A allocates** `sum_logprob` as `[B]` fp32, **once per chunk
  batch**, and **zeroes it at decode start** (before the first step) --
  the kernel only ever accumulates (`+=`), it never resets on your behalf
  (same "caller zeroes, kernel never resets" pattern as `finished`).
- It is **valid to read after any step** -- it always holds the correct
  running sum of contributions so far, there is no "commit" step. For the
  temperature-retry ladder's actual use (deciding whether a completed
  chunk's average logprob is bad enough to retry at a different
  temperature), read it once decoding for that chunk batch has finished
  (all rows' `finished` are 1, or you hit the step cap) and divide by
  `len(tokens)+1` yourself, same denominator convention as openai's
  `avg_logprobs: [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]`.
  This kernel does not compute the average or know your token-count
  convention -- it only provides the sum.
- **Numerics:** fp32 throughout (both the online logsumexp accumulator and
  `sum_logprob` itself). Verified against a step-by-step torch reference
  over a chained 20-step run at rtol/atol=1e-4 in `test_logits.py`'s
  `chained-20-steps` case -- accumulation drift over a realistic run stays
  well inside that tolerance (max abs diff ~1.5e-5 to ~2.3e-5 measured at
  B=8/32).
- **-inf handling:** the online logsumexp accumulation skips `-inf`
  elements exactly the way SS3c's Rule 4 fix does (see that section and
  `logits.hip`'s comments) -- suppressed/banned/rule-masked tokens
  correctly contribute zero, and the chosen token itself is guaranteed
  never `-inf` (it won the argmax), so there is no log(0) risk.

**Cost:** measured in `test_logits.py` at B in {8, 32}: **+10.2 to
+10.3us** per call -- genuinely "nearly for free" relative to SS3c's
timestamp rules (+44-49us, a real second full-row pass) precisely because
this reuses the argmax kernel's existing single pass instead of adding
another one.

## 4. Who allocates what

- **Agent A allocates and owns the lifetime of**: `k_cache`/`v_cache` (self,
  per chunk batch, `[B,20,448,64]` fp16, zero-init not required — the
  kernel only ever reads positions `< step+1`, so uninitialized future
  positions are never touched), `k_cache_x`/`v_cache_x` (cross, produced by
  the encoder-output GEMM once per chunk), `finished` (`[B]` int32, init to
  0 at the start of each chunk batch, one per batch not per step), and
  `suppress_mask` (`[51866]` fp32, build **once** for the whole run, reused
  every step and every chunk — it never changes).
- **`decode_api` allocates transient outputs on demand** (`out`,
  `next_tokens`, `all_finished`) if you don't pass them in. For
  correctness/one-off calls that's fine; for the actual decode loop,
  **preallocate all three once per chunk batch and pass them in every
  step** via the optional args — this avoids a fresh caching-allocator
  round trip 40x/step x up to 448 steps and keeps every buffer's address
  stable, which also makes step 6 below (HIP graph capture) possible later.

## 5. Step-counter convention (why a host `int`, not a device tensor)

`step` is a plain Python `int`, not a device-side length tensor. This is
safe specifically because DESIGN.md SS7's decode loop is **lock-step**: every
sequence in a batch advances exactly one cache position per host-loop
iteration, with no per-sequence branching on length (finished sequences
stay in the batch and just get forced-eot outputs from K5, they are not
dropped or resized out). Under that invariant a single host integer names
the current write/attend position for *every* sequence simultaneously, so
it is exact, not an approximation — and it avoids a host<->device
round trip that a real per-sequence device-side length would otherwise
force the first time the host needed to inspect it. `step=0` is the first
generated token; the kernel appends at `step` and attends over
`t in [0, step]` inclusive. **Your host loop just does `step += 1` each
iteration** (or track it however is natural) and passes the current value
in — no GPU read-back involved.

If a future revision ever needs per-sequence early-exit (e.g. to shrink the
batch instead of masking), that's a bigger change than this contract
supports today and should go through a review, not a silent reinterpretation
of `step`.

## 6. Polling `all_finished` without a per-step stall

DESIGN.md SS7 specifies checking a pinned-memory copy of `all_finished`
every 8 steps, lagged/async, instead of syncing every step. `decode_api`
provides the two halves of that pattern:

```python
pinned = decode_api.make_pinned_flag()   # once, at chunk-batch start

for step in range(max_steps):
    ...  # K1/K2/K4/K5 launches for this step, as in DESIGN.md SS7's per-layer chain
    next_tokens, all_finished = decode_api.logits_argmax_step(
        logits, suppress_mask, finished, next_tokens_buf, all_finished_buf)

    if step % 8 == 0:
        decode_api.poll_all_finished_async(all_finished, pinned)  # non-blocking D2H

    if step % 8 == 7:
        # by now the async copy queued 7 steps ago has almost certainly
        # landed on its own; a sync here is cheap (1 byte, work already done)
        torch.cuda.current_stream().synchronize()
        if bool(pinned.item()):
            break
```

The copy-then-check is deliberately split across two different steps (queue
at `step % 8 == 0`, only look at `step % 8 == 7`) so the D2H copy has slack
to complete on its own before you ever call `.item()` on it — the
`synchronize()` right before `.item()` is what DESIGN.md accepts as the
"lagged" cost (1/8th as often as a naive per-step check), not a new
per-step stall. If you want a truly zero-stall check, replace the
`synchronize()` with a `torch.cuda.Event` recorded right after the async
copy and `event.query()` instead — `decode_api` doesn't wrap that because
it's a host-loop-structure choice, not a kernel concern.

## 7. Numeric contract

Every accumulation inside both K4 kernels (Q.K^T dot products, softmax
max/sum, P.V accumulate) runs in fp32; only the cache read/append and the
final output write touch fp16. K5's mask-add and argmax comparison are also
fp32. This matters across the full ~448-step decode loop, not just per
step — see the cheatsheet's stateful-kernel warning (SS13.3): a fused
kernel with a different accumulation order than the eventual torch
reference is exactly the kind of thing that looks fine on step 1 and has
visibly drifted by step 400. `test_attention_decode.py`/`test_logits.py`
check this at `step=447` (full 448-length cache) specifically for that
reason, not just at `step=0`.

## 8. Known limitation / stretch goal not attempted

DESIGN.md's K5 description also floats fusing the suppress+argmax epilogue
directly into a custom split-K GEMV over the tied embedding matrix (i.e.
replacing the torch `(B,1280)x(1280,51866)` GEMM itself, not just its
epilogue). This was not attempted — the epilogue-only design above already
beats the torch op sequence honestly at the tested shapes (see the final
report for numbers), and DESIGN.md's own guidance is that hipBLASLt is hard
to beat at that GEMM shape. If a future pass wants to try it, budget real
time for it: it needs its own correctness/perf gate independent of this
epilogue, per DESIGN.md SS6's "lands only if it wins" rule.
