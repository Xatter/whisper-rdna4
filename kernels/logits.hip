// logits.hip
//
// K5 — fused final-projection epilogue chain (DESIGN.md section 6), owned
// by Agent C.
//
// Design decision (per task instructions): the (B,1280)x(1280,51866) GEMM
// itself stays in torch/hipBLASLt -- that shape is a dense, well-tiled GEMM
// that a hand kernel is unlikely to beat honestly. This file is the
// epilogue that runs AFTER torch produces logits[B, VOCAB]:
//   0a. OPTIONAL: a 3-gram-repeat ban pre-pass (mirrors HF's
//      NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3) exactly),
//      mutating specific (row, token) positions of `logits` to -inf. Off
//      by default; see launch_ngram_ban below and DECODE_INTEGRATION.md.
//   0b. OPTIONAL: OpenAI Whisper's ApplyTimestampRules (verbatim port from
//      openai-whisper's whisper/decoding.py -- see launch_timestamp_rules
//      below), also mutating `logits` in place. Runs AFTER 0a (ban is part
//      of "other masks" in openai's filter order: suppress/other masks,
//      then timestamp rules) and BEFORE stage 1's suppress-mask+argmax.
//      Off by default.
//   1. add a precomputed suppress-mask vector [VOCAB] (-inf at suppressed
//      token ids) to every row,
//   2. per-row argmax via a two-stage block reduction (stage 1: each thread
//      strides over its slice of the vocab; stage 2: LDS tree reduction
//      across the block),
//   3. force next_token = EOT_TOKEN (50257) for rows whose `finished` flag
//      is already set (so a finished sequence keeps "generating" eot
//      forever, harmless once the host-side text stitcher stops reading
//      past its first eot),
//   4. update finished[b] |= (next_token == EOT_TOKEN),
//   4b. OPTIONAL (temperature-retry ladder, Jim-approved): accumulate the
//      chosen token's logprob into a per-row running sum, fused into the
//      SAME reduction pass as steps 1-4 -- see logits_argmax_kernel_lp /
//      launch_logits_argmax_lp below and DECODE_INTEGRATION.md SS3d. Off
//      by default; a completely separate kernel from steps 1-4's, so the
//      default path (no sum_logprob passed) is byte-for-byte unchanged.
//   5. a second, tiny kernel AND-reduces finished[0..B) into one
//      all_finished byte.
//
// Everything here runs on-device; no host synchronization anywhere in this
// file. The AND-reduce is a *second* kernel launch (not an atomic into a
// value the caller must remember to reset) specifically so there is no
// required-but-easy-to-forget host-side reset step between calls -- see
// DECODE_INTEGRATION.md.
//
// Read-only contract note: steps 1-5 (logits_argmax_kernel) never modify
// `logits` -- that contract is unchanged. Step 0 (ngram_ban_kernel) is the
// one exception: when the caller enables it, it mutates specific positions
// of the SAME `logits` buffer to -inf before logits_argmax_kernel runs.
// This is safe because `logits` is a per-step scratch buffer the GEMM
// overwrites fresh every step anyway (same reasoning HF's own
// LogitsProcessor relies on when it masks a scores tensor in place). When
// the ban is disabled (the default), `logits` is truly never written, same
// as before this feature existed.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define EOT_TOKEN 50257
#define ARGMAX_BLOCK 256
#define NGRAM_BAN_WAVE 32
#define TIMESTAMP_BEGIN 50365
#define TS_RULES_BLOCK 256
// max_initial_timestamp=1.0s / precision=0.02s (CHUNK_LENGTH/n_audio_ctx) =
// round(1.0/0.02) = 50, per openai-whisper/whisper/decoding.py's
// DecodingTask.__init__. Hardcoded rather than a runtime parameter because
// this whole kernel set is built for one fixed vocab/config (large-v3-turbo,
// 0.02s timestamp steps) -- see DECODE_INTEGRATION.md SS3c for the case
// where a future caller genuinely needs a different value.
#define MAX_INITIAL_TS_INDEX 50

// ---------------------------------------------------------------------
// Optional 3-gram-repeat ban pre-pass (K5 extension, benchmark-validity
// review). Mirrors HF's NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3)
// exactly, INCLUDING its lack of any EOT/EOS exemption: if a trigram
// (t1, t2, X) already occurred earlier in a row's history, X is banned as
// this step's candidate even if X == EOT_TOKEN. That is intentional, not
// a bug to special-case -- HF's own processor does not exempt eos either,
// and matching it exactly is the whole point of this pass.
//
// One workgroup (1 wave, 32 threads) per row -- this is a trivial scan:
// len <= 448 and the vocab-write is sparse (at most len-2 scatter writes,
// almost always far fewer), so more parallelism than one wave would just
// add sync overhead for no benefit.
//
// Skips entirely (zero writes) when len < 3 (no trigram is possible yet)
// or the row is already finished (a finished row's output is forced to
// EOT downstream by logits_argmax_kernel regardless of what its logits
// say, so banning it would be wasted work with no observable effect).
//
// token_history: [B, hist_stride] int32, read-only. token_history[b,0:len)
// is row b's full generated-so-far token sequence (prompt tokens
// included, matching HF's `input_ids` convention), in generation order.
// Only indices < len are ever read.
// finished:      [B] int32, read-only here (the pre-ban snapshot -- this
//                runs before logits_argmax_kernel updates it).
// logits:        [B, vocab] fp16, MUTATED at banned (row, token_id)
//                positions (set to -inf). See the read-only contract note
//                at the top of this file.
// ---------------------------------------------------------------------
__global__ void ngram_ban_kernel(
    __half* __restrict__ logits,
    const int* __restrict__ token_history,
    const int* __restrict__ finished,
    int B, int vocab, int len, int hist_stride)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    if (finished[b]) return;
    if (len < 3) return;

    const int lane = threadIdx.x;
    const int* hist = token_history + (size_t)b * hist_stride;
    __half* row = logits + (size_t)b * vocab;

    const int t1 = hist[len - 2];
    const int t2 = hist[len - 1];
    const __half neg_inf = __float2half(-INFINITY);

    // i ranges over every position that could START an earlier (t1,t2,X)
    // trigram, i.e. i+2 < len (the current, still-being-decided trigram at
    // i = len-2 is correctly excluded: history[len] does not exist yet).
    for (int i = lane; i < len - 2; i += NGRAM_BAN_WAVE) {
        if (hist[i] == t1 && hist[i + 1] == t2) {
            int banned = hist[i + 2];
            row[banned] = neg_inf;  // idempotent if multiple i's agree
        }
    }
}

extern "C" int launch_ngram_ban(
    void* logits, const void* token_history, const void* finished,
    int B, int vocab, int len, int hist_stride, void* stream)
{
    if (len < 3) return 0;  // nothing any row could ban yet; valid no-op
    dim3 grid(B, 1, 1);
    dim3 block(NGRAM_BAN_WAVE, 1, 1);
    ngram_ban_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        (__half*)logits, (const int*)token_history, (const int*)finished,
        B, vocab, len, hist_stride);
    return (int)hipGetLastError();
}

// ---------------------------------------------------------------------
// Optional OpenAI-Whisper timestamp rules pre-pass (Jim-approved timestamps
// decode mode). This is a verbatim port of ApplyTimestampRules.apply from
// openai-whisper's whisper/decoding.py (read directly from the venv on
// gpu-host, not from memory -- see the final report for the exact source
// excerpt this was ported from). Four rules, applied in the same order as
// the reference:
//
//   1. Pair rule: timestamps must appear in pairs, except directly before
//      EOT. If the last generated token was a timestamp: if the one
//      before it was ALSO a timestamp, the next token cannot be a
//      timestamp (mask [TIMESTAMP_BEGIN, vocab)); otherwise the next
//      token cannot be ordinary text (mask [0, EOT_TOKEN) -- EOT itself
//      stays allowed).
//   2. Monotonicity: timestamps must not decrease, and every segment must
//      have nonzero length. Find the most recently generated timestamp
//      token anywhere in the row's history (not just the last position);
//      mask [TIMESTAMP_BEGIN, threshold) where threshold is that
//      timestamp's value (equal allowed -- can repeat) if rule 1's
//      "cannot be text" branch just fired, else that value + 1 (strictly
//      greater required).
//   3. First position: at the very first generated token, no ordinary
//      text is allowed (mask [0, TIMESTAMP_BEGIN)), and the initial
//      timestamp is capped at max_initial_timestamp (1.0s, index 50 at
//      this vocab's 0.02s step -- mask everything past
//      TIMESTAMP_BEGIN + MAX_INITIAL_TS_INDEX).
//   4. Probability rule: after rules 1-3 (and the suppress mask -- see
//      below) are accounted for, if the combined probability mass of ALL
//      timestamp tokens exceeds the single best text token's probability,
//      force a timestamp by masking every text token. OpenAI computes
//      this via a full log_softmax(logits) then compares
//      logsumexp(logprobs[ts_range]) against max(logprobs[text_range]).
//      This kernel computes logsumexp(RAW masked val[ts_range]) against
//      max(RAW masked val[text_range]) directly instead -- algebraically
//      the identical decision, because log_softmax subtracts the same
//      row-wide constant (logsumexp of the full row) from every element,
//      and that constant cancels in the comparison:
//        logsumexp(x - c) = logsumexp(x) - c   for any element-wise shift c
//        max(x - c)        = max(x) - c
//      so (logsumexp(x[ts]) - c) > (max(x[text]) - c)  <=>  logsumexp(x[ts]) > max(x[text]).
//      This sidesteps computing the full-row logsumexp entirely, saving a
//      pass. Verified bit-for-bit against a literal log_softmax port in
//      test_logits.py, not just argued algebraically.
//
// The "suppress mask" openai's ApplyTimestampRules relies on being already
// applied (it operates on a `logits` tensor earlier filters already wrote
// -inf into) is NOT physically written into `logits` in this codebase --
// see logits_argmax_kernel's header for why (register-only add, so the
// buffer can be reused next step without re-zeroing). Rule 4's reduction
// therefore reads `val = row[v] + suppress_mask[v]`, matching what the
// downstream argmax will see, without ever writing suppress_mask into
// `logits`. Rules 1-3 do NOT need suppress_mask at all -- they are pure
// token-identity bans, independent of the precomputed suppress list, and
// (matching openai) they write real -inf directly into `logits`.
//
// no_timestamps (50364) suppression, which openai's ApplyTimestampRules
// also does (`logits[:, no_timestamps] = -inf`), is NOT done here --
// Agent A's mode-specific suppress_mask already suppresses it (see
// DECODE_INTEGRATION.md SS3c). Doing it twice would be harmless but
// redundant; the responsibility lives with the mask by design.
//
// Interaction with the ngram ban (0a above): when both are enabled, ban
// runs first and its -inf writes are already reflected in `row[v]` by the
// time this kernel's rule-4 reduction reads it -- matching openai's filter
// order (SuppressBlank, SuppressTokens, then ApplyTimestampRules).
//
// Rules apply only to unfinished rows (`if (finished[b]) return;`) --
// finished rows keep the existing forced-EOT behavior in
// logits_argmax_kernel untouched, matching the task's explicit
// requirement. Note: openai's own rule 3 (first-position mask) is applied
// to the WHOLE batch unconditionally in the reference (a single vectorized
// op outside the per-row loop, since `tokens.shape[1] == sample_begin` is
// a batch-wide, not per-row, condition under lock-step decoding) -- but a
// finished row's output is forced to EOT downstream regardless of what its
// logits say, so skipping it here changes zero observable behavior. This
// is a safe optimization, not a semantic deviation; documented so it does
// not read as an oversight.
//
// One workgroup (TS_RULES_BLOCK=256 threads, same block size as
// logits_argmax_kernel for a matching reduction shape) per row.
// ---------------------------------------------------------------------
__global__ void timestamp_rules_kernel(
    __half* __restrict__ logits,
    const int* __restrict__ token_history,
    const int* __restrict__ finished,
    const float* __restrict__ suppress_mask,
    int B, int vocab, int hist_stride, int prompt_len, int history_len)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    if (finished[b]) return;

    const int tid = threadIdx.x;
    const int* hist = token_history + (size_t)b * hist_stride;
    __half* row = logits + (size_t)b * vocab;
    const __half neg_inf_half = __float2half(-INFINITY);

    const int seq_len = history_len - prompt_len;   // len(seq) in openai's code
    const int* seq = hist + prompt_len;              // seq[i] == sampled_tokens[i]

    // ---- Rule 1: pair rule ----
    bool last_was_ts = (seq_len >= 1) && (seq[seq_len - 1] >= TIMESTAMP_BEGIN);
    bool penult_was_ts = (seq_len < 2) || (seq[seq_len - 2] >= TIMESTAMP_BEGIN);

    if (last_was_ts) {
        if (penult_was_ts) {
            for (int v = TIMESTAMP_BEGIN + tid; v < vocab; v += TS_RULES_BLOCK) {
                row[v] = neg_inf_half;
            }
        } else {
            for (int v = tid; v < EOT_TOKEN; v += TS_RULES_BLOCK) {
                row[v] = neg_inf_half;
            }
        }
    }
    __syncthreads();

    // ---- Rule 2: monotonicity ----
    // Need the LAST (chronologically) timestamp token value anywhere in
    // seq, not just at the tail -- e.g. pattern [TS, text, text] has no
    // timestamp at the tail but still has one to enforce monotonicity
    // against. O(seq_len) <= 444, trivial next to rule 4's O(vocab) work
    // in this same kernel; a single thread does it, no point parallelizing.
    __shared__ int s_found_ts;
    __shared__ int s_last_ts;
    if (tid == 0) {
        s_found_ts = 0;
        for (int i = seq_len - 1; i >= 0; --i) {
            if (seq[i] >= TIMESTAMP_BEGIN) {
                s_found_ts = 1;
                s_last_ts = seq[i];
                break;
            }
        }
    }
    __syncthreads();

    if (s_found_ts) {
        int threshold = (last_was_ts && !penult_was_ts) ? s_last_ts : (s_last_ts + 1);
        for (int v = TIMESTAMP_BEGIN + tid; v < threshold; v += TS_RULES_BLOCK) {
            row[v] = neg_inf_half;
        }
    }
    __syncthreads();

    // ---- Rule 3: first position ----
    if (seq_len == 0) {
        for (int v = tid; v < TIMESTAMP_BEGIN; v += TS_RULES_BLOCK) {
            row[v] = neg_inf_half;
        }
        const int last_allowed = TIMESTAMP_BEGIN + MAX_INITIAL_TS_INDEX;  // inclusive
        for (int v = last_allowed + 1 + tid; v < vocab; v += TS_RULES_BLOCK) {
            row[v] = neg_inf_half;
        }
    }
    __syncthreads();

    // ---- Rule 4: logsumexp-vs-max probability rule (see file header for
    // the derivation of why raw-logit comparison matches openai's
    // log_softmax-based comparison exactly) ----
    //
    // BUG FIX (production repro from Agent A's timestamps-mode integration,
    // documented in RESULTS.md "Timestamps mode" / this file's final
    // report): when rules 1-2 have already masked a big enough chunk of
    // the LOW end of the timestamp range -- which routinely happens from
    // rule 2's monotonicity threshold once decoding is a few timestamps
    // in, not some contrived edge case -- a thread's FIRST timestamp-range
    // element in its strided walk can itself be an already-masked -inf
    // value, while ts_local_m is STILL at its "nothing seen yet" sentinel
    // (-INFINITY). The old code's `else` branch then computed
    // `expf(val - ts_local_m)` = `expf(-inf - (-inf))` = `expf(NaN)` =
    // NaN, which poisons ts_local_s for that thread permanently (NaN +
    // anything = NaN), and the block merge below propagates that NaN into
    // the row's final ts_lse. Any comparison against NaN
    // (`ts_lse > text_max`) is false in IEEE754, so rule 4 silently never
    // fired whenever this happened -- exactly the reported symptom (a
    // decisive ts-mass-wins case where EOT/text stayed unbanned). Fix:
    // skip -inf elements in the timestamp-range accumulation outright --
    // mathematically exact (a -inf logit contributes exp(-inf)=0 to the
    // sum either way), and it sidesteps the inf-inf subtraction entirely
    // rather than trying to special-case it after the fact. The max-based
    // text_local_max side never had this problem (max(-inf,-inf)=-inf is
    // well-defined, no subtraction involved), so it is unchanged.
    float text_local_max = -INFINITY;
    float ts_local_m = -INFINITY;   // online logsumexp running max (flash-attention-style)
    float ts_local_s = 0.0f;        // online logsumexp running sum, scaled by exp(-ts_local_m)
    for (int v = tid; v < vocab; v += TS_RULES_BLOCK) {
        float val = __half2float(row[v]) + suppress_mask[v];
        if (v < TIMESTAMP_BEGIN) {
            text_local_max = val > text_local_max ? val : text_local_max;
        } else if (val != -INFINITY) {
            if (val > ts_local_m) {
                ts_local_s = ts_local_s * expf(ts_local_m - val) + 1.0f;
                ts_local_m = val;
            } else {
                ts_local_s += expf(val - ts_local_m);
            }
        }
    }

    __shared__ float s_text_max[TS_RULES_BLOCK];
    __shared__ float s_ts_m[TS_RULES_BLOCK];
    __shared__ float s_ts_s[TS_RULES_BLOCK];
    s_text_max[tid] = text_local_max;
    s_ts_m[tid] = ts_local_m;
    s_ts_s[tid] = ts_local_s;
    __syncthreads();

#pragma unroll
    for (int stride = TS_RULES_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_text_max[tid + stride] > s_text_max[tid]) {
                s_text_max[tid] = s_text_max[tid + stride];
            }
            // online-logsumexp merge of two (m, s) partials, same shape as
            // a flash-attention block combine.
            float m1 = s_ts_m[tid], s1 = s_ts_s[tid];
            float m2 = s_ts_m[tid + stride], s2 = s_ts_s[tid + stride];
            float m = m1 > m2 ? m1 : m2;
            float s = 0.0f;
            if (m != -INFINITY) {
                s = s1 * expf(m1 - m) + s2 * expf(m2 - m);
            }
            s_ts_m[tid] = m;
            s_ts_s[tid] = s;
        }
        __syncthreads();
    }

    __shared__ int s_mask_text;
    if (tid == 0) {
        float text_max = s_text_max[0];
        float ts_lse = (s_ts_s[0] > 0.0f) ? (s_ts_m[0] + logf(s_ts_s[0])) : -INFINITY;
        s_mask_text = (ts_lse > text_max) ? 1 : 0;
    }
    __syncthreads();

    if (s_mask_text) {
        for (int v = tid; v < TIMESTAMP_BEGIN; v += TS_RULES_BLOCK) {
            row[v] = neg_inf_half;
        }
    }
}

extern "C" int launch_timestamp_rules(
    void* logits, const void* token_history, const void* finished,
    const void* suppress_mask,
    int B, int vocab, int hist_stride, int prompt_len, int history_len, void* stream)
{
    dim3 grid(B, 1, 1);
    dim3 block(TS_RULES_BLOCK, 1, 1);
    timestamp_rules_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        (__half*)logits, (const int*)token_history, (const int*)finished,
        (const float*)suppress_mask,
        B, vocab, hist_stride, prompt_len, history_len);
    return (int)hipGetLastError();
}

// ---------------------------------------------------------------------
// Stage-1 + stage-2 per-row argmax over logits[b, :] + suppress_mask[:],
// forced-eot handling, finished[] update, next_tokens[] write.
//
// logits:        [B, VOCAB] fp16 (read-only; NOT modified -- the mask add
//                is done in registers, so callers can reuse the same
//                logits buffer next step without re-zeroing anything).
// suppress_mask: [VOCAB] fp32, 0.0f at allowed ids, -INFINITY at
//                suppressed ids. Precomputed once, shared across the whole
//                run (same mask every decode step).
// finished:      [B] int32, in/out. 0 = still generating, 1 = done.
// next_tokens:   [B] int32, out.
// ---------------------------------------------------------------------
__global__ void logits_argmax_kernel(
    const __half* __restrict__ logits,
    const float* __restrict__ suppress_mask,
    int* __restrict__ finished,
    int* __restrict__ next_tokens,
    int B, int vocab)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;

    __shared__ float best_val[ARGMAX_BLOCK];
    __shared__ int best_idx[ARGMAX_BLOCK];

    const __half* row = logits + (size_t)b * vocab;

    // ---- Stage 1: strided per-thread scan ----
    // Tie-break toward the LOWER vocab index, matching torch.argmax's
    // documented guarantee (first/lowest index wins on exact ties). Strict
    // '>' alone is a bug here: it just keeps whichever value this thread
    // saw first in its strided walk, which is not necessarily the lower
    // original index.
    float local_best_val = -INFINITY;
    int local_best_idx = 0;
    for (int v = tid; v < vocab; v += ARGMAX_BLOCK) {
        float val = __half2float(row[v]) + suppress_mask[v];
        if (val > local_best_val || (val == local_best_val && v < local_best_idx)) {
            local_best_val = val;
            local_best_idx = v;
        }
    }
    best_val[tid] = local_best_val;
    best_idx[tid] = local_best_idx;
    __syncthreads();

    // ---- Stage 2: LDS tree reduction across the block ----
    // Same tie-break fix. Tree position does not correspond to original
    // vocab-index order, so "keep the left slot on ties" (the previous,
    // buggy version) does not imply "keep the lower index" -- a
    // higher-index winner from a deeper round can occupy a low tid and
    // then incorrectly beat a genuinely lower index later. Compare indices
    // explicitly on ties instead of relying on tree position.
#pragma unroll
    for (int stride = ARGMAX_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            bool other_wins = best_val[tid + stride] > best_val[tid] ||
                (best_val[tid + stride] == best_val[tid] && best_idx[tid + stride] < best_idx[tid]);
            if (other_wins) {
                best_val[tid] = best_val[tid + stride];
                best_idx[tid] = best_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int was_finished = finished[b];
        int token = was_finished ? EOT_TOKEN : best_idx[0];
        int now_finished = was_finished | (token == EOT_TOKEN ? 1 : 0);
        next_tokens[b] = token;
        finished[b] = now_finished;
    }
}

extern "C" int launch_logits_argmax(
    const void* logits, const void* suppress_mask,
    void* finished, void* next_tokens,
    int B, int vocab, void* stream)
{
    dim3 grid(B, 1, 1);
    dim3 block(ARGMAX_BLOCK, 1, 1);
    logits_argmax_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        (const __half*)logits, (const float*)suppress_mask,
        (int*)finished, (int*)next_tokens, B, vocab);
    return (int)hipGetLastError();
}

// ---------------------------------------------------------------------
// Optional per-step chosen-token logprob accumulation (temperature-retry
// ladder, Jim-approved). A SEPARATE kernel/entry point rather than a
// runtime branch bolted onto logits_argmax_kernel above, deliberately:
// that kernel is the one with the hard-won tie-break fix (see its header)
// and 78 passing regression cases behind it, and this feature is fully
// optional -- duplicating the ~30 lines of shared reduction shape here
// costs a little maintenance surface but means logits_argmax_kernel
// itself, and every existing caller that doesn't pass sum_logprob, is
// byte-for-byte untouched. If the tie-break logic is ever revisited,
// both copies need the same fix -- flagged here on purpose so that isn't
// missed later.
//
// Semantics: ported from openai-whisper's whisper/decoding.py
// GreedyDecoder.update (read directly from the venv on gpu-host):
//   next_tokens = logits.argmax(dim=-1)
//   logprobs = F.log_softmax(logits.float(), dim=-1)
//   current_logprobs = logprobs[arange(B), next_tokens]
//   sum_logprobs += current_logprobs * (tokens[:, -1] != eot)
// The multiply-by-"previous token wasn't eot" is exactly this codebase's
// `was_finished` flag (the PRE-step finished state already computed
// below): a row accumulates its chosen-token logprob for every step
// while it is still alive, INCLUDING the very step where it first
// produces EOT (openai's check is on the token *before* this step, not
// the one just chosen) -- and stops accumulating once was_finished is
// true, matching "avg_logprob covers tokens up to and including the
// first EOT" exactly. logprob(chosen) = logit[chosen] - logsumexp(full
// row), computed on the SAME masked-logits state logits_argmax_kernel
// sees (suppress mask added in registers, any ban/timestamp-rule -inf
// writes already physically in `logits`) -- not a separate buffer state.
//
// logsumexp is computed via the same online (running max, running sum)
// trick as timestamp_rules_kernel's Rule 4, added to the SAME stage-1
// strided scan/stage-2 tree reduction this kernel already does for
// argmax (one pass, not two -- "nearly for free" per the task). Elements
// equal to -inf (suppressed tokens, or positions an earlier ban/
// timestamp-rule pass already masked) are skipped in the logsumexp
// accumulation for exactly the reason documented at length in
// timestamp_rules_kernel's Rule 4 fix: an inf-inf subtraction produces
// NaN and silently poisons the whole block's reduction. The chosen token
// itself is guaranteed not -inf (it won the max), so it always
// contributes at least one finite term -- s_lse_s[0] > 0 is guaranteed,
// no log(0)/log(negative) risk.
//
// sum_logprob: [B] fp32, in/out. Caller (Agent A) zeroes it once at
// decode start and this kernel accumulates into it every step it's
// passed -- see DECODE_INTEGRATION.md SS3d for the full contract.
// ---------------------------------------------------------------------
__global__ void logits_argmax_kernel_lp(
    const __half* __restrict__ logits,
    const float* __restrict__ suppress_mask,
    int* __restrict__ finished,
    int* __restrict__ next_tokens,
    float* __restrict__ sum_logprob,
    int B, int vocab)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;

    __shared__ float best_val[ARGMAX_BLOCK];
    __shared__ int best_idx[ARGMAX_BLOCK];
    __shared__ float s_lse_m[ARGMAX_BLOCK];
    __shared__ float s_lse_s[ARGMAX_BLOCK];

    const __half* row = logits + (size_t)b * vocab;

    // ---- Stage 1: strided per-thread scan (argmax + online logsumexp) ----
    float local_best_val = -INFINITY;
    int local_best_idx = 0;
    float lse_m = -INFINITY;  // online logsumexp running max
    float lse_s = 0.0f;       // online logsumexp running sum, scaled by exp(-lse_m)
    for (int v = tid; v < vocab; v += ARGMAX_BLOCK) {
        float val = __half2float(row[v]) + suppress_mask[v];
        if (val > local_best_val || (val == local_best_val && v < local_best_idx)) {
            local_best_val = val;
            local_best_idx = v;
        }
        if (val != -INFINITY) {  // skip -inf: exp(-inf)=0 anyway, and this avoids
                                  // ever computing expf(-inf - -inf) = NaN below.
            if (val > lse_m) {
                lse_s = lse_s * expf(lse_m - val) + 1.0f;
                lse_m = val;
            } else {
                lse_s += expf(val - lse_m);
            }
        }
    }
    best_val[tid] = local_best_val;
    best_idx[tid] = local_best_idx;
    s_lse_m[tid] = lse_m;
    s_lse_s[tid] = lse_s;
    __syncthreads();

    // ---- Stage 2: LDS tree reduction (argmax tie-break + logsumexp merge) ----
#pragma unroll
    for (int stride = ARGMAX_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            bool other_wins = best_val[tid + stride] > best_val[tid] ||
                (best_val[tid + stride] == best_val[tid] && best_idx[tid + stride] < best_idx[tid]);
            if (other_wins) {
                best_val[tid] = best_val[tid + stride];
                best_idx[tid] = best_idx[tid + stride];
            }
            // online-logsumexp merge; guard both-empty (m==-inf) the same
            // way timestamp_rules_kernel's Rule 4 does.
            float m1 = s_lse_m[tid], s1 = s_lse_s[tid];
            float m2 = s_lse_m[tid + stride], s2 = s_lse_s[tid + stride];
            float m = m1 > m2 ? m1 : m2;
            float s = 0.0f;
            if (m != -INFINITY) {
                s = s1 * expf(m1 - m) + s2 * expf(m2 - m);
            }
            s_lse_m[tid] = m;
            s_lse_s[tid] = s;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int was_finished = finished[b];
        int token = was_finished ? EOT_TOKEN : best_idx[0];
        int now_finished = was_finished | (token == EOT_TOKEN ? 1 : 0);
        next_tokens[b] = token;
        finished[b] = now_finished;

        if (!was_finished) {
            float row_lse = s_lse_m[0] + logf(s_lse_s[0]);
            float chosen_logprob = best_val[0] - row_lse;
            sum_logprob[b] += chosen_logprob;
        }
        // was_finished rows: no accumulation at all, matching openai's
        // `current_logprobs * (tokens[:, -1] != eot)` zeroing exactly --
        // whatever a forced-EOT row's "chosen" logprob would have been is
        // irrelevant, it contributes 0 either way.
    }
}

extern "C" int launch_logits_argmax_lp(
    const void* logits, const void* suppress_mask,
    void* finished, void* next_tokens, void* sum_logprob,
    int B, int vocab, void* stream)
{
    dim3 grid(B, 1, 1);
    dim3 block(ARGMAX_BLOCK, 1, 1);
    logits_argmax_kernel_lp<<<grid, block, 0, (hipStream_t)stream>>>(
        (const __half*)logits, (const float*)suppress_mask,
        (int*)finished, (int*)next_tokens, (float*)sum_logprob, B, vocab);
    return (int)hipGetLastError();
}

// ---------------------------------------------------------------------
// AND-reduce finished[0..B) into a single byte. Single block, single
// wave is enough for B <= 64 (the batch sizes this design targets); use
// a block sized to the next wave32 multiple >= B, strided-safe for any B.
// ---------------------------------------------------------------------
__global__ void all_finished_reduce_kernel(
    const int* __restrict__ finished,
    unsigned char* __restrict__ all_finished,
    int B)
{
    const int tid = threadIdx.x;
    int local_and = 1;
    for (int b = tid; b < B; b += blockDim.x) {
        local_and &= finished[b];
    }
    // butterfly AND reduction across the block's first wave; blockDim is
    // capped at 64 (<=2 waves) by the launcher for realistic B, so fall
    // back to LDS for the (rare) cross-wave combine.
    __shared__ int partial[64];
    partial[tid] = local_and;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] &= partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *all_finished = (unsigned char)(partial[0] & 1);
    }
}

extern "C" int launch_all_finished_reduce(
    const void* finished, void* all_finished, int B, void* stream)
{
    int block = 1;
    while (block < B && block < 64) block <<= 1;
    if (block < 1) block = 1;
    dim3 grid(1, 1, 1);
    dim3 blk(block, 1, 1);
    all_finished_reduce_kernel<<<grid, blk, 0, (hipStream_t)stream>>>(
        (const int*)finished, (unsigned char*)all_finished, B);
    return (int)hipGetLastError();
}
