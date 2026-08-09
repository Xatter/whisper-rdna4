"""Flag-dispatch seam between the pure-torch M1 pipeline and the custom HIP
kernels landed by Agents B/C in kernels/*.py (DESIGN.md S5, S6, S11).

Every op named in DESIGN.md S6 (K1-K5) goes through one of the functions
below. Each function checks a small registry: if a kernel-backed
implementation has been registered for that op AND USE_KERNELS is enabled,
it dispatches there; otherwise it falls back to the torch implementation
defined in this file.

Agents B/C integrate by calling, e.g.:

    from whisper_rocm import ops

    @ops.register_kernel("layer_norm")
    def _k1_layer_norm(x, weight, bias, residual=None):
        ...  # ctypes call into libwhisper_kernels.so
        return out

No changes to model.py or decode.py are needed when a kernel lands: they
only ever call ops.<name>(...).
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Global on/off switch. DESIGN.md S5/S11: kernels land behind a flag with a
# torch fallback. Default OFF until a kernel is registered AND verified
# (Agents B/C flip this on per-op once tolerance + microbench gates pass;
# see M2 gate in DESIGN.md S10).
# ---------------------------------------------------------------------------
USE_KERNELS = os.environ.get("WHISPER_ROCM_USE_KERNELS", "0") == "1"

_registry: dict[str, Callable] = {}


def register_kernel(name: str) -> Callable:
    """Decorator: register a kernel-backed implementation for op `name`.

    Used by Agents B/C's kernels/*.py wrappers. Registering does not by
    itself enable the kernel path — WHISPER_ROCM_USE_KERNELS=1 (or setting
    whisper_rocm.ops.USE_KERNELS = True at runtime) must also be set, so a
    kernel can be registered for testing/microbench without silently
    becoming the default path.
    """

    def deco(fn: Callable) -> Callable:
        _registry[name] = fn
        return fn

    return deco


# Per-op kernel bypass, independent of the USE_KERNELS global (M5 addition):
# lets a caller keep some kernels on while forcing a specific op back onto
# the torch path -- e.g. K4 (self/cross-attn) is unaffected by timestamps
# mode, but K5 needs Agent C's SS3c extension for enable_timestamp_rules,
# which may not have landed yet. Without this, USE_KERNELS is all-or-nothing
# and a not-yet-ready K5 would force K4 off too, understating the "kernels
# on" signal for no reason.
_kernel_disabled_ops: set[str] = set()


def disable_kernel_for(name: str) -> None:
    _kernel_disabled_ops.add(name)


def enable_kernel_for(name: str) -> None:
    _kernel_disabled_ops.discard(name)


def _dispatch(name: str, torch_impl: Callable, *args, **kwargs):
    if USE_KERNELS and name in _registry and name not in _kernel_disabled_ops:
        return _registry[name](*args, **kwargs)
    return torch_impl(*args, **kwargs)


def kernel_available(name: str) -> bool:
    return name in _registry


# ---------------------------------------------------------------------------
# K1 — fused residual + LayerNorm (and plain LayerNorm).
#   x = LN(residual + y)   when residual is not None
#   x = LN(y)              when residual is None
# fp32 math on fp16 tensors, matching openai-whisper's LayerNorm semantics.
# ---------------------------------------------------------------------------


# weight/bias are static per call site (the same LayerNorm parameter
# tensors every step, DESIGN.md's decode loop never swaps them) -- casting
# them to fp32 fresh on every call was a measurable chunk of decode-step
# CPU dispatch time in an M2 torch.profiler capture (each cast is its own
# hipLaunchKernel). Cache the fp32 copy per (id(weight), id(bias)) instead
# of recomputing it ~13x/step x up to 448 steps. Keyed by id(), not a weak
# ref -- these tensors live for the whole process (WhisperWeights owns
# them), so there's no lifetime hazard, and the cache is small (~65 entries
# for the full model).
_ln_fp32_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _layer_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if residual is not None:
        x = residual + x
    key = (id(weight), id(bias))
    cached = _ln_fp32_cache.get(key)
    if cached is None:
        cached = (weight.float(), bias.float())
        _ln_fp32_cache[key] = cached
    weight_f32, bias_f32 = cached
    out = F.layer_norm(x.float(), (x.shape[-1],), weight_f32, bias_f32, eps)
    return out.to(x.dtype)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _dispatch("layer_norm", _layer_norm_torch, x, weight, bias, residual)


# ---------------------------------------------------------------------------
# K2 — GEMM with fused epilogue variants: BIAS / BIAS_GELU / BIAS_RESIDUAL.
# torch fallback: F.linear (+ optional GELU, + optional residual add).
# weight is stored [out_features, in_features] (torch.nn.Linear convention);
# weights.py keeps this layout for M1 -- kernel prepacking into K2's tiled
# layout is Agent B's concern and happens inside their registered callable.
# ---------------------------------------------------------------------------


def _gemm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    epilogue: Optional[str] = None,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = F.linear(x, weight, bias)
    if epilogue == "gelu":
        out = F.gelu(out)
    elif epilogue is not None and epilogue != "bias":
        raise ValueError(f"unknown epilogue {epilogue!r}")
    if residual is not None:
        out = out + residual
    return out


def gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue: Optional[str] = None,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _dispatch("gemm", _gemm_torch, x, weight, bias, epilogue, residual)


# ---------------------------------------------------------------------------
# K3 — encoder (bidirectional, no mask) flash attention.
# q, k, v: (B, n_head, T, head_dim). torch fallback uses SDPA (fp16 in,
# internally fp32 softmax on the backend's math path).
# ---------------------------------------------------------------------------


def _encoder_attention_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def encoder_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _dispatch("encoder_attention", _encoder_attention_torch, q, k, v)


# ---------------------------------------------------------------------------
# K4 — decode-step attention: self (causal, KV-cache append) and cross
# (fixed KV, read-only). Shapes and dtypes match Agent C's kernel contract
# exactly (kernels/DECODE_INTEGRATION.md SS1-2), not a generic attention
# signature, so the torch fallback and the registered kernel are drop-in
# for each other with no per-call adapter:
#   q, k_new, v_new, out : (B, 20, 64) fp16, no time dimension (T=1 is
#     implicit -- DESIGN.md SS7 is a strict one-token-per-step loop).
#   k_cache, v_cache     : (B, 20, T_max, 64) fp16, MUTATED IN PLACE --
#     self_attn_decode writes k_new/v_new into position `step` before
#     computing attention (causal-inclusive: attends over t in [0, step]).
#   step                 : plain python int, 0-indexed decode position.
#   out                  : optional preallocated (B, 20, 64) fp16 buffer;
#     pass the same buffer every step to avoid a caching-allocator round
#     trip per call (DECODE_INTEGRATION.md SS4).
# ---------------------------------------------------------------------------

EOT_TOKEN = 50257  # matches kernels/decode_api.py's EOT_TOKEN


def _self_attn_decode_torch(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    step: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    k_cache[:, :, step, :] = k_new
    v_cache[:, :, step, :] = v_new
    t = step + 1
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhd,bhtd->bht", q.float() * scale, k_cache[:, :, :t, :].float())
    probs = torch.softmax(scores, dim=-1)
    result = torch.einsum("bht,bhtd->bhd", probs, v_cache[:, :, :t, :].float()).to(torch.float16)
    if out is not None:
        out.copy_(result)
        return out
    return result


def self_attn_decode(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    step: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _dispatch(
        "self_attn_decode", _self_attn_decode_torch, q, k_new, v_new, k_cache, v_cache, step, out
    )


def _cross_attn_decode_torch(
    q: torch.Tensor,
    k_cache_x: torch.Tensor,
    v_cache_x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhd,bhtd->bht", q.float() * scale, k_cache_x.float())
    probs = torch.softmax(scores, dim=-1)
    result = torch.einsum("bht,bhtd->bhd", probs, v_cache_x.float()).to(torch.float16)
    if out is not None:
        out.copy_(result)
        return out
    return result


def cross_attn_decode(
    q: torch.Tensor,
    k_cache_x: torch.Tensor,
    v_cache_x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _dispatch("cross_attn_decode", _cross_attn_decode_torch, q, k_cache_x, v_cache_x, out)


# ---------------------------------------------------------------------------
# K5 — logits fusion: add suppress mask, argmax, force-eot on already
# -finished rows, update `finished` IN PLACE (int32, matching
# kernels/decode_api.py exactly), AND-reduce into `all_finished` (uint8).
# logits are fp16 (the final decoder GEMM's native output dtype); mask-add
# and argmax happen in fp32 internally either way (DECODE_INTEGRATION.md
# SS7). next_tokens/all_finished are optional preallocated buffers, reused
# every step in the hot loop.
# ---------------------------------------------------------------------------


TIMESTAMP_BEGIN = 50365  # matches whisper_rocm.tokenizer.TIMESTAMP_BEGIN
DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX = 50  # round(1.0s / (30s/1500 frames)) -- openai-whisper default


def _apply_ngram_ban_torch(
    logits: torch.Tensor, token_history: torch.Tensor, history_len: int
) -> None:
    """Mirrors HF's NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3)
    exactly (kernels/DECODE_INTEGRATION.md SS3b), including banning EOT: if
    trigram (t1, t2, EOT) already occurred, EOT is banned too, even though
    that keeps an otherwise-finished row generating. In place, matching the
    kernel's own in-place contract.
    """
    if history_len < 3:
        return
    t1 = token_history[:, history_len - 2].long()
    t2 = token_history[:, history_len - 1].long()
    hist = token_history[:, : history_len - 2].long()
    hist_next = token_history[:, 1 : history_len - 1].long()
    match = (hist == t1.unsqueeze(1)) & (hist_next == t2.unsqueeze(1))
    if not bool(match.any()):
        return
    banned_ids = token_history[:, 2:history_len].long()
    rows, cols = match.nonzero(as_tuple=True)
    logits[rows, banned_ids[rows, cols]] = float("-inf")


def _apply_timestamp_rules_torch(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    history_len: int,
    sample_begin: int,
    timestamp_begin: int = TIMESTAMP_BEGIN,
    eot_token: int = EOT_TOKEN,
    max_initial_timestamp_index: Optional[int] = DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX,
) -> None:
    """Ports openai-whisper's decoding.py:ApplyTimestampRules exactly (M5
    timestamps-mode directive). Operates on `logits` in place -- caller must
    have already added the general suppress mask (this mirrors the
    reference's own filter order: SuppressTokens runs before
    ApplyTimestampRules, and the final logsumexp-vs-max-text-logprob rule
    below specifically depends on already-suppressed tokens not skewing the
    comparison).

    no_timestamps (timestamp_begin - 1) is suppressed unconditionally.
    Timestamps must appear in pairs (except directly before EOT), must be
    non-decreasing, and each segment must have nonzero length. The first
    generated position is forced to be a timestamp, capped by
    max_initial_timestamp_index. Finally, if the summed probability mass
    over all timestamp tokens exceeds the single most likely text token,
    the step is forced to sample a timestamp.
    """
    no_timestamps_token = timestamp_begin - 1
    logits[:, no_timestamps_token] = float("-inf")

    b = logits.shape[0]
    for k in range(b):
        seq = token_history[k, sample_begin:history_len].tolist()
        last_was_ts = len(seq) >= 1 and seq[-1] >= timestamp_begin
        penultimate_was_ts = len(seq) < 2 or seq[-2] >= timestamp_begin

        if last_was_ts:
            if penultimate_was_ts:  # has to be non-timestamp
                logits[k, timestamp_begin:] = float("-inf")
            else:  # cannot be normal text tokens
                logits[k, :eot_token] = float("-inf")

        timestamps = [t for t in seq if t >= timestamp_begin]
        if timestamps:
            # timestamps shouldn't decrease; force nonzero segment length
            if last_was_ts and not penultimate_was_ts:
                timestamp_last = timestamps[-1]
            else:
                timestamp_last = timestamps[-1] + 1
            logits[k, timestamp_begin:timestamp_last] = float("-inf")

    if history_len == sample_begin:
        # suppress generating non-timestamp tokens at the very first position
        logits[:, :timestamp_begin] = float("-inf")
        if max_initial_timestamp_index is not None:
            last_allowed = timestamp_begin + max_initial_timestamp_index
            logits[:, last_allowed + 1 :] = float("-inf")

    # if summed probability over timestamps beats the single best text token, sample a timestamp
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    for k in range(b):
        timestamp_logprob = torch.logsumexp(logprobs[k, timestamp_begin:], dim=-1)
        max_text_token_logprob = logprobs[k, :timestamp_begin].max()
        if timestamp_logprob > max_text_token_logprob:
            logits[k, :timestamp_begin] = float("-inf")


def _logits_argmax_torch(
    logits: torch.Tensor,
    suppress_mask: torch.Tensor,
    finished: torch.Tensor,
    next_tokens: Optional[torch.Tensor] = None,
    all_finished: Optional[torch.Tensor] = None,
    token_history: Optional[torch.Tensor] = None,
    history_len: int = 0,
    enable_ngram_ban: bool = False,
    sample_begin: Optional[int] = None,
    enable_timestamp_rules: bool = False,
    timestamp_begin: int = TIMESTAMP_BEGIN,
    max_initial_timestamp_index: Optional[int] = DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX,
    sum_logprob: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Suppress mask goes on FIRST (matches openai-whisper's filter order:
    # SuppressTokens before ApplyTimestampRules) -- the timestamp rules'
    # logsumexp-vs-max-text rule below needs already-suppressed tokens to
    # not contribute probability mass to either side of that comparison.
    masked = logits.float() + suppress_mask

    if enable_ngram_ban and token_history is not None:
        _apply_ngram_ban_torch(masked, token_history, history_len)
    if enable_timestamp_rules and token_history is not None:
        assert sample_begin is not None, "sample_begin required when enable_timestamp_rules=True"
        _apply_timestamp_rules_torch(
            masked, token_history, history_len, sample_begin,
            timestamp_begin, EOT_TOKEN, max_initial_timestamp_index,
        )

    forced = finished.to(torch.bool)  # state BEFORE this step (openai's `tokens[:, -1] == eot`)

    # M6 (temperature-retry ladder): mirrors openai-whisper's GreedyDecoder.update
    # exactly -- temperature==0 is argmax, temperature>0 samples from
    # Categorical(logits=masked/temperature) (== softmax(masked/temperature)).
    # No kernel supports sampling (kernels/decode_api.py's logits_argmax_step
    # has no temperature arg at all), so this branch is torch-only regardless
    # of USE_KERNELS -- enforced one level up, in logits_argmax() below, by
    # never dispatching to the kernel when temperature>0.
    if temperature and temperature > 0:
        chosen = Categorical(logits=masked / temperature).sample().to(torch.int32)
    else:
        chosen = masked.argmax(dim=-1).to(torch.int32)

    if sum_logprob is not None:
        # openai's GreedyDecoder.update: current_logprobs computed on the
        # chosen token BEFORE the already-finished override below, summed
        # in only for rows that were NOT already finished going into this
        # step (DECODE_INTEGRATION.md SS3d matches this exactly).
        logprobs = torch.log_softmax(masked, dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0], device=logprobs.device), chosen.long()]
        sum_logprob += current_logprobs * (~forced).float()

    next_tok = torch.where(forced, torch.full_like(chosen, EOT_TOKEN), chosen)
    new_finished = (forced | (next_tok == EOT_TOKEN)).to(torch.int32)
    finished.copy_(new_finished)  # in-place, matches the kernel's in/out contract
    all_fin = new_finished.to(torch.bool).all().to(torch.uint8).reshape(1)

    if next_tokens is not None:
        next_tokens.copy_(next_tok)
    else:
        next_tokens = next_tok
    if all_finished is not None:
        all_finished.copy_(all_fin)
    else:
        all_finished = all_fin
    return next_tokens, all_finished


def logits_argmax(
    logits: torch.Tensor,
    suppress_mask: torch.Tensor,
    finished: torch.Tensor,
    next_tokens: Optional[torch.Tensor] = None,
    all_finished: Optional[torch.Tensor] = None,
    token_history: Optional[torch.Tensor] = None,
    history_len: int = 0,
    enable_ngram_ban: bool = False,
    sample_begin: Optional[int] = None,
    enable_timestamp_rules: bool = False,
    timestamp_begin: int = TIMESTAMP_BEGIN,
    max_initial_timestamp_index: Optional[int] = DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX,
    sum_logprob: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature and temperature > 0:
        # No kernel supports sampling -- always torch for t>0, regardless
        # of USE_KERNELS (M6 orchestrator directive: "use the torch
        # K5-fallback path ... K4 kernels stay on for retries").
        return _logits_argmax_torch(
            logits, suppress_mask, finished, next_tokens, all_finished,
            token_history, history_len, enable_ngram_ban, sample_begin, enable_timestamp_rules,
            timestamp_begin, max_initial_timestamp_index, sum_logprob, temperature,
        )
    return _dispatch(
        "logits_argmax", _logits_argmax_torch, logits, suppress_mask, finished,
        next_tokens, all_finished, token_history, history_len, enable_ngram_ban,
        sample_begin, enable_timestamp_rules, timestamp_begin, max_initial_timestamp_index,
        sum_logprob, temperature,
    )
