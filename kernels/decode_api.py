"""decode_api.py -- Python/ctypes wrappers for Agent C's decode kernels
(attention_decode.hip: K4 self-attn + cross-attn decode steps; logits.hip:
K5 suppress+argmax+finished epilogue).

Owned by Agent C. See DECODE_INTEGRATION.md in this directory for the full
integration contract (who allocates what, tensor layouts, the step-counter
convention, and the pinned-memory all_finished polling pattern).

Library loading: this module first looks for the combined
`libwhisper_kernels.so` that kernels/build.sh (Agent B) produces from every
kernel file. If that doesn't exist yet (e.g. running this module's own
tests before build.sh lands, or before Agent B's kernels are ready), it
falls back to compiling a private `.libdecode_kernels.so` from just this
file's two owned sources (attention_decode.hip, logits.hip) so K4/K5 can be
developed, tested, and even integrated against standalone. Both .so files
export the same symbol names, so callers of this module never need to know
which one is loaded.
"""

import ctypes
import os
import subprocess
import threading

import torch

# ---------------------------------------------------------------------
# Constants shared with the kernels (must match attention_decode.hip /
# logits.hip -- these are compiled-in, not passed at runtime).
# ---------------------------------------------------------------------
HEAD_DIM = 64
EOT_TOKEN = 50257
SELF_T_CAP = 448      # self-attn cache capacity (whisper large-v3-turbo)
CROSS_T_CAP = 1500    # cross-attn encoder sequence length

_KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))
_HIPCC = "/opt/rocm/bin/hipcc"
_OFFLOAD_ARCH = "gfx1201"
_COMBINED_SO = os.path.join(_KERNELS_DIR, "libwhisper_kernels.so")
_PRIVATE_SO = os.path.join(_KERNELS_DIR, ".libdecode_kernels.so")
_OWN_SOURCES = [
    os.path.join(_KERNELS_DIR, "attention_decode.hip"),
    os.path.join(_KERNELS_DIR, "logits.hip"),
]

_lib_lock = threading.Lock()
_lib = None


def _needs_rebuild(so_path: str, sources) -> bool:
    if not os.path.exists(so_path):
        return True
    so_mtime = os.path.getmtime(so_path)
    return any(os.path.getmtime(s) > so_mtime for s in sources)


def _compile_private_so() -> str:
    if _needs_rebuild(_PRIVATE_SO, _OWN_SOURCES):
        cmd = [
            _HIPCC, "-fPIC", "-shared",
            f"--offload-arch={_OFFLOAD_ARCH}", "-O2",
            "-mno-wavefrontsize64",  # wave32 everywhere, see cheatsheet SS4
            "-o", _PRIVATE_SO,
        ] + _OWN_SOURCES
        subprocess.run(cmd, check=True)
    return _PRIVATE_SO


def _load_lib():
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        # Prefer the combined build.sh output, BUT only if it is not older
        # than our own two owned sources -- kernels/ is a shared directory
        # (Agent B's build.sh assembles libwhisper_kernels.so from every
        # kernel file) and a stale combined .so built before a fix here
        # landed would silently shadow that fix with old code every time
        # this module runs. This was caught in practice: a tie-break fix
        # to logits.hip tested correct in isolation but test_logits.py
        # still failed, because the combined .so on disk predated the fix.
        # This check does not touch build.sh or any other agent's file --
        # once build.sh reruns after this file changes, the combined .so
        # is newer again and gets preferred automatically, same as before.
        use_combined = os.path.exists(_COMBINED_SO) and not _needs_rebuild(_COMBINED_SO, _OWN_SOURCES)
        so_path = _COMBINED_SO if use_combined else _compile_private_so()
        lib = ctypes.CDLL(so_path)

        lib.launch_self_attn_decode.restype = ctypes.c_int
        lib.launch_self_attn_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q, k_new, v_new
            ctypes.c_void_p, ctypes.c_void_p,                    # k_cache, v_cache
            ctypes.c_void_p,                                     # out
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B,H,step,T_max
            ctypes.c_void_p,                                     # stream
        ]

        lib.launch_cross_attn_decode.restype = ctypes.c_int
        lib.launch_cross_attn_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q, k_cache_x, v_cache_x
            ctypes.c_void_p,                                     # out
            ctypes.c_int, ctypes.c_int, ctypes.c_int,            # B,H,T_cross
            ctypes.c_void_p,                                     # stream
        ]

        lib.launch_logits_argmax.restype = ctypes.c_int
        lib.launch_logits_argmax.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,                    # logits, suppress_mask
            ctypes.c_void_p, ctypes.c_void_p,                    # finished, next_tokens
            ctypes.c_int, ctypes.c_int,                          # B, vocab
            ctypes.c_void_p,                                     # stream
        ]

        lib.launch_logits_argmax_lp.restype = ctypes.c_int
        lib.launch_logits_argmax_lp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,                    # logits, suppress_mask
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # finished, next_tokens, sum_logprob
            ctypes.c_int, ctypes.c_int,                          # B, vocab
            ctypes.c_void_p,                                     # stream
        ]

        lib.launch_ngram_ban.restype = ctypes.c_int
        lib.launch_ngram_ban.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # logits, token_history, finished
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B, vocab, len, hist_stride
            ctypes.c_void_p,                                     # stream
        ]

        lib.launch_timestamp_rules.restype = ctypes.c_int
        lib.launch_timestamp_rules.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # logits, token_history, finished
            ctypes.c_void_p,                                     # suppress_mask
            ctypes.c_int, ctypes.c_int, ctypes.c_int,            # B, vocab, hist_stride
            ctypes.c_int, ctypes.c_int,                          # prompt_len, history_len
            ctypes.c_void_p,                                     # stream
        ]

        lib.launch_all_finished_reduce.restype = ctypes.c_int
        lib.launch_all_finished_reduce.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ]

        _lib = lib
        return _lib


def _stream_ptr() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(err: int, name: str) -> None:
    if err != 0:
        raise RuntimeError(f"{name} failed with hip error code {err}")


# ---------------------------------------------------------------------
# K4a: self-attention decode step (append + attend).
# ---------------------------------------------------------------------
def self_attn_decode_step(q, k_new, v_new, k_cache, v_cache, step, out=None):
    """One fused self-attention decode step.

    q, k_new, v_new : [B, H, 64] fp16 CUDA tensors, contiguous.
    k_cache, v_cache: [B, H, T_max, 64] fp16 CUDA tensors, contiguous.
                       Mutated IN PLACE: the new k_new/v_new row is written
                       to cache position `step` before the attention math
                       runs, so the just-appended token is included in the
                       softmax (causal, inclusive).
    step            : int, 0-indexed decode position. T_max must be > step
                       (T_max is read from k_cache.shape[2], not passed
                       separately, so caller cannot desync cache stride
                       from cache capacity).
    out             : optional preallocated [B, H, 64] fp16 tensor. If
                       omitted, a fresh tensor is allocated each call --
                       fine for tests, but the hot decode loop should
                       preallocate and pass this in (see
                       DECODE_INTEGRATION.md).

    Returns out, shape [B, H, 64] fp16. Caller reshapes to [B, H*64] for
    the following K2 projection (this is a no-copy view since H*64 is the
    innermost contiguous layout already).
    """
    assert q.dtype == torch.float16 and k_new.dtype == torch.float16 and v_new.dtype == torch.float16
    assert k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16
    assert q.is_cuda and q.is_contiguous()
    assert k_new.is_contiguous() and v_new.is_contiguous()
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    B, H, Dh = q.shape
    assert Dh == HEAD_DIM
    assert k_cache.shape[:2] == (B, H) and k_cache.shape[3] == HEAD_DIM
    T_max = k_cache.shape[2]
    assert 0 <= step < T_max, f"step {step} out of range for T_max {T_max}"

    if out is None:
        out = torch.empty((B, H, Dh), dtype=torch.float16, device=q.device)

    lib = _load_lib()
    err = lib.launch_self_attn_decode(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k_new.data_ptr()),
        ctypes.c_void_p(v_new.data_ptr()),
        ctypes.c_void_p(k_cache.data_ptr()),
        ctypes.c_void_p(v_cache.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(step), ctypes.c_int(T_max),
        _stream_ptr(),
    )
    _check(err, "launch_self_attn_decode")
    return out


# ---------------------------------------------------------------------
# K4b: cross-attention decode step (no append, fixed encoder length).
# ---------------------------------------------------------------------
def cross_attn_decode_step(q, k_cache_x, v_cache_x, out=None):
    """One fused cross-attention decode step against the precomputed
    encoder-side K/V cache (computed once per chunk, read-only here).

    q               : [B, H, 64] fp16 CUDA tensor, contiguous.
    k_cache_x, v_cache_x: [B, H, T_cross, 64] fp16 CUDA tensors,
                       contiguous, read-only (T_cross <= 1500).
    out             : optional preallocated [B, H, 64] fp16 tensor.

    Returns out, shape [B, H, 64] fp16.
    """
    assert q.dtype == torch.float16 and q.is_cuda and q.is_contiguous()
    assert k_cache_x.dtype == torch.float16 and v_cache_x.dtype == torch.float16
    assert k_cache_x.is_contiguous() and v_cache_x.is_contiguous()
    B, H, Dh = q.shape
    assert Dh == HEAD_DIM
    assert k_cache_x.shape[:2] == (B, H) and k_cache_x.shape[3] == HEAD_DIM
    T_cross = k_cache_x.shape[2]
    assert 0 < T_cross <= CROSS_T_CAP

    if out is None:
        out = torch.empty((B, H, Dh), dtype=torch.float16, device=q.device)

    lib = _load_lib()
    err = lib.launch_cross_attn_decode(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k_cache_x.data_ptr()),
        ctypes.c_void_p(v_cache_x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(T_cross),
        _stream_ptr(),
    )
    _check(err, "launch_cross_attn_decode")
    return out


# ---------------------------------------------------------------------
# K5: suppress-mask + argmax + finished-flag + all-finished epilogue.
# ---------------------------------------------------------------------
def logits_argmax_step(logits, suppress_mask, finished, next_tokens=None, all_finished=None,
                        token_history=None, history_len=0, enable_ngram_ban=False,
                        enable_timestamp_rules=False, prompt_len=0,
                        sum_logprob=None):
    """Fused logits epilogue: add suppress mask, per-row argmax, force
    EOT for already-finished rows, update `finished` in place, and
    AND-reduce into a single `all_finished` byte.

    logits        : [B, vocab] fp16 CUDA tensor. Read-only UNLESS
                    enable_ngram_ban=True or enable_timestamp_rules=True, in
                    which case specific (row, token) positions get mutated
                    to -inf in place before the argmax runs (see the
                    paragraphs below). With both flags left at their
                    defaults (False), this is unchanged from before: never
                    modified.
    suppress_mask : [vocab] fp32 CUDA tensor, 0.0 at allowed ids,
                    -inf at suppressed ids. Precompute once, reuse every
                    step and every chunk.
    finished      : [B] int32 CUDA tensor, in/out. 0/1. Caller allocates
                    once per chunk batch, initialized to 0, and this
                    function updates it in place every step.
    next_tokens   : optional preallocated [B] int32 CUDA tensor.
    all_finished  : optional preallocated [1] uint8 CUDA tensor.

    Optional 3-gram-repeat ban (benchmark-validity review: matches HF's
    NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3) exactly, including
    banning EOT itself when a repeated trigram happens to complete with
    it -- HF does not exempt eos either). OFF BY DEFAULT: existing callers
    that omit the three args below get byte-identical behavior and do not
    pay for an extra kernel launch -- the ban kernel is not even invoked.
      token_history    : [B, T_hist_cap] int32 CUDA tensor, required when
                          enable_ngram_ban=True OR enable_timestamp_rules=True
                          (shared buffer -- see SS3c). token_history[b,0:history_len]
                          is row b's full generated-so-far token sequence
                          (prompt tokens included), in generation order.
                          Agent A allocates/owns this buffer and is
                          responsible for appending each step's
                          next_tokens[b] into it -- see DECODE_INTEGRATION.md.
      history_len       : int, host value. Same lock-step convention as
                          `step` in self_attn_decode_step: one value for
                          the whole batch, no per-row divergence. Number of
                          valid entries in token_history[:, :history_len].
      enable_ngram_ban  : bool, default False.

    Optional OpenAI Whisper timestamp rules (Jim-approved timestamps decode
    mode; verbatim port of ApplyTimestampRules from openai-whisper's
    whisper/decoding.py -- see DECODE_INTEGRATION.md SS3c for the full
    semantics). Runs AFTER the ngram ban (if both enabled) and BEFORE the
    suppress-mask+argmax stage below, matching openai's own filter order.
    OFF BY DEFAULT: existing callers see zero behavior/cost change.
      enable_timestamp_rules : bool, default False.
      prompt_len              : int, host value, lock-step convention like
                          `history_len`. The number of prompt tokens at the
                          front of `token_history` (3 in this mode:
                          [sot, lang, transcribe], no notimestamps token --
                          NOT the same value as the 4-token prompt some
                          other mode might use for ngram-ban bookkeeping;
                          this arg is independent of enable_ngram_ban).
                          Required (and must satisfy 0 <= prompt_len <=
                          history_len) when enable_timestamp_rules=True.

    Optional per-step chosen-token logprob accumulation (temperature-retry
    ladder). OFF BY DEFAULT: omitting sum_logprob calls the exact same
    launch_logits_argmax path as before this feature existed -- byte-for-
    byte unchanged, not just "equivalent". Passing it switches to a
    separate kernel (launch_logits_argmax_lp) that does everything the
    default one does PLUS the logprob accumulation in the SAME pass (one
    launch either way, not two) -- see DECODE_INTEGRATION.md SS3d for the
    full contract (who zeroes it, exact accumulation semantics ported from
    openai-whisper's GreedyDecoder.update).
      sum_logprob : optional [B] fp32 CUDA tensor, in/out. Accumulated in
                    place every step it's passed. Caller zeroes it once at
                    decode start.

    Returns (next_tokens, all_finished). all_finished is a 1-element uint8
    GPU tensor (1 iff every sequence in the batch has hit EOT) -- see
    DECODE_INTEGRATION.md for the recommended pinned-memory polling
    pattern to check it without a per-step host sync.
    """
    assert logits.dtype == torch.float16 and logits.is_cuda and logits.is_contiguous()
    assert suppress_mask.dtype == torch.float32 and suppress_mask.is_contiguous()
    assert finished.dtype == torch.int32 and finished.is_contiguous()
    B, vocab = logits.shape
    assert suppress_mask.shape == (vocab,)
    assert finished.shape == (B,)

    if next_tokens is None:
        next_tokens = torch.empty(B, dtype=torch.int32, device=logits.device)
    if all_finished is None:
        all_finished = torch.empty(1, dtype=torch.uint8, device=logits.device)

    lib = _load_lib()
    stream = _stream_ptr()

    if enable_ngram_ban and history_len >= 3:
        assert token_history is not None, "token_history required when enable_ngram_ban=True"
        assert token_history.dtype == torch.int32 and token_history.is_cuda
        assert token_history.is_contiguous()
        assert token_history.shape[0] == B
        hist_stride = token_history.shape[1]
        assert 0 <= history_len <= hist_stride, \
            f"history_len {history_len} exceeds token_history capacity {hist_stride}"
        err = lib.launch_ngram_ban(
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(token_history.data_ptr()),
            ctypes.c_void_p(finished.data_ptr()),
            ctypes.c_int(B), ctypes.c_int(vocab), ctypes.c_int(history_len), ctypes.c_int(hist_stride),
            stream,
        )
        _check(err, "launch_ngram_ban")

    if enable_timestamp_rules:
        assert token_history is not None, "token_history required when enable_timestamp_rules=True"
        assert token_history.dtype == torch.int32 and token_history.is_cuda
        assert token_history.is_contiguous()
        assert token_history.shape[0] == B
        hist_stride = token_history.shape[1]
        assert 0 <= prompt_len <= history_len <= hist_stride, \
            f"need 0 <= prompt_len({prompt_len}) <= history_len({history_len}) <= capacity({hist_stride})"
        err = lib.launch_timestamp_rules(
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(token_history.data_ptr()),
            ctypes.c_void_p(finished.data_ptr()),
            ctypes.c_void_p(suppress_mask.data_ptr()),
            ctypes.c_int(B), ctypes.c_int(vocab), ctypes.c_int(hist_stride),
            ctypes.c_int(prompt_len), ctypes.c_int(history_len),
            stream,
        )
        _check(err, "launch_timestamp_rules")

    if sum_logprob is not None:
        assert sum_logprob.dtype == torch.float32 and sum_logprob.is_cuda
        assert sum_logprob.is_contiguous()
        assert sum_logprob.shape == (B,)
        err = lib.launch_logits_argmax_lp(
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(suppress_mask.data_ptr()),
            ctypes.c_void_p(finished.data_ptr()),
            ctypes.c_void_p(next_tokens.data_ptr()),
            ctypes.c_void_p(sum_logprob.data_ptr()),
            ctypes.c_int(B), ctypes.c_int(vocab),
            stream,
        )
        _check(err, "launch_logits_argmax_lp")
    else:
        err = lib.launch_logits_argmax(
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(suppress_mask.data_ptr()),
            ctypes.c_void_p(finished.data_ptr()),
            ctypes.c_void_p(next_tokens.data_ptr()),
            ctypes.c_int(B), ctypes.c_int(vocab),
            stream,
        )
        _check(err, "launch_logits_argmax")

    err = lib.launch_all_finished_reduce(
        ctypes.c_void_p(finished.data_ptr()),
        ctypes.c_void_p(all_finished.data_ptr()),
        ctypes.c_int(B),
        stream,
    )
    _check(err, "launch_all_finished_reduce")
    return next_tokens, all_finished


# ---------------------------------------------------------------------
# Pinned-memory polling helper for all_finished (see
# DECODE_INTEGRATION.md "every 8 steps" pattern).
# ---------------------------------------------------------------------
def make_pinned_flag() -> torch.Tensor:
    """Allocate the host-side pinned byte used to poll all_finished
    without a blocking per-step sync."""
    return torch.zeros(1, dtype=torch.uint8, pin_memory=True)


def poll_all_finished_async(all_finished_gpu: torch.Tensor, pinned_host: torch.Tensor) -> None:
    """Issue a non-blocking D2H copy of the 1-byte all_finished flag into
    pinned host memory. Does not sync. Caller reads pinned_host.item()
    only after a later sync point (see integration doc)."""
    pinned_host.copy_(all_finished_gpu, non_blocking=True)
