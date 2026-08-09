"""Registers Agent C's K4/K5 kernels (kernels/decode_api.py) into
whisper_rocm.ops's seam. Read-only import of kernels/ -- never edits it
(that directory is Agent B/C's, per the project's file-ownership rules).

Usage: call register() once, early (e.g. right after argparse in
bench_e2e.py / check_stages.py), then set whisper_rocm.ops.USE_KERNELS =
True (or WHISPER_ROCM_USE_KERNELS=1 in the environment) to actually route
through them. Registering without enabling the flag is inert -- useful for
tests that want to call the kernel path explicitly without flipping the
global default.

Only K4 (self_attn_decode, cross_attn_decode) and K5 (logits_argmax) are
registered here. Agent B's K1/K2/K3 are deliberately NOT wired in: the
orchestrator's M2 review found they lose to torch at the tested shapes
(K2 0.5-0.7x, K3 0.2x), so the encoder and the decode-loop GEMMs/LayerNorms
stay pure torch. gemm_pack.py's tile-prepacking is skipped entirely for
the same reason -- there is currently no kernel that would consume it.
"""

from __future__ import annotations

import inspect
import os
import sys

from . import ops

_KERNELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels"
)

_registered = False


def _import_decode_api():
    if _KERNELS_DIR not in sys.path:
        sys.path.insert(0, _KERNELS_DIR)
    import decode_api  # type: ignore  # noqa: E402

    return decode_api


def register() -> None:
    """Idempotent. Safe to call multiple times / from multiple entry points."""
    global _registered
    if _registered:
        return

    decode_api = _import_decode_api()
    assert decode_api.EOT_TOKEN == ops.EOT_TOKEN, (
        f"EOT token mismatch: decode_api={decode_api.EOT_TOKEN} ops={ops.EOT_TOKEN}"
    )

    # DECODE_INTEGRATION.md SS3c (timestamps mode, Agent C) had not landed as
    # of this registration -- probe decode_api's actual signature rather than
    # assuming. If a caller asks for enable_timestamp_rules=True while the
    # installed decode_api doesn't support it yet, fail loudly (below) rather
    # than silently dropping the flag and computing a wrong (non-timestamp)
    # answer while claiming kernels are on.
    _k5_sig = inspect.signature(decode_api.logits_argmax_step)
    _k5_supports_timestamps = "enable_timestamp_rules" in _k5_sig.parameters
    # DECODE_INTEGRATION.md SS3d (temperature-retry ladder, M6): same
    # probe-don't-assume pattern as SS3c above.
    _k5_supports_sum_logprob = "sum_logprob" in _k5_sig.parameters

    @ops.register_kernel("self_attn_decode")
    def _k4_self_attn_decode(q, k_new, v_new, k_cache, v_cache, step, out=None):
        return decode_api.self_attn_decode_step(q, k_new, v_new, k_cache, v_cache, step, out=out)

    @ops.register_kernel("cross_attn_decode")
    def _k4_cross_attn_decode(q, k_cache_x, v_cache_x, out=None):
        return decode_api.cross_attn_decode_step(q, k_cache_x, v_cache_x, out=out)

    @ops.register_kernel("logits_argmax")
    def _k5_logits_argmax(
        logits, suppress_mask, finished, next_tokens=None, all_finished=None,
        token_history=None, history_len=0, enable_ngram_ban=False,
        sample_begin=None, enable_timestamp_rules=False,
        timestamp_begin=ops.TIMESTAMP_BEGIN,
        max_initial_timestamp_index=ops.DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX,
        sum_logprob=None, temperature=0.0,
    ):
        # temperature>0 never reaches here -- ops.logits_argmax() itself
        # short-circuits to the torch path before dispatch, since no kernel
        # supports sampling. Assert rather than silently ignore, in case
        # that invariant ever breaks.
        assert not (temperature and temperature > 0), (
            "kernel_bridge's K5 wrapper received temperature>0; ops.logits_argmax "
            "should have routed this to torch before reaching the kernel dispatch"
        )
        if enable_timestamp_rules and not _k5_supports_timestamps:
            raise RuntimeError(
                "enable_timestamp_rules=True requested with kernels on, but the "
                "installed kernels/decode_api.py has no enable_timestamp_rules "
                "parameter yet (DECODE_INTEGRATION.md SS3c not landed). Run with "
                "USE_KERNELS off for the torch-fallback timestamp path, or "
                "re-register after Agent C's kernel update lands."
            )
        if sum_logprob is not None and not _k5_supports_sum_logprob:
            raise RuntimeError(
                "sum_logprob requested with kernels on, but the installed "
                "kernels/decode_api.py has no sum_logprob parameter yet "
                "(DECODE_INTEGRATION.md SS3d not landed)."
            )
        if enable_timestamp_rules:
            # decode_api's TIMESTAMP_BEGIN/MAX_INITIAL_TS_INDEX are
            # compile-time constants (kernels/logits.hip), not runtime
            # args -- verified equal to ops.py's own defaults (50365, 50)
            # by reading the .hip source directly, not assumed. Any
            # non-default value passed here would silently be ignored by
            # the kernel, so fail loudly instead of pretending to honor it.
            if timestamp_begin != ops.TIMESTAMP_BEGIN or max_initial_timestamp_index != ops.DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX:
                raise RuntimeError(
                    f"kernel's timestamp_begin/max_initial_timestamp_index are compile-time "
                    f"constants (50365/50, kernels/logits.hip) and can't honor the non-default "
                    f"values requested here (timestamp_begin={timestamp_begin}, "
                    f"max_initial_timestamp_index={max_initial_timestamp_index})."
                )
            # decode_api's parameter is named `prompt_len`, ops.py's is
            # `sample_begin` (matches openai-whisper's own DecodingTask
            # naming, which is what this whole feature ports) -- same
            # value, translated at the boundary rather than renaming
            # ops.py's public API to match the kernel's internal name.
            return decode_api.logits_argmax_step(
                logits, suppress_mask, finished, next_tokens=next_tokens, all_finished=all_finished,
                token_history=token_history, history_len=history_len, enable_ngram_ban=enable_ngram_ban,
                enable_timestamp_rules=enable_timestamp_rules, prompt_len=sample_begin,
                sum_logprob=sum_logprob,
            )
        return decode_api.logits_argmax_step(
            logits, suppress_mask, finished, next_tokens=next_tokens, all_finished=all_finished,
            token_history=token_history, history_len=history_len, enable_ngram_ban=enable_ngram_ban,
            sum_logprob=sum_logprob,
        )

    _registered = True


def k5_supports_timestamps() -> bool:
    """True once Agent C's DECODE_INTEGRATION.md SS3c kernel extension is
    installed and registered. Callers can use this to decide whether a
    kernels-on timestamps run is possible yet, instead of hitting the
    RuntimeError above."""
    if not _registered:
        return False
    decode_api = _import_decode_api()
    return "enable_timestamp_rules" in inspect.signature(decode_api.logits_argmax_step).parameters


def is_registered() -> bool:
    return _registered
