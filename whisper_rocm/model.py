"""Encoder + decoder host orchestration (DESIGN.md S3, S5, S7).

Every GEMM/LN/attention op is routed through whisper_rocm.ops so Agents
B/C can register kernel-backed replacements with zero changes here (see
ops.py's module docstring). This file only does shape bookkeeping, KV
cache indexing, and the encoder/decoder block structure.

M2 update: the decode-step path (self/cross attention) now matches Agent
C's landed K4 kernel contract exactly -- (B, 20, 64) tensors with no time
dimension, an explicit `step` int, and preallocated `out` buffers reused
across every layer and every step (kernels/DECODE_INTEGRATION.md SS4). The
encoder stays on the (B, T, D) multi-token path; K1/K2/K3 (Agent B) are
not wired in -- they lost to torch at the tested shapes, per the M2 review.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import ops
from .weights import DecoderLayerWeights, EncoderLayerWeights, WhisperWeights


# ---------------------------------------------------------------------------
# Encoder (multi-token, unchanged from M1 -- pure torch, K1/K2/K3 not wired)
# ---------------------------------------------------------------------------


def _split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    # (B, T, D) -> (B, n_head, T, head_dim)
    b, t, d = x.shape
    return x.view(b, t, n_head, d // n_head).permute(0, 2, 1, 3)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # (B, n_head, T, head_dim) -> (B, T, D)
    b, h, t, dh = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, t, h * dh)


def _encoder_block(x: torch.Tensor, w: EncoderLayerWeights, n_head: int) -> torch.Tensor:
    residual = x
    xn = ops.layer_norm(x, w.attn_ln_w, w.attn_ln_b)
    q = ops.gemm(xn, w.attn_q_w, w.attn_q_b)
    k = ops.gemm(xn, w.attn_k_w, None)
    v = ops.gemm(xn, w.attn_v_w, w.attn_v_b)
    q, k, v = _split_heads(q, n_head), _split_heads(k, n_head), _split_heads(v, n_head)
    attn = _merge_heads(ops.encoder_attention(q, k, v))
    x = ops.gemm(attn, w.attn_out_w, w.attn_out_b, residual=residual)

    residual = x
    xn = ops.layer_norm(x, w.mlp_ln_w, w.mlp_ln_b)
    h = ops.gemm(xn, w.mlp_up_w, w.mlp_up_b, epilogue="gelu")
    x = ops.gemm(h, w.mlp_down_w, w.mlp_down_b, residual=residual)
    return x


def encode(mel: torch.Tensor, weights: WhisperWeights) -> torch.Tensor:
    """mel: (B, n_mels, 3000) fp16 -> audio_features: (B, 1500, n_audio_state) fp16."""
    x = torch.nn.functional.gelu(
        torch.nn.functional.conv1d(mel, weights.conv1_w, weights.conv1_b, padding=1)
    )
    x = torch.nn.functional.gelu(
        torch.nn.functional.conv1d(
            x, weights.conv2_w, weights.conv2_b, stride=2, padding=1
        )
    )
    x = x.permute(0, 2, 1)  # (B, 1500, D)
    x = (x + weights.encoder_pos_emb).to(x.dtype)

    n_head = weights.dims.n_audio_head
    for lw in weights.encoder_layers:
        x = _encoder_block(x, lw, n_head)

    x = ops.layer_norm(x, weights.encoder_ln_post_w, weights.encoder_ln_post_b)
    return x


# ---------------------------------------------------------------------------
# Decoder KV cache
# ---------------------------------------------------------------------------


@dataclass
class KVCache:
    """Preallocated, GPU-resident self-attention KV cache (DESIGN.md S7).

    Shape per layer: (B, n_head, max_len, head_dim) -- slicing k[layer_idx]
    yields exactly the (B, 20, T_max, 64) contiguous tensor Agent C's K4
    kernel expects (kernels/DECODE_INTEGRATION.md SS1), so no reshaping is
    needed at the kernel boundary.
    """

    k: torch.Tensor  # (n_layer, B, n_head, max_len, head_dim)
    v: torch.Tensor


@dataclass
class CrossKV:
    """Cross-attention K/V, computed once per chunk from encoder output,
    reused unchanged at every decode step (DESIGN.md S3, S7)."""

    k: torch.Tensor  # (n_layer, B, n_head, 1500, head_dim)
    v: torch.Tensor


@dataclass
class DecodeScratch:
    """Buffers reused across every layer and every decode step, so the K4
    calls never round-trip the caching allocator in the hot loop
    (kernels/DECODE_INTEGRATION.md SS4). Safe to share one buffer across
    all 4 decoder layers within a step: each layer's self/cross-attn
    output is consumed by the very next GEMM call before the following
    layer's attention call overwrites the buffer -- no data hazard.
    """

    self_out: torch.Tensor  # (B, n_head, head_dim) fp16
    cross_out: torch.Tensor  # (B, n_head, head_dim) fp16


def alloc_self_kv_cache(
    weights: WhisperWeights, batch_size: int, max_len: int, device: torch.device
) -> KVCache:
    d = weights.dims
    shape = (d.n_text_layer, batch_size, d.n_text_head, max_len, d.text_head_dim)
    k = torch.zeros(shape, dtype=torch.float16, device=device)
    v = torch.zeros(shape, dtype=torch.float16, device=device)
    return KVCache(k=k, v=v)


def alloc_decode_scratch(
    weights: WhisperWeights, batch_size: int, device: torch.device
) -> DecodeScratch:
    d = weights.dims
    shape = (batch_size, d.n_text_head, d.text_head_dim)
    return DecodeScratch(
        self_out=torch.empty(shape, dtype=torch.float16, device=device),
        cross_out=torch.empty(shape, dtype=torch.float16, device=device),
    )


def precompute_cross_kv(audio_features: torch.Tensor, weights: WhisperWeights) -> CrossKV:
    """K/V for cross-attention, computed once per chunk (DESIGN.md S3: 'K/V
    computed once per chunk, reused every step')."""
    d = weights.dims
    ks, vs = [], []
    for lw in weights.decoder_layers:
        k = ops.gemm(audio_features, lw.cross_attn_k_w, None)
        v = ops.gemm(audio_features, lw.cross_attn_v_w, lw.cross_attn_v_b)
        ks.append(_split_heads(k, d.n_text_head))
        vs.append(_split_heads(v, d.n_text_head))
    return CrossKV(k=torch.stack(ks, dim=0), v=torch.stack(vs, dim=0))


def _split_heads_1(x: torch.Tensor, n_head: int) -> torch.Tensor:
    # (B, D) -> (B, n_head, head_dim); no permute needed, there's no T axis.
    b, d = x.shape
    return x.view(b, n_head, d // n_head)


def _decoder_layer_step(
    x: torch.Tensor,
    lw: DecoderLayerWeights,
    n_head: int,
    self_cache: KVCache,
    layer_idx: int,
    cross_kv: CrossKV,
    position: int,
    scratch: DecodeScratch,
) -> torch.Tensor:
    """One decoder layer, one token position (DESIGN.md S7's per-step
    ordering: K1(LN) -> K2(QKV) -> K4(self) -> K2(proj+residual) -> K1 ->
    K4(cross) -> K2(proj+residual) -> K1 -> K2(FFN up+GELU) -> K2(FFN
    down+residual)). x is always (B, D) -- matches Agent C's K4 contract
    (no time dimension at all, not even T=1), so self_attn_decode/
    cross_attn_decode's (B, 20, 64) output reshapes into the next
    projection's (B, D) input for free (D == n_head * head_dim).
    """
    b = x.shape[0]

    # ---- self attention (K1 -> K2 QKV -> K4 self -> K2 proj+residual) ----
    # NOTE (M2): a fused single-GEMM QKV projection (weights.attn_qkv_w/b,
    # DESIGN.md S6's N=3840 shape) was tried here and measured WORSE under
    # torch.profiler: K4 requires contiguous (B, n_head, head_dim) inputs,
    # and slicing the fused (B, 3840) GEMM output is non-contiguous (row
    # stride 3840 != 1280), so each of q/k/v needs its own .contiguous()
    # copy afterward -- 3 added copy-kernel launches per layer fully offset
    # (and slightly exceeded) the 2 GEMM launches saved. Reverted; kept as
    # a documented dead end so it isn't retried blind. attn_qkv_w/b remain
    # in weights.py, unused, in case a future K2 kernel changes this
    # tradeoff (a real fused-epilogue kernel wouldn't need the .contiguous()
    # copies at all).
    residual = x
    xn = ops.layer_norm(x, lw.attn_ln_w, lw.attn_ln_b)
    q = _split_heads_1(ops.gemm(xn, lw.attn_q_w, lw.attn_q_b), n_head)
    k_new = _split_heads_1(ops.gemm(xn, lw.attn_k_w, None), n_head)
    v_new = _split_heads_1(ops.gemm(xn, lw.attn_v_w, lw.attn_v_b), n_head)

    attn = ops.self_attn_decode(
        q, k_new, v_new, self_cache.k[layer_idx], self_cache.v[layer_idx], position,
        out=scratch.self_out,
    )
    x = ops.gemm(attn.reshape(b, -1), lw.attn_out_w, lw.attn_out_b, residual=residual)

    # ---- cross attention (K1 -> K4 cross -> K2 proj+residual) ----
    residual = x
    xn = ops.layer_norm(x, lw.cross_attn_ln_w, lw.cross_attn_ln_b)
    q = _split_heads_1(ops.gemm(xn, lw.cross_attn_q_w, lw.cross_attn_q_b), n_head)
    attn = ops.cross_attn_decode(
        q, cross_kv.k[layer_idx], cross_kv.v[layer_idx], out=scratch.cross_out
    )
    x = ops.gemm(attn.reshape(b, -1), lw.cross_attn_out_w, lw.cross_attn_out_b, residual=residual)

    # ---- FFN (K1 -> K2 up+GELU -> K2 down+residual) ----
    residual = x
    xn = ops.layer_norm(x, lw.mlp_ln_w, lw.mlp_ln_b)
    h = ops.gemm(xn, lw.mlp_up_w, lw.mlp_up_b, epilogue="gelu")
    x = ops.gemm(h, lw.mlp_down_w, lw.mlp_down_b, residual=residual)
    return x


def decoder_step(
    token_id: torch.Tensor,
    position: int,
    weights: WhisperWeights,
    self_cache: KVCache,
    cross_kv: CrossKV,
    scratch: DecodeScratch,
) -> torch.Tensor:
    """One decode step for every sequence in the batch, uniformly used for
    both the fixed 4-token prompt and every generated token (DESIGN.md S7
    describes a single per-step loop with no separate "prefill" phase --
    the prompt is just the first 4 positions run through the same loop).

    token_id: (B,) long. position: scalar int, 0-indexed. Returns
    logits: (B, n_vocab) fp16 -- the K5 kernel's expected input dtype
    (mask-add/argmax happen in fp32 internally either way, on the torch
    fallback or the kernel).
    """
    d = weights.dims
    x = weights.decoder_token_emb[token_id]  # (B, D)
    pos_emb = weights.decoder_pos_emb[position]  # (D,), broadcasts over B
    x = (x + pos_emb).to(torch.float16)

    for i, lw in enumerate(weights.decoder_layers):
        x = _decoder_layer_step(x, lw, d.n_text_head, self_cache, i, cross_kv, position, scratch)

    x = ops.layer_norm(x, weights.decoder_ln_w, weights.decoder_ln_b)
    # decoder_token_emb_f32_t is precomputed once at load time (weights.py)
    # -- was cast+transposed fresh every step before the M2 profiling pass.
    logits = (x.float() @ weights.decoder_token_emb_f32_t).to(torch.float16)
    return logits
