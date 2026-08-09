"""Load large-v3-turbo.pt (OpenAI checkpoint format) into fp16 tensors laid
out for whisper_rocm.model / ops.gemm (DESIGN.md S5, S3).

Checkpoint format (verified against
~/.cache/whisper/large-v3-turbo.pt on gpu-host):
    {"dims": {...}, "model_state_dict": {<587 keys>}}
dims: n_mels=128, n_vocab=51866, n_audio_ctx=1500, n_audio_state=1280,
      n_audio_head=20, n_audio_layer=32, n_text_ctx=448, n_text_state=1280,
      n_text_head=20, n_text_layer=4.

Linear weights are kept in torch.nn.Linear convention: [out_features,
in_features], fp16, contiguous -- this is what ops.gemm's torch fallback
(F.linear) expects with zero transpose. K2's tiled prepacking (DESIGN.md
S6) is Agent B's concern and happens inside their registered ops.gemm
kernel callable, not here -- see the module docstring in ops.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ModelDimensions:
    n_mels: int
    n_vocab: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

    @property
    def audio_head_dim(self) -> int:
        return self.n_audio_state // self.n_audio_head

    @property
    def text_head_dim(self) -> int:
        return self.n_text_state // self.n_text_head


@dataclass
class EncoderLayerWeights:
    attn_q_w: torch.Tensor
    attn_q_b: torch.Tensor
    attn_k_w: torch.Tensor  # no bias (matches openai-whisper's Linear(bias=False))
    attn_v_w: torch.Tensor
    attn_v_b: torch.Tensor
    attn_out_w: torch.Tensor
    attn_out_b: torch.Tensor
    attn_ln_w: torch.Tensor
    attn_ln_b: torch.Tensor
    mlp_up_w: torch.Tensor
    mlp_up_b: torch.Tensor
    mlp_down_w: torch.Tensor
    mlp_down_b: torch.Tensor
    mlp_ln_w: torch.Tensor
    mlp_ln_b: torch.Tensor


@dataclass
class DecoderLayerWeights:
    attn_q_w: torch.Tensor
    attn_q_b: torch.Tensor
    attn_k_w: torch.Tensor
    attn_v_w: torch.Tensor
    attn_v_b: torch.Tensor
    # Fused self-attn QKV projection (M2 op-count reduction, DESIGN.md S6's
    # N=3840 "fused QKV" shape): attn_qkv_w = cat([q_w, k_w, v_w], dim=0),
    # attn_qkv_b = cat([q_b, zeros(1280), v_b]) -- k has no bias in the
    # checkpoint (openai-whisper's Linear(bias=False)), so its slice of the
    # fused bias is exactly zero, making this bit-identical to 3 separate
    # GEMMs, not an approximation. One GEMM call instead of 3 in the decode
    # step's self-attention preamble.
    attn_qkv_w: torch.Tensor  # (3840, 1280)
    attn_qkv_b: torch.Tensor  # (3840,)
    attn_out_w: torch.Tensor
    attn_out_b: torch.Tensor
    attn_ln_w: torch.Tensor
    attn_ln_b: torch.Tensor
    cross_attn_q_w: torch.Tensor
    cross_attn_q_b: torch.Tensor
    cross_attn_k_w: torch.Tensor
    cross_attn_v_w: torch.Tensor
    cross_attn_v_b: torch.Tensor
    cross_attn_out_w: torch.Tensor
    cross_attn_out_b: torch.Tensor
    cross_attn_ln_w: torch.Tensor
    cross_attn_ln_b: torch.Tensor
    mlp_up_w: torch.Tensor
    mlp_up_b: torch.Tensor
    mlp_down_w: torch.Tensor
    mlp_down_b: torch.Tensor
    mlp_ln_w: torch.Tensor
    mlp_ln_b: torch.Tensor


@dataclass
class WhisperWeights:
    dims: ModelDimensions
    conv1_w: torch.Tensor
    conv1_b: torch.Tensor
    conv2_w: torch.Tensor
    conv2_b: torch.Tensor
    encoder_pos_emb: torch.Tensor
    encoder_layers: List[EncoderLayerWeights]
    encoder_ln_post_w: torch.Tensor
    encoder_ln_post_b: torch.Tensor
    decoder_token_emb: torch.Tensor  # (n_vocab, n_text_state); tied to logit head
    decoder_token_emb_f32_t: torch.Tensor  # (n_text_state, n_vocab), fp32, precomputed
    decoder_pos_emb: torch.Tensor  # (n_text_ctx, n_text_state)
    decoder_layers: List[DecoderLayerWeights]
    decoder_ln_w: torch.Tensor
    decoder_ln_b: torch.Tensor


def _t(sd: dict, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return sd[key].to(device=device, dtype=dtype).contiguous()


def load_weights(
    checkpoint_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
) -> WhisperWeights:
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dims = ModelDimensions(**ckpt["dims"])
    sd = ckpt["model_state_dict"]

    def t(key: str) -> torch.Tensor:
        return _t(sd, key, device, dtype)

    encoder_layers = []
    for i in range(dims.n_audio_layer):
        p = f"encoder.blocks.{i}."
        encoder_layers.append(
            EncoderLayerWeights(
                attn_q_w=t(p + "attn.query.weight"),
                attn_q_b=t(p + "attn.query.bias"),
                attn_k_w=t(p + "attn.key.weight"),
                attn_v_w=t(p + "attn.value.weight"),
                attn_v_b=t(p + "attn.value.bias"),
                attn_out_w=t(p + "attn.out.weight"),
                attn_out_b=t(p + "attn.out.bias"),
                attn_ln_w=t(p + "attn_ln.weight"),
                attn_ln_b=t(p + "attn_ln.bias"),
                mlp_up_w=t(p + "mlp.0.weight"),
                mlp_up_b=t(p + "mlp.0.bias"),
                mlp_down_w=t(p + "mlp.2.weight"),
                mlp_down_b=t(p + "mlp.2.bias"),
                mlp_ln_w=t(p + "mlp_ln.weight"),
                mlp_ln_b=t(p + "mlp_ln.bias"),
            )
        )

    decoder_layers = []
    for i in range(dims.n_text_layer):
        p = f"decoder.blocks.{i}."
        attn_q_w, attn_q_b = t(p + "attn.query.weight"), t(p + "attn.query.bias")
        attn_k_w = t(p + "attn.key.weight")
        attn_v_w, attn_v_b = t(p + "attn.value.weight"), t(p + "attn.value.bias")
        attn_qkv_w = torch.cat([attn_q_w, attn_k_w, attn_v_w], dim=0).contiguous()
        attn_qkv_b = torch.cat(
            [attn_q_b, torch.zeros_like(attn_q_b), attn_v_b], dim=0
        ).contiguous()
        decoder_layers.append(
            DecoderLayerWeights(
                attn_q_w=attn_q_w,
                attn_q_b=attn_q_b,
                attn_k_w=attn_k_w,
                attn_v_w=attn_v_w,
                attn_v_b=attn_v_b,
                attn_qkv_w=attn_qkv_w,
                attn_qkv_b=attn_qkv_b,
                attn_out_w=t(p + "attn.out.weight"),
                attn_out_b=t(p + "attn.out.bias"),
                attn_ln_w=t(p + "attn_ln.weight"),
                attn_ln_b=t(p + "attn_ln.bias"),
                cross_attn_q_w=t(p + "cross_attn.query.weight"),
                cross_attn_q_b=t(p + "cross_attn.query.bias"),
                cross_attn_k_w=t(p + "cross_attn.key.weight"),
                cross_attn_v_w=t(p + "cross_attn.value.weight"),
                cross_attn_v_b=t(p + "cross_attn.value.bias"),
                cross_attn_out_w=t(p + "cross_attn.out.weight"),
                cross_attn_out_b=t(p + "cross_attn.out.bias"),
                cross_attn_ln_w=t(p + "cross_attn_ln.weight"),
                cross_attn_ln_b=t(p + "cross_attn_ln.bias"),
                mlp_up_w=t(p + "mlp.0.weight"),
                mlp_up_b=t(p + "mlp.0.bias"),
                mlp_down_w=t(p + "mlp.2.weight"),
                mlp_down_b=t(p + "mlp.2.bias"),
                mlp_ln_w=t(p + "mlp_ln.weight"),
                mlp_ln_b=t(p + "mlp_ln.bias"),
            )
        )

    decoder_token_emb = t("decoder.token_embedding.weight")
    # Hoisted out of the decode loop (M2 profiling finding): the logit GEMM
    # is `x @ decoder_token_emb.float().T` every single step; casting +
    # transposing the full (51866, 1280) tied-embedding matrix per step was
    # ~39% of decode-step CPU time in a torch.profiler capture (aten::to /
    # _to_copy / copy_) for what is a load-time-constant tensor. Compute
    # the fp32 transpose once, here, and reuse it every step.
    decoder_token_emb_f32_t = decoder_token_emb.float().T.contiguous()

    return WhisperWeights(
        dims=dims,
        conv1_w=t("encoder.conv1.weight"),
        conv1_b=t("encoder.conv1.bias"),
        conv2_w=t("encoder.conv2.weight"),
        conv2_b=t("encoder.conv2.bias"),
        encoder_pos_emb=t("encoder.positional_embedding"),
        encoder_layers=encoder_layers,
        encoder_ln_post_w=t("encoder.ln_post.weight"),
        encoder_ln_post_b=t("encoder.ln_post.bias"),
        decoder_token_emb=decoder_token_emb,
        decoder_token_emb_f32_t=decoder_token_emb_f32_t,
        decoder_pos_emb=t("decoder.positional_embedding"),
        decoder_layers=decoder_layers,
        decoder_ln_w=t("decoder.ln.weight"),
        decoder_ln_b=t("decoder.ln.bias"),
    )
