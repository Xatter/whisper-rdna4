"""kernels_api.py -- ctypes wrappers around libwhisper_kernels.so for Agent B's
kernels (K1 layernorm, K2 gemm_wmma, K3 attention_encoder).

Signatures below (`layer_norm`, `gemm`, `encoder_attention`) match the names
Agent A's whisper_rocm/ops.py seam expects (per the task brief). Agent A
registers these; this module does not touch model.py/ops.py itself.

ctypes interop pattern follows ~/r9700-whisper/smoke/ on gpu-host: raw
`tensor.data_ptr()` + `torch.cuda.current_stream().cuda_stream`, no torch C++
extension API. Every launcher returns a hipError_t (0 == success); non-zero
raises RuntimeError here so failures surface immediately instead of silently
corrupting downstream tensors.
"""

import ctypes
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gemm_pack import BLOCK_K, BLOCK_N  # noqa: E402  (validation only)

_LIB_PATH = os.path.join(_HERE, "libwhisper_kernels.so")
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(_LIB_PATH)

    lib.launch_layernorm.restype = ctypes.c_int
    lib.launch_layernorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ]
    lib.launch_fused_residual_layernorm.restype = ctypes.c_int
    lib.launch_fused_residual_layernorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_void_p,
    ]

    gemm_argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    for tile in ("128", "32"):
        for epi in ("bias", "bias_gelu", "bias_residual"):
            fn = getattr(lib, f"launch_gemm_{tile}_{epi}")
            fn.restype = ctypes.c_int
            fn.argtypes = gemm_argtypes

    lib.launch_attention_encoder.restype = ctypes.c_int
    lib.launch_attention_encoder.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]

    _lib = lib
    return _lib


def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _null() -> ctypes.c_void_p:
    return ctypes.c_void_p(0)


def _stream() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(err: int, name: str):
    if err != 0:
        raise RuntimeError(f"{name} returned hipError_t={err}")


# ---------------------------------------------------------------------------
# K1: layer_norm
# ---------------------------------------------------------------------------
def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                residual: torch.Tensor = None, eps: float = 1e-5):
    """LayerNorm over the last dim (cols) of a (rows, cols) fp16 tensor.

    If `residual` is None: returns normed = LN(x).
    If `residual` is given: computes s = residual + x, returns
      (normed, s) where normed = LN(s) and s (fp16) is the updated residual
      stream for the next sublayer. This is the "fused residual+LN" variant
      from DESIGN.md SS6 (x = LN(residual + y)).
    """
    lib = _load()
    assert x.dtype == torch.float16 and weight.dtype == torch.float16 and bias.dtype == torch.float16
    assert x.dim() == 2
    rows, cols = x.shape
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    if residual is None:
        out = torch.empty_like(x)
        err = lib.launch_layernorm(_ptr(x), _ptr(weight), _ptr(bias), _ptr(out),
                                    rows, cols, ctypes.c_float(eps), _stream())
        _check(err, "launch_layernorm")
        return out

    assert residual.shape == x.shape and residual.dtype == torch.float16
    residual = residual.contiguous()
    out_normed = torch.empty_like(x)
    out_residual = torch.empty_like(x)
    err = lib.launch_fused_residual_layernorm(
        _ptr(residual), _ptr(x), _ptr(weight), _ptr(bias), _ptr(out_normed),
        _ptr(out_residual), rows, cols, ctypes.c_float(eps), _stream())
    _check(err, "launch_fused_residual_layernorm")
    return out_normed, out_residual


# ---------------------------------------------------------------------------
# K2: gemm
# ---------------------------------------------------------------------------
_EPILOGUES = {"bias", "bias_gelu", "bias_residual"}


def gemm(x: torch.Tensor, packed_weight: torch.Tensor, bias: torch.Tensor,
          epilogue: str = "bias", residual: torch.Tensor = None,
          out: torch.Tensor = None, tile: str = "auto") -> torch.Tensor:
    """Y = epilogue(X @ W + bias[, + residual]), W supplied pre-packed.

    x: (M, K) fp16, row-major, ordinary activations tensor (no packing).
    packed_weight: output of gemm_pack.pack_weights(...) for this layer's
      (K, N) weight -- shape (N/128, K/32, 32, 128) fp16.
    bias: (N,) fp16.
    epilogue: "bias" | "bias_gelu" | "bias_residual".
    residual: required iff epilogue == "bias_residual"; (M, N) fp16. May
      safely alias `out` for an in-place residual-stream update -- the
      kernel reads residual[row,col] and writes out[row,col] for the same
      element on the same thread, so no other thread ever races that
      address.
    tile: "auto" picks "32" (the decode-friendly small-M tile) for M < 128,
      else "128" (the main prefill tile); or pass "128"/"32" explicitly.

    Returns out (allocated if not supplied), shape (M, N) fp16.
    """
    lib = _load()
    assert epilogue in _EPILOGUES, f"unknown epilogue {epilogue!r}"
    assert x.dtype == torch.float16 and packed_weight.dtype == torch.float16
    M, K = x.shape
    N_blocks, K_blocks, bk, bn = packed_weight.shape
    assert bk == BLOCK_K and bn == BLOCK_N, "packed_weight not produced by gemm_pack.pack_weights"
    assert K_blocks * BLOCK_K == K, f"K mismatch: x has K={K}, packed_weight implies K={K_blocks*BLOCK_K}"
    N = N_blocks * BLOCK_N
    assert bias.shape == (N,) and bias.dtype == torch.float16

    if tile == "auto":
        tile = "32" if M < 128 else "128"
    assert tile in ("32", "128")

    x = x.contiguous()
    packed_weight = packed_weight.contiguous()
    bias = bias.contiguous()

    residual_ptr = _null()
    if epilogue == "bias_residual":
        assert residual is not None, "epilogue='bias_residual' requires `residual`"
        assert residual.shape == (M, N) and residual.dtype == torch.float16
        if residual.data_ptr() != (out.data_ptr() if out is not None else -1):
            residual = residual.contiguous()
        residual_ptr = _ptr(residual)
    else:
        assert residual is None, "residual is only used by epilogue='bias_residual'"

    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    fn = getattr(lib, f"launch_gemm_{tile}_{epilogue}")
    err = fn(_ptr(x), _ptr(packed_weight), _ptr(bias), residual_ptr, _ptr(out),
             M, N, K, _stream())
    _check(err, fn.__name__ if hasattr(fn, "__name__") else f"launch_gemm_{tile}_{epilogue}")
    return out


# ---------------------------------------------------------------------------
# K3: encoder_attention
# ---------------------------------------------------------------------------
def encoder_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float = None) -> torch.Tensor:
    """Non-causal (bidirectional, no mask) flash attention, head_dim=64.

    q, k, v: fp16 CUDA tensors, either (B, H, S, 64) or (BH, S, 64). S is the
      REAL sequence length (e.g. 1500) -- padding to the next multiple of 16
      and the 1/sqrt(64) scale fold-into-Q both happen inside this wrapper,
      per DESIGN.md SS6 ("your wrapper does the fold").

    Returns a tensor of the same shape as the input, fp16.
    """
    lib = _load()
    orig_dim = q.dim()
    if orig_dim == 4:
        B, H, S, D = q.shape
        BH = B * H
        q3 = q.reshape(BH, S, D)
        k3 = k.reshape(BH, S, D)
        v3 = v.reshape(BH, S, D)
    else:
        BH, S, D = q.shape
        q3, k3, v3 = q, k, v
    assert D == 64, "K3 is specialized to head_dim=64"
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    S_pad = ((S + 15) // 16) * 16
    dev = q3.device
    qp = torch.zeros((BH, S_pad, D), dtype=torch.float16, device=dev)
    kp = torch.zeros((BH, S_pad, D), dtype=torch.float16, device=dev)
    vp = torch.zeros((BH, S_pad, D), dtype=torch.float16, device=dev)
    qp[:, :S, :] = q3.to(torch.float16) * scale
    kp[:, :S, :] = k3.to(torch.float16)
    vp[:, :S, :] = v3.to(torch.float16)
    op = torch.empty((BH, S_pad, D), dtype=torch.float16, device=dev)

    err = lib.launch_attention_encoder(_ptr(qp), _ptr(kp), _ptr(vp), _ptr(op),
                                        BH, S_pad, S, _stream())
    _check(err, "launch_attention_encoder")

    out = op[:, :S, :].contiguous()
    if orig_dim == 4:
        out = out.reshape(B, H, S, D)
    return out
