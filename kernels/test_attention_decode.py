"""Correctness + microbenchmark for K4 (attention_decode.hip), Agent C.

Run on gpu-host:
  HIP_VISIBLE_DEVICES=1 ~/r9700-whisper/.venv/bin/python \
      ~/r9700-whisper/rocm-whisper-src/kernels/test_attention_decode.py

Correctness: compares the kernel against an inline torch reference
(masked/causal softmax attention over a KV cache, fp32 accumulation) at
B in {8, 32}, self-attn step=0 (first decode step, cache len 1) and
step=447 (T_max=448, full cache), and cross-attn T=1500.
Tolerance: max-abs error <= 5e-2 (fp16 unit-scale tensors, per DESIGN.md
section 6/8).

Microbench: kernel vs the equivalent torch op sequence it replaces
(cache write + QK^T matmul + softmax + PV matmul), warmup + synchronize +
median-of-10, matching cheatsheet section 14 measurement hygiene.
"""

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_api  # noqa: E402

HEAD_DIM = decode_api.HEAD_DIM
DEVICE = "cuda:0"  # HIP_VISIBLE_DEVICES pins this to the intended physical GPU
TOL = 5e-2


def _seed(n):
    torch.manual_seed(n)


# ---------------------------------------------------------------------
# Torch reference implementations
# ---------------------------------------------------------------------
def torch_self_attn_reference(q, k_new, v_new, k_cache, v_cache, step):
    """Mirrors the kernel's semantics exactly: append at `step`, then
    causal-attend over t in [0, step] inclusive, fp32 accumulation."""
    B, H, Dh = q.shape
    k_cache = k_cache.clone()
    v_cache = v_cache.clone()
    k_cache[:, :, step, :] = k_new
    v_cache[:, :, step, :] = v_new
    T = step + 1
    K = k_cache[:, :, :T, :].float()
    V = v_cache[:, :, :T, :].float()
    scale = 1.0 / (Dh ** 0.5)
    scores = torch.einsum("bhd,bhtd->bht", q.float() * scale, K)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bht,bhtd->bhd", probs, V)
    return out.half(), k_cache, v_cache


def torch_cross_attn_reference(q, k_cache_x, v_cache_x):
    B, H, Dh = q.shape
    K = k_cache_x.float()
    V = v_cache_x.float()
    scale = 1.0 / (Dh ** 0.5)
    scores = torch.einsum("bhd,bhtd->bht", q.float() * scale, K)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bht,bhtd->bhd", probs, V)
    return out.half()


# ---------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------
def check_self_attn(B, H, T_max, step, label):
    _seed(1234 + B + step)
    q = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k_new = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_new = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    # Pre-fill cache positions [0, step) with plausible history so a full
    # cache (step=447) is a real test, not a degenerate near-empty one.
    k_cache = torch.randn(B, H, T_max, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_cache = torch.randn(B, H, T_max, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k_cache_kernel = k_cache.clone()
    v_cache_kernel = v_cache.clone()

    ref_out, k_cache_ref, v_cache_ref = torch_self_attn_reference(q, k_new, v_new, k_cache, v_cache, step)

    kernel_out = decode_api.self_attn_decode_step(q, k_new, v_new, k_cache_kernel, v_cache_kernel, step)
    torch.cuda.synchronize()

    out_err = (kernel_out.float() - ref_out.float()).abs().max().item()
    # only position `step` was ever written by the kernel; compare exactly that slice
    k_append_err = (k_cache_kernel[:, :, step, :].float() - k_cache_ref[:, :, step, :].float()).abs().max().item()
    v_append_err = (v_cache_kernel[:, :, step, :].float() - v_cache_ref[:, :, step, :].float()).abs().max().item()

    ok = out_err <= TOL and k_append_err == 0.0 and v_append_err == 0.0
    print(f"[self-attn {label}] B={B} H={H} step={step} T={step+1} "
          f"out_max_abs_err={out_err:.4e} k_append_err={k_append_err:.4e} "
          f"v_append_err={v_append_err:.4e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_cross_attn(B, H, T_cross, label):
    _seed(5678 + B)
    q = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k_cache_x = torch.randn(B, H, T_cross, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_cache_x = torch.randn(B, H, T_cross, HEAD_DIM, dtype=torch.float16, device=DEVICE)

    ref_out = torch_cross_attn_reference(q, k_cache_x, v_cache_x)
    kernel_out = decode_api.cross_attn_decode_step(q, k_cache_x, v_cache_x)
    torch.cuda.synchronize()

    out_err = (kernel_out.float() - ref_out.float()).abs().max().item()
    ok = out_err <= TOL
    print(f"[cross-attn {label}] B={B} H={H} T={T_cross} out_max_abs_err={out_err:.4e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------
# Microbenchmark
# ---------------------------------------------------------------------
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


def bench_self_attn(B, H, T_max, step):
    q = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k_new = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_new = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k_cache = torch.randn(B, H, T_max, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_cache = torch.randn(B, H, T_max, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    out = torch.empty(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / (HEAD_DIM ** 0.5)

    def kernel_call():
        decode_api.self_attn_decode_step(q, k_new, v_new, k_cache, v_cache, step, out=out)

    def torch_call():
        # the ~6-op sequence K4a replaces: 2 cache writes (index_copy),
        # matmul, scale, softmax, matmul.
        k_cache[:, :, step, :] = k_new
        v_cache[:, :, step, :] = v_new
        T = step + 1
        K = k_cache[:, :, :T, :]
        V = v_cache[:, :, :T, :]
        scores = torch.einsum("bhd,bhtd->bht", q, K).float() * scale
        probs = torch.softmax(scores, dim=-1).half()
        _ = torch.einsum("bht,bhtd->bhd", probs, V)

    k_us = _median_us(kernel_call)
    t_us = _median_us(torch_call)
    print(f"[bench self-attn] B={B} step={step}: kernel={k_us:.1f}us torch_seq(~6 ops)={t_us:.1f}us "
          f"speedup={t_us / k_us:.2f}x")
    return k_us, t_us


def bench_cross_attn(B, H, T_cross):
    q = torch.randn(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k_cache_x = torch.randn(B, H, T_cross, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_cache_x = torch.randn(B, H, T_cross, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    out = torch.empty(B, H, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / (HEAD_DIM ** 0.5)

    def kernel_call():
        decode_api.cross_attn_decode_step(q, k_cache_x, v_cache_x, out=out)

    def torch_call():
        scores = torch.einsum("bhd,bhtd->bht", q, k_cache_x).float() * scale
        probs = torch.softmax(scores, dim=-1).half()
        _ = torch.einsum("bht,bhtd->bhd", probs, v_cache_x)

    k_us = _median_us(kernel_call)
    t_us = _median_us(torch_call)
    print(f"[bench cross-attn] B={B} T={T_cross}: kernel={k_us:.1f}us torch_seq(~4 ops)={t_us:.1f}us "
          f"speedup={t_us / k_us:.2f}x")
    return k_us, t_us


def main():
    assert torch.cuda.is_available(), "no HIP/CUDA device visible"
    H = 20
    T_max = 448
    T_cross = 1500

    all_ok = True
    for B in (8, 32):
        all_ok &= check_self_attn(B, H, T_max, step=0, label="first-step")
        all_ok &= check_self_attn(B, H, T_max, step=T_max - 1, label="full-cache")
        all_ok &= check_cross_attn(B, H, T_cross, label="T=1500")

    print("\n=== correctness summary:", "ALL PASS" if all_ok else "FAILURES ABOVE", "===\n")

    print("=== microbenchmarks (median of 10, 10 warmup) ===")
    for B in (8, 32):
        bench_self_attn(B, H, T_max, step=0)
        bench_self_attn(B, H, T_max, step=T_max - 1)
        bench_cross_attn(B, H, T_cross)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
