"""test_attention_encoder.py -- correctness + microbench for K3
(attention_encoder.hip) vs torch.nn.functional.scaled_dot_product_attention
(non-causal, no mask).

Usage:
  python test_attention_encoder.py            # correctness only
  python test_attention_encoder.py --bench     # also microbench (GPU lock)
"""
import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels_api as K

HEAD_DIM = 64
HEADS = 20
S = 1500  # real encoder seq len; K3 pads to 1504 internally
TOL = 5e-2
B_SHAPES = [1, 2, 4]  # BH = B*20; keeps correctness runtime reasonable


def make_qkv(B, device):
    torch.manual_seed(0)
    shape = (B, HEADS, S, HEAD_DIM)
    q = (torch.randn(shape, device=device) * 0.3).half()
    k = (torch.randn(shape, device=device) * 0.3).half()
    v = (torch.randn(shape, device=device) * 0.3).half()
    return q, k, v


def ref_attn(q, k, v, scale):
    # fp32 upcast reference -- more accurate baseline than fp16 SDPA, and
    # what the kernel's fp32-accumulate math should track most closely.
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=None, is_causal=False,
        scale=scale,
    ).half()


def check(name, got, ref):
    diff = (got.float() - ref.float()).abs().max().item()
    status = "PASS" if diff <= TOL else "FAIL"
    print(f"  [{status}] {name}: max-abs-err={diff:.5f} (tol {TOL})")
    return status == "PASS"


def bench(fn, *args, iters=10, warmup=3):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no HIP/CUDA device visible"
    device = "cuda:0"
    scale = 1.0 / (HEAD_DIM ** 0.5)
    all_pass = True

    print(f"=== K3 attention_encoder correctness (S={S} padded to 1504 internally) ===")
    for B in B_SHAPES:
        q, k, v = make_qkv(B, device)
        ref = ref_attn(q, k, v, scale)
        got = K.encoder_attention(q, k, v, scale=scale)
        ok = check(f"B={B:>3} BH={B*HEADS:>4} S={S}", got, ref)
        all_pass &= ok

    # explicit 1500->1504 padding edge case at a 3-D (BH,S,D) call shape too
    B = 1
    q4, k4, v4 = make_qkv(B, device)
    q3 = q4.reshape(B * HEADS, S, HEAD_DIM)
    k3 = k4.reshape(B * HEADS, S, HEAD_DIM)
    v3 = v4.reshape(B * HEADS, S, HEAD_DIM)
    ref3 = ref_attn(q4, k4, v4, scale).reshape(B * HEADS, S, HEAD_DIM)
    got3 = K.encoder_attention(q3, k3, v3, scale=scale)
    ok = check("3-D (BH,S,D) call shape, padding edge", got3, ref3)
    all_pass &= ok

    print()
    print("=== K3 attention_encoder microbench (vs F.scaled_dot_product_attention) ===")
    if args.bench:
        header = f"{'B':>4} | {'BH':>5} | {'ours (ms)':>10} | {'torch (ms)':>11} | {'speedup':>8}"
        print(header)
        print("-" * len(header))
        for B in (8, 16, 32):
            q, k, v = make_qkv(B, device)
            t_ours = bench(lambda *a: K.encoder_attention(*a, scale=scale), q, k, v)
            qh, kh, vh = q, k, v  # fp16 SDPA -- the actual torch equivalent at runtime
            t_torch = bench(
                lambda *a: F.scaled_dot_product_attention(*a, attn_mask=None, is_causal=False, scale=scale),
                qh, kh, vh)
            speedup = t_torch / t_ours if t_ours > 0 else float("inf")
            print(f"{B:>4} | {B*HEADS:>5} | {t_ours:>10.4f} | {t_torch:>11.4f} | {speedup:>7.2f}x")
    else:
        print("  (skipped: run with --bench, after taking the GPU lock)")

    print()
    print("RESULT:", "ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
