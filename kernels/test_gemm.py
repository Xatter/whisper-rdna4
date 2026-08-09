"""test_gemm.py -- correctness + microbench for K2 (gemm_wmma.hip) vs torch
matmul (+bias, +gelu, +residual as separate ops, per the task brief).

Usage:
  python test_gemm.py            # correctness only, no GPU lock needed
  python test_gemm.py --bench    # also microbench (take .lock-gpu0 first)
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
from gemm_pack import pack_weights

TOL = 5e-2
M_SHAPES = [48000, 24000, 32, 1]  # DESIGN.md SS6 edge cases (48000=32*1500 encoder B=32,
                                    # 24000=16*1500 encoder B=16 non-multiple-of-128 boundary,
                                    # 32/1 decode M=B)


def make_inputs(M, K_, N, device):
    torch.manual_seed(0)
    x = (torch.randn(M, K_, device=device) * 0.3).half()
    w_kn = (torch.randn(K_, N, device=device) * (1.0 / (K_ ** 0.5))).half()  # unit-scale-ish
    bias = (torch.randn(N, device=device) * 0.1).half()
    residual = (torch.randn(M, N, device=device) * 0.3).half()
    return x, w_kn, bias, residual


def ref_gemm(x, w_kn, bias, epilogue, residual=None):
    y = x.float() @ w_kn.float() + bias.float()
    if epilogue == "bias_gelu":
        y = F.gelu(y, approximate="none")
    elif epilogue == "bias_residual":
        y = y + residual.float()
    return y.half()


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


def torch_ref_op(epilogue):
    def op(x, w_kn, bias, residual):
        y = torch.matmul(x, w_kn)
        y = y + bias
        if epilogue == "bias_gelu":
            y = F.gelu(y, approximate="none")
        elif epilogue == "bias_residual":
            y = y + residual
        return y
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no HIP/CUDA device visible"
    device = "cuda:0"
    all_pass = True

    # (K, N, epilogue, label) -- representative of the real projections in
    # DESIGN.md SS6: attn out-proj (1280->1280, residual), QKV fused proj
    # (1280->3840, bias), FFN up (1280->5120, gelu), FFN down (5120->1280,
    # residual).
    configs = [
        (1280, 1280, "bias", "attn-proj-ish"),
        (1280, 3840, "bias", "qkv-fused"),
        (1280, 5120, "bias_gelu", "ffn-up"),
        (5120, 1280, "bias_residual", "ffn-down"),
    ]

    print("=== K2 gemm_wmma correctness ===")
    for K_, N, epilogue, label in configs:
        for M in M_SHAPES:
            x, w_kn, bias, residual = make_inputs(M, K_, N, device)
            packed = pack_weights(w_kn, already_kn=True)
            ref = ref_gemm(x, w_kn, bias, epilogue, residual)
            got = K.gemm(x, packed, bias, epilogue=epilogue,
                          residual=residual if epilogue == "bias_residual" else None)
            ok = check(f"{label:<14} M={M:>6} K={K_:>5} N={N:>5} epi={epilogue:<13}", got, ref)
            all_pass &= ok

    print()
    print("=== K2 gemm_wmma microbench (vs torch matmul+bias[+gelu/+residual]) ===")
    if args.bench:
        header = f"{'shape':<40} | {'ours (ms)':>10} | {'torch (ms)':>11} | {'speedup':>8}"
        print(header)
        print("-" * len(header))
        for K_, N, epilogue, label in configs:
            for M in (48000, 32):  # headline prefill batch + decode batch
                x, w_kn, bias, residual = make_inputs(M, K_, N, device)
                packed = pack_weights(w_kn, already_kn=True)
                res_arg = residual if epilogue == "bias_residual" else None
                t_ours = bench(lambda *a: K.gemm(*a, epilogue=epilogue, residual=res_arg),
                                x, packed, bias)
                ref_op = torch_ref_op(epilogue)
                t_torch = bench(ref_op, x, w_kn, bias, residual)
                speedup = t_torch / t_ours if t_ours > 0 else float("inf")
                shape_s = f"{label} M={M} K={K_} N={N} epi={epilogue}"
                print(f"{shape_s:<40} | {t_ours:>10.4f} | {t_torch:>11.4f} | {speedup:>7.2f}x")
    else:
        print("  (skipped: run with --bench, after taking the GPU lock)")

    print()
    print("RESULT:", "ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
