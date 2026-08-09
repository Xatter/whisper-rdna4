"""test_layernorm.py -- correctness + microbench for K1 (layernorm.hip) vs
torch.nn.functional.layer_norm.

Usage:
  python test_layernorm.py            # correctness only, no GPU lock needed
  python test_layernorm.py --bench    # also microbench (take .lock-gpu0 first)
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

COLS = 1280
TOL = 5e-2
SHAPES = [48000, 24000, 32, 1]  # M = rows, per DESIGN.md SS6 edge cases


def make_inputs(rows, cols, device):
    torch.manual_seed(0)
    x = (torch.randn(rows, cols, device=device) * 0.5).half()
    w = (torch.randn(cols, device=device) * 0.5 + 1.0).half()
    b = (torch.randn(cols, device=device) * 0.1).half()
    return x, w, b


def ref_ln(x, w, b, eps=1e-5):
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), eps).half()


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
    all_pass = True

    print("=== K1 layernorm correctness ===")
    for rows in SHAPES:
        x, w, b = make_inputs(rows, COLS, device)
        ref = ref_ln(x, w, b)
        got = K.layer_norm(x, w, b)
        ok = check(f"plain LN  M={rows:>6}", got, ref)
        all_pass &= ok

        residual = (torch.randn(rows, COLS, device=device) * 0.3).half()
        s = residual.float() + x.float()
        ref_normed = ref_ln(s.half(), w, b)
        got_normed, got_residual = K.layer_norm(x, w, b, residual=residual)
        ok1 = check(f"fused LN  M={rows:>6} (normed)", got_normed, ref_normed)
        ok2 = check(f"fused LN  M={rows:>6} (residual sum)", got_residual, s.half())
        all_pass &= ok1 and ok2

    print()
    print("=== K1 layernorm microbench (vs torch.nn.functional.layer_norm) ===")
    if args.bench:
        header = f"{'M':>8} | {'ours (ms)':>10} | {'torch (ms)':>11} | {'speedup':>8}"
        print(header)
        print("-" * len(header))
        for rows in SHAPES:
            x, w, b = make_inputs(rows, COLS, device)
            t_ours = bench(K.layer_norm, x, w, b)
            t_torch = bench(lambda *a: F.layer_norm(*a), x, (COLS,), w, b, 1e-5)
            speedup = t_torch / t_ours if t_ours > 0 else float("inf")
            print(f"{rows:>8} | {t_ours:>10.4f} | {t_torch:>11.4f} | {speedup:>7.2f}x")
    else:
        print("  (skipped: run with --bench, after taking the GPU lock)")

    print()
    print("RESULT:", "ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
