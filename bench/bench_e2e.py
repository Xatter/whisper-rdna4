#!/usr/bin/env python3
"""Corpus benchmark (DESIGN.md S9). Single-GPU and dual-GPU (two worker
subprocesses, one HIP_VISIBLE_DEVICES each) modes; JSON output with
per-file + overall RTF and a per-stage timing breakdown.

Single-GPU usage (HIP_VISIBLE_DEVICES must already be set by the caller
before this process starts -- GPU selection binds at HIP-runtime-init
time inside `import torch`, so it has to be in the environment before
python even starts):
    HIP_VISIBLE_DEVICES=0 bench/bench_e2e.py --mode single \\
        --corpus-dir ~/insanely-fast-whisper-rocm/bench-audio \\
        --checkpoint ~/.cache/whisper/large-v3-turbo.pt \\
        --batch-size 16 --out results.json

Dual-GPU usage (spawns two worker subprocesses with HIP_VISIBLE_DEVICES=0
and =1 respectively via subprocess.Popen(env=...), splits the corpus
between them, measures ONE wall clock over the whole run -- DESIGN.md S5
point 5 / S9):
    bench/bench_e2e.py --mode dual --corpus-dir ... --checkpoint ... \\
        --batch-size 16 --out results.json

--files restricts the corpus to specific filenames (comma-separated), to
run one file only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


def _list_corpus(corpus_dir: str, files: list[str] | None) -> list[str]:
    p = Path(corpus_dir)
    if files:
        return [str(p / f) for f in files]
    return sorted(str(f) for f in p.glob("*.mp3"))


def _run_single_or_worker(args) -> dict:
    """Runs in-process: load weights once, transcribe each assigned file,
    return the results dict. Requires HIP_VISIBLE_DEVICES to already be
    correct in this process's environment -- true for --mode single (set
    by the calling shell) and for --mode worker (set by the dual-mode
    parent's subprocess.Popen(env=...) before this interpreter started).
    """
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from whisper_rocm import ops as whisper_ops
    from whisper_rocm import pipeline as whisper_pipeline
    from whisper_rocm import tokenizer as whisper_tokenizer
    from whisper_rocm import weights as whisper_weights

    # M5 KERNEL BUG (found 2026-08-09): timestamp_rules_kernel's Rule 4
    # (logsumexp-over-timestamps vs max-text-token, kernels/logits.hip) was
    # NaN-poisoning its online-logsumexp accumulator (initialized at -inf,
    # hit exp(-inf - -inf) = NaN when monotonicity-masked timestamps landed
    # first in a thread's stride) and so failed to ban non-timestamp tokens
    # (incl. EOT) in cases it should have. Fixed by Agent C (skips -inf
    # elements, mathematically exact); 78/78 kernel tests pass including a
    # stride sweep over the exact repro shape. Re-verified bit-identical
    # against the torch fallback (forced-prefix replay + one full file) --
    # see RESULTS.md's "Timestamps mode" section. K5 now runs on the kernel
    # path like K4, no bypass needed.
    if args.use_kernels:
        from whisper_rocm import kernel_bridge

        kernel_bridge.register()
        whisper_ops.USE_KERNELS = True

    device = torch.device("cuda:0")  # index within *this process's* HIP_VISIBLE_DEVICES
    files = _list_corpus(args.corpus_dir, args.files.split(",") if args.files else None)
    if not files:
        raise RuntimeError(f"no files found in {args.corpus_dir}")

    t0 = time.perf_counter()
    w = whisper_weights.load_weights(args.checkpoint, device=device)
    tok = whisper_tokenizer.Tokenizer(n_vocab=w.dims.n_vocab)
    torch.cuda.synchronize(device)
    load_s = time.perf_counter() - t0

    def _prepare(f):
        return whisper_pipeline.prepare_chunks(f, chunking=args.chunking)

    def _transcribe(prepared):
        return whisper_pipeline.transcribe_prepared(
            prepared, w, tok, device, batch_size=args.batch_size,
            enable_ngram_ban=args.ngram_ban, timestamps=args.timestamps,
            retry_ladder_enabled=not args.no_retry_ladder,
        )

    # One warmup pass (compile/alloc effects, clock ramp after WoL/idle) -- DESIGN.md S9.
    warmup_prepared = _prepare(files[0])
    _ = _transcribe(warmup_prepared)
    torch.cuda.synchronize(device)

    # CPU/GPU overlap (orchestrator directive 2026-08-10, "VAD cost
    # recovery"): while file N's GPU work runs below, a background thread
    # prepares file N+1's chunks (audio load + VAD if chunking="vad"), so
    # file N+1's GPU work can start the instant file N's finishes instead of
    # blocking on VAD CPU time again. VAD releases the GIL during its native
    # (onnxruntime/torch) compute, so this is genuine wall-clock overlap, not
    # concurrency theater. File 0 reuses warmup_prepared directly (already
    # computed above) rather than re-running VAD on it a second time.
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    next_future: "Future" = Future()
    next_future.set_result(warmup_prepared)

    per_file = []
    corpus_wall_t0 = time.perf_counter()
    for idx, f in enumerate(files):
        t0 = time.perf_counter()
        prepared = next_future.result()
        if idx + 1 < len(files):
            next_future = prefetch_pool.submit(_prepare, files[idx + 1])
        result = _transcribe(prepared)
        wall_s = time.perf_counter() - t0
        rtf = wall_s / result.duration_s if result.duration_s > 0 else float("nan")
        per_file.append(
            {
                "file": os.path.basename(f),
                "duration_s": result.duration_s,
                "wall_s": wall_s,
                "rtf": rtf,
                "num_chunks": result.num_chunks,
                "text": result.text,
                "segments": result.segments,
                "retry_telemetry": result.retry_telemetry,
                "vad_stats": result.vad_stats,
                "stage_timings_s": {
                    "load": result.timings.load_s,
                    "vad": result.timings.vad_s,  # CPU-only, 0 for hard/stride
                    "mel": result.timings.mel_s,
                    "encode": result.timings.encode_s,
                    "decode": result.timings.decode_s,
                    "detok": result.timings.detok_s,
                },
            }
        )
    corpus_wall_s = time.perf_counter() - corpus_wall_t0
    prefetch_pool.shutdown(wait=False)

    total_duration = sum(pf["duration_s"] for pf in per_file)
    total_wall = sum(pf["wall_s"] for pf in per_file)
    # overall_rtf uses corpus_wall_s (the actual wall clock spanning the
    # whole loop), not sum(per-file wall_s): with the CPU/GPU overlap above,
    # per-file wall_s is now just each file's GPU-critical-path time (its
    # chunks were already prefetched), so summing it would understate real
    # elapsed time by hiding the one file's worth of CPU prep that can't be
    # overlapped with anything (the last file's tail, or the whole first
    # file if the corpus has only one). corpus_wall_s is the same "one wall
    # clock over the whole run" metric --mode dual already treats as
    # authoritative, applied here to single/worker mode too.
    return {
        "batch_size": args.batch_size,
        "use_kernels": args.use_kernels,
        "chunking": args.chunking,
        "ngram_ban": args.ngram_ban,
        "timestamps": args.timestamps,
        "retry_ladder": not args.no_retry_ladder,
        "model_load_s": load_s,
        "per_file": per_file,
        "total_audio_duration_s": total_duration,
        "total_wall_s": total_wall,
        "overall_rtf": corpus_wall_s / total_duration if total_duration > 0 else float("nan"),
        "corpus_wall_s": corpus_wall_s,
    }


def _mode_single_or_worker(args):
    result = _run_single_or_worker(args)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}: overall_rtf={result['overall_rtf']:.5f}")


def _balanced_split(files: list[str]) -> tuple[list[str], list[str]]:
    """Longest-processing-time-first greedy split by audio duration, not
    file order -- a plain alternating split badly imbalances a small,
    duration-skewed corpus like this one (5 files, one 3x longer than the
    two shortest combined), which would bottleneck the whole dual-GPU
    wall clock on a single overloaded worker and understate what dual-GPU
    actually buys. Only soundfile is needed for duration probing, so the
    dual-mode parent process still never imports torch.
    """
    import soundfile as sf

    durations = [(f, sf.info(f).frames / sf.info(f).samplerate) for f in files]
    durations.sort(key=lambda x: -x[1])
    shard_a: list[str] = []
    shard_b: list[str] = []
    dur_a = dur_b = 0.0
    for f, dur in durations:
        if dur_a <= dur_b:
            shard_a.append(f)
            dur_a += dur
        else:
            shard_b.append(f)
            dur_b += dur
    return shard_a, shard_b


def _mode_dual(args):
    files = _list_corpus(args.corpus_dir, args.files.split(",") if args.files else None)
    if len(files) < 2:
        print(
            "WARNING: dual mode with <2 files leaves one GPU idle; "
            "still valid, just not a meaningful dual-GPU comparison.",
            file=sys.stderr,
        )
    shard_a, shard_b = _balanced_split(files)

    out_a = args.out + ".gpu0.json"
    out_b = args.out + ".gpu1.json"

    script = str(Path(__file__).resolve())
    common = [
        sys.executable, script, "--mode", "worker",
        "--corpus-dir", args.corpus_dir,
        "--checkpoint", args.checkpoint,
        "--batch-size", str(args.batch_size),
        "--chunking", args.chunking,
    ]
    if args.use_kernels:
        common.append("--use-kernels")
    if args.ngram_ban:
        common.append("--ngram-ban")
    if args.timestamps:
        common.append("--timestamps")
    if args.no_retry_ladder:
        common.append("--no-retry-ladder")

    def _env_for(gpu: int) -> dict:
        env = dict(os.environ)
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        return env

    procs = []
    if shard_a:
        procs.append(
            subprocess.Popen(
                common + ["--files", ",".join(os.path.basename(f) for f in shard_a), "--out", out_a],
                env=_env_for(0),
            )
        )
    if shard_b:
        procs.append(
            subprocess.Popen(
                common + ["--files", ",".join(os.path.basename(f) for f in shard_b), "--out", out_b],
                env=_env_for(1),
            )
        )

    wall_t0 = time.perf_counter()
    for p in procs:
        p.wait()
    dual_wall_s = time.perf_counter() - wall_t0
    for p in procs:
        if p.returncode != 0:
            raise RuntimeError(f"worker exited with code {p.returncode}")

    merged_per_file = []
    total_duration = 0.0
    for out in (out_a, out_b):
        if not os.path.exists(out):
            continue
        with open(out) as f:
            data = json.load(f)
        merged_per_file.extend(data["per_file"])
        total_duration += data["total_audio_duration_s"]

    result = {
        "mode": "dual",
        "batch_size": args.batch_size,
        "per_file": merged_per_file,
        "total_audio_duration_s": total_duration,
        # The one number that matters for dual-GPU: wall time from "both
        # workers started" to "both workers finished", over the WHOLE
        # corpus -- not a sum/average of two independently-measured RTFs.
        "dual_wall_s": dual_wall_s,
        "overall_rtf": dual_wall_s / total_duration if total_duration > 0 else float("nan"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"wrote {args.out}: dual-GPU overall_rtf={result['overall_rtf']:.5f} "
        f"(one wall clock over the whole corpus, {dual_wall_s:.1f}s)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "dual", "worker"], default="single")
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--files", default=None, help="comma-separated filenames to restrict to")
    ap.add_argument("--use-kernels", action="store_true", help="enable Agent C's K4/K5 kernels")
    ap.add_argument(
        "--chunking", choices=["hard", "stride", "vad"], default="vad",
        help="vad (default since 2026-08-10) = Silero-VAD speech windows greedily packed to <=30s, "
             "closed on silence gaps; hard = non-overlapping 30s chunks; stride = 30s window, 5s "
             "stride each side, LCS merge",
    )
    ap.add_argument(
        "--ngram-ban", action="store_true",
        help="enable K5's no_repeat_ngram_size=3 ban (default OFF in notimestamps mode -- see "
             "decode.py docstring; measured per-mode in timestamps mode, M5)",
    )
    ap.add_argument(
        "--timestamps", action="store_true",
        help="decode with openai-whisper's timestamp grammar (M5); also emits a "
             "{Start,End,Text} segments list per file, matching Castria's transcript format",
    )
    ap.add_argument(
        "--no-retry-ladder", action="store_true",
        help="disable the M6 temperature-retry ladder (default ON) -- for A/B comparison "
             "against the pre-ladder RTF/quality numbers",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode in ("single", "worker"):
        _mode_single_or_worker(args)
    elif args.mode == "dual":
        _mode_dual(args)


if __name__ == "__main__":
    main()
