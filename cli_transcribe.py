#!/usr/bin/env python3
"""One-shot CLI transcription. Loads the model, transcribes exactly one file,
prints the transcript to stdout, writes `{text, segments}` JSON to --out.

Used by the Docker image's `transcribe` entrypoint mode (see entrypoint.sh);
also runnable directly outside Docker. For corpus benchmarking see
bench/bench_e2e.py instead -- this script loads weights fresh every
invocation, which is fine for one file, wasteful for many.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from whisper_rocm import kernel_bridge  # noqa: E402
from whisper_rocm import ops as whisper_ops  # noqa: E402
from whisper_rocm import pipeline as whisper_pipeline  # noqa: E402
from whisper_rocm import tokenizer as whisper_tokenizer  # noqa: E402
from whisper_rocm import weights as whisper_weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio_file")
    ap.add_argument("--out", default="/tmp/segments.json", help="Where to write {text, segments} JSON")
    ap.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "WHISPER_CHECKPOINT", os.path.expanduser("~/.cache/whisper/large-v3-turbo.pt")
        ),
    )
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("WHISPER_BATCH_SIZE", "16")))
    ap.add_argument(
        "--chunking",
        default=os.environ.get("WHISPER_CHUNKING", "vad"),
        choices=["vad", "hard", "stride"],
    )
    ap.add_argument("--response-format", default="verbose_json", choices=["text", "json", "verbose_json"])
    args = ap.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"error: no such file: {args.audio_file}", file=sys.stderr)
        raise SystemExit(1)

    kernel_bridge.register()
    whisper_ops.USE_KERNELS = True

    device = torch.device("cuda:0")
    t0 = time.perf_counter()
    w = whisper_weights.load_weights(args.checkpoint, device=device)
    tok = whisper_tokenizer.Tokenizer(n_vocab=w.dims.n_vocab)
    torch.cuda.synchronize(device)
    print(
        f"[cli] model loaded in {time.perf_counter() - t0:.1f}s on {torch.cuda.get_device_name(device)}",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    result = whisper_pipeline.transcribe_file(
        args.audio_file, w, tok, device,
        batch_size=args.batch_size, chunking=args.chunking, timestamps=True,
    )
    wall_s = time.perf_counter() - t0
    rtf = wall_s / max(result.duration_s, 1e-6)
    print(
        f"[cli] transcribed {result.duration_s:.1f}s audio in {wall_s:.1f}s wall (RTF {rtf:.4f})",
        file=sys.stderr,
    )

    print(result.text)

    segments = [{"start": s["Start"], "end": s["End"], "text": s["Text"]} for s in result.segments]
    with open(args.out, "w") as f:
        json.dump({"text": result.text, "segments": segments}, f, indent=2)
    print(f"[cli] segments written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
