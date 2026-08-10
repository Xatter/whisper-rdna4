#!/usr/bin/env python3
"""FastAPI transcription server wrapping whisper_rocm's pipeline (DESIGN.md S5).

Response shape mirrors the OpenAI Whisper API's `/v1/audio/transcriptions`
endpoint (`response_format=json` -> `{"text"}`; `verbose_json` ->
`{"text", "segments": [{id, seek, start, end, text, tokens, temperature,
avg_logprob, compression_ratio, no_speech_prob}], "language"}`), so this is a
drop-in swap for anything already speaking to that endpoint shape. Common
non-OpenAI form fields some clients send (model, dtype, device, batch_size,
chunk_length, ...) are accepted and ignored -- this instance's own
WHISPER_BATCH_SIZE / WHISPER_CHUNKING env vars govern actual behavior.

One process per GPU: set HIP_VISIBLE_DEVICES *before* this process starts --
GPU selection binds at HIP-runtime-init time inside `import torch`, so it
cannot be changed after the interpreter is up. If you have more than one GPU
and don't set it, everything defaults to device 0; mask any non-target GPU
(e.g. an iGPU) the same way.

Usage:
    HIP_VISIBLE_DEVICES=0 PORT=8100 python serve.py
    HIP_VISIBLE_DEVICES=0 python serve.py --port 8101
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402

from whisper_rocm import kernel_bridge  # noqa: E402
from whisper_rocm import ops as whisper_ops  # noqa: E402
from whisper_rocm import pipeline as whisper_pipeline  # noqa: E402
from whisper_rocm import tokenizer as whisper_tokenizer  # noqa: E402
from whisper_rocm import weights as whisper_weights  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("whisper-rdna4-serve")

DEFAULT_CHECKPOINT = os.environ.get(
    "WHISPER_CHECKPOINT", os.path.expanduser("~/.cache/whisper/large-v3-turbo.pt")
)
# Pipeline's own current defaults (whisper_rocm/pipeline.py transcribe_file
# signature): chunking="vad", retry ladder on, ngram ban off. batch_size 16
# is a reasonable default for one GPU (see README's benchmark numbers).
DEFAULT_BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "16"))
DEFAULT_CHUNKING = os.environ.get("WHISPER_CHUNKING", "vad")

app = FastAPI(title="whisper-rdna4", version="0.1.0")


class _State:
    weights = None
    tok = None
    device: "torch.device | None" = None
    gpu_index_env: str = ""
    gpu_name: str = ""
    load_s: float = 0.0
    lock: "asyncio.Lock | None" = None
    started_at: float = 0.0
    request_count: int = 0
    ready: bool = False


state = _State()


@app.on_event("startup")
async def _startup() -> None:
    state.started_at = time.time()
    state.gpu_index_env = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        logger.warning(
            "HIP_VISIBLE_DEVICES not set; defaulting to device 0. If your "
            "system has an iGPU or multiple GPUs, set it explicitly."
        )

    logger.info("Registering kernel_bridge (K4 self/cross-attn decode, K5 logits) ...")
    kernel_bridge.register()
    whisper_ops.USE_KERNELS = True  # only K4/K5 beat torch on the tested shapes; K1-3 stay off.

    state.device = torch.device("cuda:0")  # index WITHIN this process's HIP_VISIBLE_DEVICES
    state.lock = asyncio.Lock()

    t0 = time.perf_counter()
    logger.info(
        "Loading weights from %s onto %s (HIP_VISIBLE_DEVICES=%s) ...",
        DEFAULT_CHECKPOINT, state.device, state.gpu_index_env,
    )
    state.weights = whisper_weights.load_weights(DEFAULT_CHECKPOINT, device=state.device)
    state.tok = whisper_tokenizer.Tokenizer(n_vocab=state.weights.dims.n_vocab)
    torch.cuda.synchronize(state.device)
    state.load_s = time.perf_counter() - t0
    state.gpu_name = torch.cuda.get_device_name(state.device)
    state.ready = True
    logger.info(
        "Model loaded in %.1fs on %s (HIP_VISIBLE_DEVICES=%s). chunking=%s batch_size=%d",
        state.load_s, state.gpu_name, state.gpu_index_env, DEFAULT_CHUNKING, DEFAULT_BATCH_SIZE,
    )


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "ok" if state.ready else "loading",
            "model": "large-v3-turbo",
            "checkpoint": DEFAULT_CHECKPOINT,
            "gpu_index": state.gpu_index_env,
            "gpu_name": state.gpu_name,
            "kernels_enabled": bool(whisper_ops.USE_KERNELS),
            "chunking_default": DEFAULT_CHUNKING,
            "batch_size_default": DEFAULT_BATCH_SIZE,
            "load_seconds": round(state.load_s, 2),
            "uptime_seconds": round(time.time() - state.started_at, 1) if state.started_at else 0,
            "requests_served": state.request_count,
            "busy": state.lock.locked() if state.lock else False,
        }
    )


def _run_transcribe(path: str):
    return whisper_pipeline.transcribe_file(
        path,
        state.weights,
        state.tok,
        state.device,
        batch_size=DEFAULT_BATCH_SIZE,
        chunking=DEFAULT_CHUNKING,
        enable_ngram_ban=False,
        timestamps=True,  # segments are the point -- always compute them.
        retry_ladder_enabled=True,
    )


def _build_response(result, response_format: str):
    if response_format == "text":
        return PlainTextResponse(result.text, media_type="text/plain; charset=utf-8")

    if response_format == "verbose_json":
        segments = []
        for idx, seg in enumerate(result.segments):
            segments.append(
                {
                    "id": idx,
                    "seek": 0,
                    "start": seg["Start"],
                    "end": seg["End"],
                    "text": seg["Text"],
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                }
            )
        return JSONResponse({"text": result.text, "segments": segments, "language": "en"})

    # default "json" (also the fallback for any unrecognized format)
    return JSONResponse({"text": result.text})


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
    task: Optional[str] = Form("transcribe"),
    prompt: Optional[str] = Form(None),
    timestamp_type: Optional[str] = Form(None),
    stabilize: Optional[bool] = Form(None),
    demucs: Optional[bool] = Form(None),
    vad: Optional[bool] = Form(None),
    vad_threshold: Optional[float] = Form(None),
    # Non-OpenAI fields some clients send; accepted-and-ignored for
    # compatibility. This instance's own DEFAULT_BATCH_SIZE / DEFAULT_CHUNKING
    # govern actual behavior, not caller-supplied values.
    model: Optional[str] = Form(None),
    dtype: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    batch_size: Optional[int] = Form(None),
    chunk_length: Optional[int] = Form(None),
):
    if not state.ready:
        return JSONResponse({"error": "model still loading"}, status_code=503)

    suffix = Path(file.filename or "upload.audio").suffix or ".mp3"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="whisper-upload-")
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        await file.close()

        async with state.lock:  # serialize: one transcription at a time per instance
            t0 = time.perf_counter()
            result = await asyncio.to_thread(_run_transcribe, tmp_path)
            wall_s = time.perf_counter() - t0
            state.request_count += 1
        logger.info(
            "Transcribed %s (%.1fs audio) in %.1fs wall (%d chunks, %d segments)",
            file.filename, result.duration_s, wall_s, result.num_chunks, len(result.segments),
        )
        return _build_response(result, response_format)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", os.environ.get("WHISPER_PORT", "8100"))),
    )
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
