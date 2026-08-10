# whisper-rdna4 -- fast Whisper large-v3-turbo on AMD RDNA4 (gfx1201) GPUs.
#
# Base image's ROCm major (7.2) must match the hipcc used to compile
# kernels/*.hip below -- this is the version-matching trap the README's
# "Requirements" section warns about (mixed torch/ROCm majors fail with HSA
# symbol errors at import/launch time, not at build time). If you bump this
# base image to a different ROCm major, you're on your own for retesting;
# nothing here checks it for you.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

WORKDIR /app

# Repo deps not already in the base image. torch/torchvision/torchaudio ship
# with rocm/pytorch, ROCm-matched -- do not reinstall them here (that's
# exactly the trap above: a pip-resolved torch could silently be a different
# ROCm build, or CUDA, if PyPI ever gets consulted).
RUN pip install --no-cache-dir \
        openai-whisper \
        soundfile \
        scipy \
        tiktoken \
        numpy \
        silero-vad \
        onnxruntime \
        fastapi \
        uvicorn \
        python-multipart

COPY . /app
RUN chmod +x /app/entrypoint.sh

# Compile the fused decode kernels. No GPU needed at build time --
# hipcc cross-compiles directly for --offload-arch=gfx1201.
RUN cd kernels && bash build.sh

ENV WHISPER_CHECKPOINT=/root/.cache/whisper/large-v3-turbo.pt \
    PORT=8100 \
    HIP_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1

# Checkpoint (~1.6GB) is NOT fetched at build time -- no network guarantee
# during `docker build`, and baking it in would make every rebuild re-pull
# it. entrypoint.sh downloads it on first `docker run` if missing; mount a
# volume at $WHISPER_CHECKPOINT's parent dir (see README) so it survives
# container recreation instead of re-downloading every run.

EXPOSE 8100

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
