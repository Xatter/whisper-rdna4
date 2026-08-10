#!/usr/bin/env bash
# Docker entrypoint. Two modes:
#   serve                          -- start the API server (default)
#   transcribe <file> [out.json]   -- one-shot CLI transcription, then exit
# Anything else is exec'd as-is (escape hatch, e.g. `docker run ... bash`).
set -euo pipefail

CHECKPOINT="${WHISPER_CHECKPOINT:-$HOME/.cache/whisper/large-v3-turbo.pt}"

download_checkpoint_if_missing() {
  if [ ! -f "$CHECKPOINT" ]; then
    echo "[entrypoint] checkpoint not found at $CHECKPOINT -- downloading large-v3-turbo (~1.6GB, one time)" >&2
    mkdir -p "$(dirname "$CHECKPOINT")"
    python -c "import whisper; whisper.load_model('large-v3-turbo')"
  fi
}

mode="${1:-serve}"

case "$mode" in
  serve)
    download_checkpoint_if_missing
    exec python /app/serve.py --port "${PORT:-8100}"
    ;;
  transcribe)
    shift
    if [ "$#" -lt 1 ]; then
      echo "usage: docker run ... whisper-rdna4 transcribe <audio-file> [output.json]" >&2
      exit 1
    fi
    audio_file="$1"
    out_json="${2:-/tmp/segments.json}"
    download_checkpoint_if_missing
    exec python /app/cli_transcribe.py "$audio_file" --out "$out_json"
    ;;
  *)
    exec "$@"
    ;;
esac
