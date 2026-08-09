#!/usr/bin/env bash
# build.sh -- builds every kernels/*.hip into a single libwhisper_kernels.so.
# Wildcards the source list so this doesn't break as Agent C's K4
# (attention_decode.hip) and K5 (logits.hip) land.
#
# Usage: ./build.sh            (release build)
#        ./build.sh --resource-usage   (adds -Rpass-analysis=kernel-resource-usage,
#                                        VGPR counts go to build_resource_usage.log)
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=/opt/rocm/bin/hipcc
ARCH=gfx1201
OUT=libwhisper_kernels.so

SOURCES=(*.hip)
if [ ! -e "${SOURCES[0]}" ]; then
  echo "no .hip sources found in $(pwd)" >&2
  exit 1
fi

FLAGS=(-fPIC -shared --offload-arch="$ARCH" -O2 -mno-wavefrontsize64)

if [[ "${1:-}" == "--resource-usage" ]]; then
  FLAGS+=(-Rpass-analysis=kernel-resource-usage)
  echo "Building ${SOURCES[*]} -> $OUT (with -Rpass-analysis=kernel-resource-usage)"
  "$HIPCC" "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT" 2> build_resource_usage.log
  echo "Resource-usage log: $(pwd)/build_resource_usage.log"
  grep -E "Function Name|VGPR|Occupancy" build_resource_usage.log || true
else
  echo "Building ${SOURCES[*]} -> $OUT"
  "$HIPCC" "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT"
fi

echo "OK: $(pwd)/$OUT"
