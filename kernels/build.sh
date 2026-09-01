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

# hipcc location differs by ROCm packaging:
#   ROCm <= 7.2  -- system install, /opt/rocm/bin/hipcc
#   ROCm 10      -- pip/wheel SDK (_rocm_sdk_core), hipcc only on PATH
# Resolve in that order of specificity; $HIPCC overrides everything.
if [ -z "${HIPCC:-}" ]; then
  if command -v hipcc >/dev/null 2>&1; then
    HIPCC=$(command -v hipcc)
  elif [ -x "${ROCM_PATH:-/opt/rocm}/bin/hipcc" ]; then
    HIPCC="${ROCM_PATH:-/opt/rocm}/bin/hipcc"
  else
    echo "hipcc not found: not on PATH and not at \${ROCM_PATH:-/opt/rocm}/bin/hipcc" >&2
    echo "set HIPCC=/path/to/hipcc to override" >&2
    exit 1
  fi
fi
ARCH=gfx1201
OUT=libwhisper_kernels.so

# ROCm 10's pip/wheel SDK (_rocm_sdk_core) ships only libamdhip64.so.<N>, the
# SONAME -- not the unversioned libamdhip64.so linker name -- so the link step
# fails with
#   ld.lld: error: cannot open .../libamdhip64.so: No such file or directory
# hipcc hands the linker that absolute path, so -L on a shim dir does not help;
# the name has to exist next to the SONAME. Create it. Both realistic ROCm 10
# installs are writable here (root in a container, the user's own venv on bare
# metal). `pip install rocm[devel]` also supplies the name, at the cost of a
# whole development tree for one symlink.
# A system ROCm (<= 7.2, /opt/rocm) ships both names, so none of this runs.
HIPCONFIG="$(dirname "$HIPCC")/hipconfig"
if [ -x "$HIPCONFIG" ]; then
  HIP_LIB="$("$HIPCONFIG" --path 2>/dev/null)/lib"
  if [ -d "$HIP_LIB" ] && [ ! -e "$HIP_LIB/libamdhip64.so" ]; then
    SONAME=$(ls "$HIP_LIB"/libamdhip64.so.* 2>/dev/null | head -1)
    if [ -n "$SONAME" ]; then
      if ln -sf "$SONAME" "$HIP_LIB/libamdhip64.so" 2>/dev/null; then
        echo "note: added missing linker name $HIP_LIB/libamdhip64.so -> $(basename "$SONAME")"
      else
        echo "$HIP_LIB is not writable and has no libamdhip64.so linker name." >&2
        echo "Run: ln -s $SONAME $HIP_LIB/libamdhip64.so   (or pip install 'rocm[devel]')" >&2
        exit 1
      fi
    fi
  fi
fi

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
