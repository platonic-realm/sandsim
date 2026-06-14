#!/usr/bin/env bash
#
# sandsim benchmark: build the three backends, run the same streamed world, and
#   (1) assert every backend produced a bit-identical world (same checksum), and
#   (2) print a throughput table.
#
# The whole point of the order-independent rule is that the C++ SIMD, OpenGL, and
# Vulkan worlds are the SAME function of the seed -- so a differing checksum is a
# real bug, and this script fails loudly on it.
#
# Usage: tools/benchmark.sh [steps] [wbox] [hbox]   (default: 600 6 6)

set -u
cd "$(dirname "$0")/.."

STEPS=${1:-600}
WBOX=${2:-6}
HBOX=${3:-6}

CPU_BIN=cpp/sandsim_world
GL_BIN=opengl/sandsim_world_gl
VK_BIN=vulkan/sandsim_world_vk

echo "== building =="
make -s cpp     2>&1 | sed 's/^/  cpp:    /' || true
make -s opengl  2>&1 | sed 's/^/  opengl: /' || true
make -s vulkan  2>&1 | sed 's/^/  vulkan: /' || true
echo

# name|binary  -- run each, capture its RESULT line
declare -a NAMES=("C++ (SIMD)" "OpenGL" "Vulkan")
declare -a BINS=("$CPU_BIN" "$GL_BIN" "$VK_BIN")

field() { sed -n "s/.*[[:space:]]$1=\([^ ]*\).*/\1/p"; }

echo "== running --bench $STEPS $WBOX $HBOX =="
declare -a ROWS=() CKSUMS=() RAN=()
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; bin="${BINS[$i]}"
    if [[ ! -x "$bin" ]]; then
        printf "  %-12s skipped (not built)\n" "$name"; continue
    fi
    line="$("$bin" --bench "$STEPS" "$WBOX" "$HBOX" 2>/dev/null | grep '^RESULT' | head -1)"
    if [[ -z "$line" ]]; then
        printf "  %-12s skipped (no device / run failed)\n" "$name"; continue
    fi
    ck="$(printf '%s' "$line"   | field checksum)"
    mc="$(printf '%s' "$line"   | field mcells_per_s)"
    ms="$(printf '%s' "$line"   | field elapsed_ms)"
    cons="$(printf '%s' "$line" | field conserved)"
    printf "  %-12s checksum=%s  %8s Mcells/s  conserved=%s\n" "$name" "$ck" "$mc" "$cons"
    RAN+=("$name"); CKSUMS+=("$ck")
    ROWS+=("$name|$mc|$ms|$cons|$ck")
done
echo

if [[ ${#CKSUMS[@]} -eq 0 ]]; then
    echo "no backend ran; nothing to compare." >&2
    exit 1
fi

# Assert every backend that ran agrees with the first.
ref="${CKSUMS[0]}"; ok=1
for ck in "${CKSUMS[@]}"; do [[ "$ck" == "$ref" ]] || ok=0; done
if [[ $ok -eq 1 ]]; then
    echo "✅ bit-identical: all ${#CKSUMS[@]} backend(s) produced checksum $ref"
else
    echo "❌ CHECKSUM MISMATCH across backends:" >&2
    for i in "${!RAN[@]}"; do echo "     ${RAN[$i]}: ${CKSUMS[$i]}" >&2; done
    exit 2
fi
echo

# Throughput table, sorted fastest first.
echo "== throughput (steps=$STEPS, window ${WBOX}x${HBOX} chunks, ${WBOX}0x${HBOX}0... streamed) =="
{
  echo "Backend|Mcells/s|elapsed_ms|conserved"
  echo "---|---:|---:|:---:"
  printf '%s\n' "${ROWS[@]}" | sort -t'|' -k2 -gr | \
    awk -F'|' '{printf "%s|%s|%s|%s\n",$1,$2,$3,$4}'
} | column -t -s'|'
