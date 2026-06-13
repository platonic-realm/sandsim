#!/usr/bin/env bash
#
# sandsim chunked-streaming-world cross-check.
#
# Runs every language port of the streamed world (--bench), prints a table, and
# verifies they all agree on the whole-world checksum and conserved per-material
# counts -- and that every run reports conserved=yes (the streaming round-trip
# is lossless).
#
# Usage: tools/bench_world.sh [steps] [wch] [hch]   (default: 200 4 4)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${1:-200}"
WCH="${2:-4}"
HCH="${3:-4}"

RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

echo "sandsim world cross-check  (steps=$STEPS  world=${WCH}x${HCH} chunks of 64)"
echo "building + running world implementations..."

try_build() { ( cd "$ROOT/$1" && shift && "$@" ) >/dev/null 2>&1 || true; }
collect() {
    local name="$1" dir="$2"; shift 2
    local line
    line=$( cd "$ROOT/$dir" 2>/dev/null && "$@" --bench "$STEPS" "$WCH" "$HCH" 2>/dev/null \
            | grep -m1 '^RESULT' )
    if [ -n "$line" ]; then echo "$line" >> "$RESULTS_FILE"; printf '  ok    %s\n' "$name"
    else printf '  skip  %s (not built)\n' "$name"; fi
}

command -v g++   >/dev/null 2>&1 && try_build cpp  make sandsim_world
command -v cc    >/dev/null 2>&1 && try_build c    make sandsim_world
command -v cargo >/dev/null 2>&1 && try_build rust cargo build --release --offline
command -v zig   >/dev/null 2>&1 && try_build zig  zig build-exe sandsim_world.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim_world

collect cpp_world    cpp    ./sandsim_world
collect c_world      c      ./sandsim_world
collect rust_world   rust   ./target/release/sandsim_world
collect zig_world    zig    ./sandsim_world

if [ ! -s "$RESULTS_FILE" ]; then echo "No world implementations ran. Try: make world" >&2; exit 1; fi

echo
awk '
function field(line, key,   m) {
    if (match(line, key "=[^ ]+")) { m = substr(line, RSTART, RLENGTH); sub(key "=", "", m); return m }
    return "-"
}
{
    impl[NR] = field($0,"impl"); csum[NR] = field($0,"checksum"); cons[NR] = field($0,"conserved")
    res[NR] = field($0,"resident_max"); dw[NR] = field($0,"disk_writes"); dr[NR] = field($0,"disk_reads")
    cnt[NR] = field($0,"wall") "/" field($0,"sand") "/" field($0,"water") "/" field($0,"gas")
    n = NR
}
END {
    print "| Implementation | Checksum         | wall/sand/water/gas | resident_max | disk w/r | conserved |"
    print "|----------------|------------------|---------------------|-------------:|----------|-----------|"
    for (i = 1; i <= n; i++)
        printf "| %-14s | %s | %-19s | %12s | %s/%s | %-9s |\n",
               impl[i], csum[i], cnt[i], res[i], dw[i], dr[i], cons[i]
    ok = 1
    for (i = 1; i <= n; i++) if (cons[i] != "yes") ok = 0
    for (i = 2; i <= n; i++) if (csum[i] != csum[1] || cnt[i] != cnt[1]) ok = 0
    print ""
    if (ok) printf "world cross-language agreement: PASS (%d implementations share %s, all conserved)\n", n, csum[1]
    else { print "world cross-language agreement: FAIL"; exit 3 }
}
' "$RESULTS_FILE"
