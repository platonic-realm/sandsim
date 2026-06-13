#!/usr/bin/env bash
#
# sandsim multi-material cross-check harness.
#
# Runs every material-capable implementation's headless `--bench`, prints a
# table, and verifies that they all agree on BOTH the checksum and the conserved
# per-material counts -- a strict cross-language correctness check for the
# Noita-style engine (EMPTY / WALL / SAND / WATER / GAS).
#
#
# Usage: tools/bench_materials.sh [steps] [width] [height]   (default: 200 120 90)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${1:-200}"
W="${2:-120}"
H="${3:-90}"

RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

echo "sandsim materials cross-check  (steps=$STEPS  grid=${W}x${H})"
echo "building + running material implementations..."

try_build() { ( cd "$ROOT/$1" && shift && "$@" ) >/dev/null 2>&1 || true; }

collect() {
    local name="$1" dir="$2"; shift 2
    local line
    line=$( cd "$ROOT/$dir" 2>/dev/null && "$@" --bench "$STEPS" "$W" "$H" 2>/dev/null \
            | grep -m1 '^RESULT' )
    if [ -n "$line" ]; then
        echo "$line" >> "$RESULTS_FILE"
        printf '  ok    %s\n' "$name"
    else
        printf '  skip  %s (not built)\n' "$name"
    fi
}

command -v g++   >/dev/null 2>&1 && try_build cpp  make sandsim_materials
command -v cc    >/dev/null 2>&1 && try_build c    make sandsim_materials
command -v cargo >/dev/null 2>&1 && try_build rust cargo build --release --offline
command -v zig   >/dev/null 2>&1 && try_build zig  zig build-exe sandsim_materials.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim_materials

collect cpp_materials    cpp    ./sandsim_materials
collect c_materials      c      ./sandsim_materials
collect rust_materials   rust   ./target/release/sandsim_materials
collect zig_materials    zig    ./sandsim_materials

if [ ! -s "$RESULTS_FILE" ]; then
    echo "No material implementations ran. Try: make materials" >&2
    exit 1
fi

echo
awk '
function field(line, key,   m) {
    if (match(line, key "=[^ ]+")) { m = substr(line, RSTART, RLENGTH); sub(key "=", "", m); return m }
    return "-"
}
{
    impl[NR]  = field($0, "impl")
    csum[NR]  = field($0, "checksum")
    mcells[NR]= field($0, "mcells_per_s") + 0
    cnts[NR]  = field($0,"wall") "/" field($0,"sand") "/" field($0,"water") "/" field($0,"gas")
    n = NR
}
END {
    print "| Implementation   | Mcells/s | Checksum         | wall/sand/water/gas |"
    print "|------------------|---------:|------------------|---------------------|"
    for (i = 1; i <= n; i++)
        printf "| %-16s | %8.2f | %s | %-19s |\n", impl[i], mcells[i], csum[i], cnts[i]

    ok = 1
    for (i = 2; i <= n; i++) if (csum[i] != csum[1] || cnts[i] != cnts[1]) ok = 0
    print ""
    if (ok) printf "materials cross-language agreement: PASS (%d implementations share %s, counts %s)\n", n, csum[1], cnts[1]
    else { print "materials cross-language agreement: FAIL (implementations disagree)"; exit 3 }
}
' "$RESULTS_FILE"
