#!/usr/bin/env bash
#
# sandsim benchmark harness.
#
# Builds (best effort) and runs every implementation's headless `--bench` mode,
# parses the one-line RESULT each prints, renders a Markdown comparison table
# sorted by throughput, and verifies that all "scalar" rule implementations
# agree on their checksum (a strict cross-language correctness check). GPU
# implementations form their own rule group; their checksum can vary run to run
# (atomic contention) so only their conserved sand count is reported.
#
# Usage: tools/bench.sh [steps] [width] [height]   (defaults: 1000 400 300)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${1:-1000}"
W="${2:-400}"
H="${3:-300}"

RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

echo "sandsim benchmark  (steps=$STEPS  grid=${W}x${H})"
echo "building + running implementations..."

# try_build <dir> <command...> : build quietly, never fail the harness.
try_build() { ( cd "$ROOT/$1" && shift && "$@" ) >/dev/null 2>&1 || true; }

# collect <name> <dir> <run-command...> : run --bench, capture the RESULT line.
collect() {
    local name="$1" dir="$2"; shift 2
    local line
    line=$( cd "$ROOT/$dir" 2>/dev/null && "$@" --bench "$STEPS" "$W" "$H" 2>/dev/null \
            | grep -m1 '^RESULT' )
    if [ -n "$line" ]; then
        echo "$line" >> "$RESULTS_FILE"
        printf '  ok    %s\n' "$name"
    else
        printf '  skip  %s (not built, or no GPU/display)\n' "$name"
    fi
}

# Build benchmarkable implementations (quietly; missing toolchains are skipped).
command -v g++   >/dev/null 2>&1 && try_build cpp    make sandsim_scalar_sb
command -v cc    >/dev/null 2>&1 && try_build c      make
command -v cargo >/dev/null 2>&1 && try_build rust   cargo build --release --offline
command -v zig   >/dev/null 2>&1 && try_build zig    zig build-exe sandsim.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim
pkg-config --exists glfw3 glew gl 2>/dev/null && try_build opengl make
command -v glslc >/dev/null 2>&1 && try_build vulkan make sandsim_vulkan_compute
command -v hipcc >/dev/null 2>&1 && try_build hip    make

# Run each (vulkan must run from its own dir so it finds shaders/sand.comp.spv).
collect cpp_scalar_sb cpp    ./sandsim_scalar_sb
collect c             c      ./sandsim
collect rust          rust   ./target/release/sandsim
collect zig           zig    ./sandsim
collect opengl        opengl ./sandsim_gl
collect vulkan        vulkan ./sandsim_vulkan_compute
collect hip           hip    ./sandsim_hip

if [ ! -s "$RESULTS_FILE" ]; then
    echo "No implementations ran. Build something first (make all)." >&2
    exit 1
fi

# --- table + verification (awk over the collected RESULT lines) -------------
echo
awk '
function field(line, key,   m) {
    if (match(line, key "=[^ ]+")) { m = substr(line, RSTART, RLENGTH); sub(key "=", "", m); return m }
    return "-"
}
{
    impl[NR]   = field($0, "impl")
    rule[NR]   = field($0, "rule")
    mcells[NR] = field($0, "mcells_per_s") + 0
    csum[NR]   = field($0, "checksum")
    sand[NR]   = field($0, "sand")
    n = NR
}
END {
    # selection sort by mcells descending
    for (i = 1; i <= n; i++) order[i] = i
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (mcells[order[j]] > mcells[order[i]]) { t = order[i]; order[i] = order[j]; order[j] = t }

    print "| Implementation | Rule | Mcells/s | Checksum | Sand |"
    print "|----------------|------|---------:|----------|-----:|"
    for (k = 1; k <= n; k++) {
        i = order[k]
        printf "| %-14s | %-4s | %8.2f | %s | %s |\n", impl[i], rule[i], mcells[i], csum[i], sand[i]
    }

    # verify scalar-group checksums all match
    ref = ""; ok = 1; cnt = 0
    for (i = 1; i <= n; i++) if (rule[i] == "scalar") {
        cnt++
        if (ref == "") ref = csum[i]
        else if (csum[i] != ref) ok = 0
    }
    print ""
    if (cnt == 0) {
        print "No scalar-rule implementations ran; cross-language checksum check skipped."
    } else if (ok) {
        printf "scalar-rule checksum agreement: PASS (%d implementations share %s)\n", cnt, ref
    } else {
        print "scalar-rule checksum agreement: FAIL (implementations disagree)"
        exit 3
    }
}
' "$RESULTS_FILE"
