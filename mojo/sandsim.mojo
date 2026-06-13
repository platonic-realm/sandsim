# sandsim - Mojo implementation
#
# Canonical "scalar" falling-sand rule, a direct port of
# cpp/sandsim_scalar_sb.cpp: scanning bottom-to-top, sand falls straight down,
# else down-left, else down-right, else stays.
#
# Modes:
#   (default)                 dependency-free ASCII animation in the terminal.
#   --bench [steps] [w] [h]   headless: deterministic seed, time the update
#                             loop, print one RESULT line whose checksum matches
#                             every other scalar-rule implementation.
#
# NOTE: this host has no Mojo toolchain installed, so this file is written to
# spec and not compiled here. It targets the Mojo standard library and uses the
# same fixed-width integer arithmetic as the other ports so the --bench checksum
# is identical (31128ca3d1fcadc6 at the default 400x300 / 1000 steps). Run with:
#   mojo run sandsim.mojo --bench 1000 400 300

from sys import argv
from time import perf_counter_ns, sleep
from collections import List

alias SAND: UInt8 = 1
alias EMPTY: UInt8 = 0

# ---------------------------------------------------------------------------
# Shared, language-independent helpers (must match the other implementations).
# ---------------------------------------------------------------------------

# Deterministic per-cell seed (~30% sand). All arithmetic is masked to 32 bits
# so the result matches C's `(uint32_t)` wraparound exactly.
fn seed_cell(x: Int, y: Int) -> UInt8:
    var h: UInt64 = (UInt64(x) * 374761393) & 0xFFFFFFFF
    h = (h + ((UInt64(y) * 668265263) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) & 0xFFFFFFFF
    h = (h * 1274126177) & 0xFFFFFFFF
    return SAND if (h % 100) < 30 else EMPTY

# FNV-1a over the grid, row-major, 64-bit wraparound.
fn checksum(buf: List[UInt8]) -> UInt64:
    var c: UInt64 = 14695981039346656037
    for i in range(len(buf)):
        c = (c ^ UInt64(buf[i])) * 1099511628211
    return c

# One update step: sand falls down, else down-left, else down-right.
fn update(inout buf: List[UInt8], width: Int, height: Int):
    var y = height - 2
    while y >= 0:
        for x in range(width):
            var i = y * width + x
            if buf[i] == SAND:
                var below = (y + 1) * width + x
                if buf[below] == EMPTY:
                    buf[below] = SAND
                    buf[i] = EMPTY
                elif x > 0 and buf[below - 1] == EMPTY:
                    buf[below - 1] = SAND
                    buf[i] = EMPTY
                elif x < width - 1 and buf[below + 1] == EMPTY:
                    buf[below + 1] = SAND
                    buf[i] = EMPTY
        y -= 1

fn make_grid(size: Int) -> List[UInt8]:
    var buf = List[UInt8](capacity=size)
    for _ in range(size):
        buf.append(EMPTY)
    return buf

# Format a 64-bit value as 16 lowercase hex digits (matches "%016llx").
fn hex16(v: UInt64) -> String:
    alias digits = "0123456789abcdef"
    var s = String("")
    var shift = 60
    while shift >= 0:
        var nibble = Int((v >> shift) & 0xF)
        s += digits[nibble : nibble + 1]
        shift -= 4
    return s

# ---------------------------------------------------------------------------
# Headless benchmark.
# ---------------------------------------------------------------------------
fn run_bench(steps: Int, width: Int, height: Int):
    var buf = make_grid(width * height)
    for y in range(height):
        for x in range(width):
            buf[y * width + x] = seed_cell(x, y)

    var start = perf_counter_ns()
    for _ in range(steps):
        update(buf, width, height)
    var elapsed_ms = Float64(perf_counter_ns() - start) / 1.0e6

    var cells = Float64(width * height) * Float64(steps)
    var mcells = (cells / (elapsed_ms / 1000.0) / 1e6) if elapsed_ms > 0.0 else 0.0

    var line = String("RESULT impl=mojo rule=scalar")
    line += " width=" + String(width)
    line += " height=" + String(height)
    line += " steps=" + String(steps)
    line += " elapsed_ms=" + String(elapsed_ms)
    line += " mcells_per_s=" + String(mcells)
    line += " checksum=" + hex16(checksum(buf))
    print(line)

# ---------------------------------------------------------------------------
# Default ASCII animation (no graphics dependency).
# ---------------------------------------------------------------------------
fn run_ascii(width: Int, height: Int):
    var buf = make_grid(width * height)
    var rng: UInt64 = 0x9E3779B97F4A7C15

    # Run a bounded number of frames so the demo terminates cleanly.
    for frame in range(400):
        # Occasionally drop a small chunk of sand near the top-center.
        rng = rng * 6364136223846793005 + 1442695040888963407
        if (rng >> 33) % 4 == 0:
            var cx = Int((rng >> 17) % UInt64(width))
            for dy in range(3):
                for dx in range(-2, 3):
                    var nx = cx + dx
                    if nx >= 0 and nx < width and dy < height:
                        buf[dy * width + nx] = SAND

        update(buf, width, height)

        # Clear screen (ANSI) and draw.
        print("\x1b[2J\x1b[H", end="")
        for y in range(height):
            var row = String("")
            for x in range(width):
                row += "#" if buf[y * width + x] == SAND else " "
            print(row)
        sleep(0.05)

# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
fn to_int(s: String, fallback: Int) -> Int:
    try:
        return Int(s)
    except:
        return fallback

fn main():
    var args = argv()
    if len(args) > 1 and args[1] == "--bench":
        var steps = to_int(String(args[2]), 1000) if len(args) > 2 else 1000
        var width = to_int(String(args[3]), 400) if len(args) > 3 else 400
        var height = to_int(String(args[4]), 300) if len(args) > 4 else 300
        run_bench(steps, width, height)
    else:
        run_ascii(70, 30)
