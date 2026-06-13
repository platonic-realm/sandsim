# Multi-material engine (Noita-style)

The base project simulates a single material (sand). The **materials** track
extends it to several material classes that interact, in the spirit of falling-
sand games like Noita. It is an additive feature: the original single-material
simulations and their benchmark are unchanged.

## Materials

| Material | Class  | Behaviour |
|----------|--------|-----------|
| `EMPTY`  | —      | Air / nothing. |
| `WALL`   | solid  | Never moves. |
| `SAND`   | powder | Falls straight down, then piles diagonally. |
| `WATER`  | liquid | Falls, then spreads horizontally to find its level. |
| `GAS`    | gas    | Rises, then spreads horizontally under ceilings. |

## The rule

Movement is always a **swap** between a cell and its target, so every material
is conserved — only `EMPTY` shuffles around. A mover may swap into a target only
if the target is strictly *lighter* in the relevant direction:

| Mover   | May swap into        | Effect |
|---------|----------------------|--------|
| `SAND`  | `EMPTY`, `WATER`, `GAS` | sinks through water and gas |
| `WATER` | `EMPTY`, `GAS`         | sinks through gas; rests on sand |
| `GAS`   | `EMPTY`                | rises through air |

Because `GAS` never re-enters `WATER` (only `WATER` initiates the water/gas
swap, by sinking), the water/gas interface settles instead of oscillating. The
result is a natural density stack: gas on top, then water, then sand, with walls
holding it all in place.

Each cell is processed in place, bottom-to-top, with a per-frame `moved` flag so
no cell takes part in two swaps in one step. That single flag makes the update
deterministic for both down-movers (sand, water) and up-movers (gas). A
position-parity tie-break (`(x + y + frame) & 1`) chooses left-vs-right so
motion is unbiased yet reproducible.

## Implementations

The engine is implemented identically in five languages:

- [C](c/sandsim_materials.c)
- [C++](cpp/sandsim_materials.cpp) — the canonical reference
- [Python](python/sandsim_materials.py) — the readable reference
- [Rust](rust/src/materials.rs)
- [Zig](zig/sandsim_materials.zig)

There are also **SIMD** variants — [`cpp/sandsim_materials_sse.cpp`](cpp/sandsim_materials_sse.cpp)
and `_avx.cpp` — that run the same rule on one connected grid with the single-grid
SSE/AVX2 technique (see [`cpp/simd_core.h`](cpp/simd_core.h) and [WORLD.md](WORLD.md)).
They use a pass-based update order, so their checksum differs from the scalar
ports above (but SSE and AVX agree with each other and conserve every material).

The GPU ports of the base simulation are not (yet) extended: the
GPU back-ends use an atomic single-target "claim" model that does not map
cleanly onto liquids and gases, which need horizontal flow and density swaps.
The model above is the reference for porting them.

## Running

Build them with `make materials`, then either run interactively or benchmark:

```sh
# Interactive (SDL2 window; Python uses pygame)
cpp/sandsim_materials
python3 python/sandsim_materials.py

# Headless benchmark / cross-check
make bench-materials                       # runs all five and verifies they agree
cpp/sandsim_materials --bench 1000 400 300 # one implementation

# Render a snapshot to an image (no display needed):
cpp/sandsim_materials --ppm out.ppm 300 200 150 && magick out.ppm out.png
```

### Controls (interactive)

- Number keys pick the material to paint: `1` Wall, `2` Sand, `3` Water,
  `4` Gas, `0` Eraser.
- Left mouse button paints the selected material.
- `C` clears the grid.

## Verification

`--bench` seeds a deterministic scene (walls framing a box with a perforated
shelf; sand on top, water in the middle, gas at the bottom), runs the update
loop, and prints a `RESULT` line with a checksum and per-material counts:

```
RESULT impl=<name> rule=materials width=W height=H steps=N elapsed_ms=.. \
       mcells_per_s=.. checksum=<hex16> empty=.. wall=.. sand=.. water=.. gas=..
```

Two correctness properties are checked:

1. **Conservation** — the `wall`/`sand`/`water`/`gas` counts are invariant for
   any number of steps (swaps never create or destroy material).
2. **Cross-language agreement** — all five implementations use identical
   fixed-width integer arithmetic, so they produce the **same checksum** for the
   same seed. `make bench-materials` asserts both and fails loudly otherwise.

For reference, `--bench 200 120 90` yields checksum `79eb588fa7c08480` with
counts `wall=411 sand=1437 water=1182 gas=676`.
