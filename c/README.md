# sandsim C Implementation

A C port of the canonical scalar falling-sand simulation, a direct translation
of [`cpp/sandsim_scalar_sb.cpp`](../cpp/sandsim_scalar_sb.cpp). Uses SDL2 for
the interactive window and shares the project's deterministic `--bench` mode.

## Requirements

- A C11 compiler (GCC or Clang)
- SDL2

## Build

```sh
make            # or: cc -std=c11 -O3 sandsim.c -o sandsim $(pkg-config --cflags --libs sdl2)
```

## Run

```sh
./sandsim                       # interactive 400x300 window (rendered 2x -> 800x600)
./sandsim --bench 1000 400 300  # headless benchmark, prints one RESULT line
make bench
```

## Controls

- Left mouse drag: add sand
- `C`: clear
- `R`: randomize (~30% density)

## Benchmark

In `--bench` mode the program seeds the grid deterministically, times the update
loop, and prints a `RESULT` line whose checksum matches every other scalar-rule
implementation (`31128ca3d1fcadc6` at the default 1000 steps / 400×300). See
[BENCHMARKS.md](../BENCHMARKS.md).

## Materials variant

`sandsim_materials.c` adds the Noita-style multi-material engine (wall, sand,
water, gas). Build with `make sandsim_materials` and run `./sandsim_materials`;
number keys pick a material and the mouse paints. See [MATERIALS.md](../MATERIALS.md).


## Streaming world variant

The chunked, disk-streamed "big world" (Noita-style: only a few live boxes around
a camera resident, the rest saved to disk) is in `sandsim_world.c`. Build it with `make world`
from the repo root and cross-check all languages with `make bench-world`. See
[WORLD.md](../WORLD.md).
