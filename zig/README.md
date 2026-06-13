# sandsim Zig Implementation

A Zig port of the canonical scalar falling-sand simulation. SDL2 is pulled in
via `@cImport`, and the single `RESULT` line is printed with libc's `printf`,
keeping the program independent of std.io changes across Zig releases.

## Requirements

- Zig 0.16 (developed and tested against 0.16.0)
- SDL2

## Build

```sh
zig build-exe sandsim.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim
```

(`make zig` from the repo root runs exactly this command.)

## Run

```sh
./sandsim                       # interactive window
./sandsim --bench 1000 400 300  # headless benchmark
```

## Controls

- Left mouse drag: add sand
- `C`: clear
- `R`: randomize (~30% density)

## Benchmark

`--bench` seeds the grid deterministically, times the update loop, and prints a
`RESULT` line whose checksum matches every other scalar-rule implementation
(`31128ca3d1fcadc6` at 1000 steps / 400×300). See [BENCHMARKS.md](../BENCHMARKS.md).

## Note on Zig versions

Zig's standard library moves quickly. This file targets 0.16: it takes a
`std.process.Init.Minimal` argument in `main`, iterates arguments with
`std.process.Args.Iterator`, and times with libc `clock_gettime`. On a different
Zig version these few touch points may need adjusting; the simulation logic does
not.

## Materials variant

`sandsim_materials.zig` adds the Noita-style multi-material engine (wall, sand,
water, gas). Build it with:

```sh
zig build-exe sandsim_materials.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim_materials
```

(or `make materials` from the repo root). Run `./sandsim_materials`; number keys
pick a material and the mouse paints. See [MATERIALS.md](../MATERIALS.md).


## Streaming world variant

The chunked, disk-streamed "big world" (Noita-style: only a few live boxes around
a camera resident, the rest saved to disk) is in `sandsim_world.zig`. Build it with `make world`
from the repo root and cross-check all languages with `make bench-world`. See
[WORLD.md](../WORLD.md).
