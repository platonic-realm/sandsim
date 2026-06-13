# sandsim Rust Implementation

A Rust port of the canonical scalar falling-sand simulation. It is **standard
library only** — no third-party crates — so it builds fully offline. The
interactive window talks to system SDL2 through a small hand-written FFI shim
(`mod sdl` in [`src/main.rs`](src/main.rs)); the `--bench` path is pure safe Rust.

## Requirements

- `cargo` / `rustc` (2021 edition)
- SDL2 (system library, linked via `build.rs`)

## Build

```sh
cargo build --release          # add --offline to force no network access
```

## Run

```sh
./target/release/sandsim                       # interactive window
./target/release/sandsim --bench 1000 400 300  # headless benchmark
```

## Controls

- Left mouse drag: add sand
- `C`: clear
- `R`: randomize (~30% density)

## Benchmark

`--bench` seeds the grid deterministically, times the update loop, and prints a
`RESULT` line whose checksum matches every other scalar-rule implementation
(`31128ca3d1fcadc6` at 1000 steps / 400×300). See [BENCHMARKS.md](../BENCHMARKS.md).

## Materials variant

The crate has a second binary, `sandsim_materials` (`src/materials.rs`), with the
Noita-style multi-material engine (wall, sand, water, gas). `cargo build --release`
builds it alongside `sandsim`; run `./target/release/sandsim_materials`. Number
keys pick a material and the mouse paints. See [MATERIALS.md](../MATERIALS.md).


## Streaming world variant

The chunked, disk-streamed "big world" (Noita-style: only a few live boxes around
a camera resident, the rest saved to disk) is in `src/world.rs (binary sandsim_world)`. Build it with `make world`
from the repo root and cross-check all languages with `make bench-world`. See
[WORLD.md](../WORLD.md).
