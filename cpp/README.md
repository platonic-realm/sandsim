# SandSim C++ Implementation

Three falling-sand engines, each in three SIMD tiers:

| Engine | scalar | SSE (16-wide) | AVX2 (32-wide) |
|--------|--------|---------------|----------------|
| **sand** (single material) | `sandsim_scalar_sb.cpp` | `sandsim_sse_sb.cpp` | `sandsim_avx_sb.cpp` |
| **materials** (sand/water/gas/wall) | `sandsim_materials.cpp` | `sandsim_materials_sse.cpp` | `sandsim_materials_avx.cpp` |
| **world** (chunked, disk-streamed) | `sandsim_world.cpp` | `sandsim_world_sse.cpp` | `sandsim_world_avx.cpp` |

The SIMD materials and world share [`simd_core.h`](simd_core.h), which holds the
**single-grid** SIMD update (lanes = adjacent cells of one connected grid)
templated over the vector width — the SSE and AVX builds of an engine produce
**identical** results (the width doesn't change the rule), AVX just ~2× faster.
See [WORLD.md](../WORLD.md) for the design.

> The multi-buffer (`_mb`) and NEON variants were removed: the multi-buffer
> technique packs *independent* simulations into the lanes, which can't model a
> connected world (lanes don't communicate).

## Requirements

- A C++17 compiler (GCC/Clang) and SDL2.

## Building

```sh
make                       # builds all nine variants (scalar/SSE/AVX x sand/materials/world)
make sandsim_world_avx     # build one variant
make bench                 # build sandsim_scalar_sb and run its headless benchmark
make clean
```

Individual compiles follow the pattern (SSE needs `-msse4.1`, AVX2 needs `-mavx2`):

```sh
g++ -std=c++17 -O3 sandsim_scalar_sb.cpp    -o sandsim_scalar_sb    $(pkg-config --cflags --libs sdl2)
g++ -std=c++17 -O3 -msse4.1 sandsim_world_sse.cpp -o sandsim_world_sse $(pkg-config --cflags --libs sdl2)
g++ -std=c++17 -O3 -mavx2   sandsim_world_avx.cpp -o sandsim_world_avx $(pkg-config --cflags --libs sdl2)
```

## Running

```sh
./sandsim_scalar_sb            # sand, interactive (left-drag adds sand, C clear, R randomize)
./sandsim_materials_sse        # materials, interactive (number keys pick material, mouse paints)
./sandsim_world_avx            # streamed world (arrows pan, number keys paint)
```

Each also has a headless `--bench` (and the materials/world SIMD variants a
`--ppm <file>` snapshot mode).

## Benchmark / verification

`sandsim_scalar_sb` is the golden reference for the cross-language sand checksum
(`31128ca3d1fcadc6` at 1000 steps / 400×300; see [BENCHMARKS.md](../BENCHMARKS.md)).
The SIMD engines verify against each other instead: SSE and AVX of the same
engine print the same checksum and conserve every material. The scalar materials
and world are the cross-language references in [MATERIALS.md](../MATERIALS.md)
and [WORLD.md](../WORLD.md).

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE)
file in the root directory for details.
