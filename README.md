# sandsim

A multi-material falling-sand **world**: a huge grid of materials (sand, water,
gas, walls) that is chunked and streamed to/from disk around a camera, so the
world can be far larger than memory. There is one engine, with a single
implementation per platform:

| Platform | Status | Notes |
|----------|--------|-------|
| [C++ (SIMD)](cpp/) | ✅ | One binary that picks AVX2 (32-wide) or SSE4.1 (16-wide) at runtime. |
| [OpenGL](opengl/)  | 🚧 migrating | GPU compute, disk-streamed. |
| [Vulkan](vulkan/)  | 🚧 migrating | GPU compute, disk-streamed. |

## The simulation

Materials: `EMPTY`, `WALL` (solid), `SAND` (powder), `WATER` (liquid), `GAS`.
Movement is a density swap (sand sinks through water, water sinks through gas,
gas rises), so every material is conserved.

The update rule is **order-independent**: each frame is a fixed sequence of
sub-passes, and within a pass every move is between a disjoint pair of cells
(vertical moves split by row parity, diagonal/horizontal by column parity). That
makes it a pure function of the previous frame — which is what lets the
massively-parallel GPU backends reproduce the CPU result **bit-for-bit**. See
[WORLD.md](WORLD.md) for the design and the Noita techniques it adapts.

## Build & run

```sh
make            # build all platforms
make cpp        # just the C++ SIMD world
./cpp/sandsim_world                       # interactive: arrows pan, number keys paint
./cpp/sandsim_world --bench 600 6 6       # headless: whole-world checksum + conserved counts
SANDSIM_SIMD=sse ./cpp/sandsim_world --bench 600 6 6   # force SSE (default: widest the CPU has)
make benchmark  # build all three, verify identical output, print a throughput table
```

Dependencies: a C++17 compiler + SDL2; GLEW + GLFW (OpenGL); the Vulkan SDK +
`glslc` (Vulkan).

## License

MIT — see [LICENSE](LICENSE).
