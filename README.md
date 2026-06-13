# sandsim

A multi-material falling-sand **world**: a huge grid of materials (sand, water,
gas, walls) that is chunked and streamed to/from disk around a camera, so the
world can be far larger than memory. There is one engine, with a single
implementation per platform:

| Platform | Status | Notes |
|----------|--------|-------|
| [C++ (SIMD)](cpp/) | ✅ | One binary that picks AVX2 (32-wide) or SSE4.1 (16-wide) at runtime. |
| [OpenGL](opengl/)  | ✅ | GPU compute (4.3), disk-streamed. Bit-identical to the C++ build. |
| [Vulkan](vulkan/)  | ✅ | GPU compute, disk-streamed. Bit-identical to the C++ build. |

All three produce the **same world from the same seed** — `make benchmark`
builds them, asserts the checksums match, and prints a throughput table.

## The simulation

Materials: `EMPTY`, `WALL` (solid), `SAND` (powder), `WATER`, `GAS`, `OIL`,
`FIRE`, `LAVA`. Movement is a density swap — heaviest to lightest is
`SAND > LAVA > WATER > OIL > air > GAS > FIRE` — so sand sinks through lava, oil
floats on water, gas rises, and fire licks upward. On top of movement there are
reactions, all order-independent and bit-identical on CPU and GPU:

- `FIRE` rises like flame and **burns out over time** (a deterministic per-cell,
  frame-varying transform — the same hash on CPU and GPU).
- `FIRE`/`LAVA` **ignite `OIL`** they touch: dab fire (or pour lava) into an oil
  pool and it combusts, the flame front creeping through the fuel one layer per
  frame before burning away.
- `LAVA` meeting `WATER` **freezes to stone** (`WALL`) at the interface.

Paint with the mouse and pick a material from the on-screen palette (or number
keys `0`-`7`); `[` / `]` size the brush. The palette is the same on all three
backends, and every rule — movement, the time-varying burn-out, and the
neighbour reactions — is bit-identical across CPU SIMD, OpenGL, and Vulkan.

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
./cpp/sandsim_world --res 1280x800 --scale 3   # window resolution + virtual-pixel size
./cpp/sandsim_world --bench 600 6 6       # headless: whole-world checksum + conserved counts
SANDSIM_SIMD=sse ./cpp/sandsim_world --bench 600 6 6   # force SSE (default: widest the CPU has)
make benchmark  # build all three, verify identical output, print a throughput table
```

The interactive view renders each cell as a **virtual pixel** of `scale × scale`
screen pixels, so the resident window is `winW/scale × winH/scale` cells. The
**physics rate is decoupled from rendering** (a fixed-timestep accumulator keyed
to real time), so the simulation runs at the same wall-clock speed on every
backend, whatever the frame rate. All configurable the same way on all three —
`--res WxH` / `--scale N` / `--sps STEPS_PER_SEC`, or `SANDSIM_RES` /
`SANDSIM_SCALE` / `SANDSIM_SPS` (default **1024×768, 2×2, 60 steps/s**).

Dependencies: a C++17 compiler + SDL2; GLEW + GLFW (OpenGL); the Vulkan SDK +
`glslc` (Vulkan).

## License

MIT — see [LICENSE](LICENSE).
