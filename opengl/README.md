# sandsim — OpenGL (compute)

The multi-material streaming world on the GPU via an **OpenGL 4.3 compute
shader**, bit-identical to the C++ and Vulkan builds.

Each frame is the same 16 disjoint sub-passes as [`cpp/simd_core.h`](../cpp/simd_core.h),
one compute dispatch each, with a memory barrier between. A thread is the
**source** of a move based purely on its `(x,y)` coordinates, so the in-place
update is race-free and reproduces the CPU result exactly. The live `4×4`-chunk
window lives in an SSBO where the step runs; a CPU shadow drives the identical
chunk↔disk streaming, and the buffer is synced only when the camera moves — so
huge worlds stream while the per-frame simulation stays on the GPU. Rendering is
a fullscreen-triangle fragment shader that reads the grid SSBO directly (no
readback). Shaders are embedded in
[`sandsim_world_gl.cpp`](sandsim_world_gl.cpp) as strings.

## Requirements

- A C++17 compiler
- GLEW, GLFW, and an OpenGL 4.3-capable GPU/driver

## Build & run

```sh
make
./sandsim_world_gl                          # interactive: arrows pan, number keys paint
./sandsim_world_gl --res 1280x800 --scale 3 --sps 120  # window res / virtual-pixel size / physics rate
./sandsim_world_gl --bench 600 6 6          # headless streaming benchmark (one RESULT line)
```

`--bench` creates a hidden context, so it still needs a reachable GPU. GLEW
resolves entry points through GLX, so the program prefers GLFW's X11 backend
(which also works under XWayland) and tolerates the benign `NO_GLX_DISPLAY`
notice on EGL-backed contexts.

## Controls

- Arrows: pan the camera by a chunk
- Click a swatch in the on-screen palette to pick a material (or keys `0`-`9`:
  `0` Eraser · `1` Wall · `2` Sand · `3` Water · `4` Gas · `5` Oil · `6` Fire · `7` Lava · `8` Steam · `9` Wood · `P` Plant · `A` Acid · `M` Smoke · `G` Glass · `I` Ice · `S` Spring · `T` TNT · `H` Ash · `V` Volcano · `X` Void · `D` Mud · `Z` Virus · `E` Spark · `O` Obsidian · `L` Salt · `N` Snow · `Q` Mercury · `B` Gunpowder · `K` Thermite · `F` Frost · `W` Wisp · `C` Coal · `R` Ember · `U` Cloner · `Y` Crystal · `J` Antimatter · `;` Moss · `,` Fumes · `.` Wire · `/` e-head · `'` e-tail · `-` Igniter · `=` Sensor · `\` Life · ``` Geyser · Lye, Sodium, Coral, Phosphorus, Cement, Chlorine, Battery, Fuse, Cryo, Lamp, Petrify, Firework, Leviton, Sprout are palette-only)
- Left mouse: paint · `[` / `]`: brush size · `Esc`: quit

The `RESULT` checksum matches the C++ and Vulkan builds bit-for-bit; see
[`../tools/benchmark.sh`](../tools/benchmark.sh) and [WORLD.md](../WORLD.md).
