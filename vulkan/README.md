# sandsim — Vulkan (compute)

The multi-material streaming world on the GPU via a **Vulkan compute shader**
([`shaders/world.comp`](shaders/world.comp)), bit-identical to the C++ and
OpenGL builds.

Each frame is the same 16 disjoint sub-passes as [`cpp/simd_core.h`](../cpp/simd_core.h),
one dispatch each (with a pipeline barrier between); a thread is the **source**
of a move based purely on its `(x,y)` coordinates, so the in-place update is
race-free and reproduces the CPU result exactly. The live `4×4`-chunk window
lives in a **device-local** storage buffer (fast VRAM, where the compute runs); a
**host-visible staging buffer** is what the CPU streams chunks to/from disk into,
copied to/from the device only at camera moves (and seed/summary). Consecutive
frames between camera moves are recorded into one command buffer and submitted
together (one fence wait), so huge worlds stream while the simulation runs on the
GPU.

## Requirements

- The Vulkan SDK (headers, loader, and `glslc`)
- SDL2 (interactive window)
- A Vulkan-capable GPU and driver

## Build & run

```sh
make                                        # glslc shaders/world.comp -> .spv, then the app
./sandsim_world_vk                          # interactive: arrows pan, number keys paint
./sandsim_world_vk --res 1280x800 --scale 3 --sps 120  # window res / virtual-pixel size / physics rate
./sandsim_world_vk --bench 600 6 6          # headless streaming benchmark (one RESULT line)
```

The SPIR-V is located next to the executable, so the binary runs from anywhere
(e.g. the repo root via `tools/benchmark.sh`).

## Controls

- Arrows: pan the camera by a chunk
- Click a swatch in the on-screen palette to pick a material (or keys `0`-`9`:
  `0` Eraser · `1` Wall · `2` Sand · `3` Water · `4` Gas · `5` Oil · `6` Fire · `7` Lava · `8` Steam · `9` Wood · `P` Plant · `A` Acid · `M` Smoke · `G` Glass · `I` Ice · `S` Spring · `T` TNT · `H` Ash · `V` Volcano · `X` Void · `D` Mud · `Z` Virus · `E` Spark · `O` Obsidian · `L` Salt · `N` Snow · `Q` Mercury · `B` Gunpowder · `K` Thermite · `F` Frost · `W` Wisp · `C` Coal · `R` Ember · `U` Cloner · `Y` Crystal · `J` Antimatter · `;` Moss · `,` Fumes · `.` Wire · `/` e-head · `'` e-tail · `-` Igniter · `=` Sensor · `\` Life · ``` Geyser · Lye, Sodium are palette-only)
- Left mouse: paint · `[` / `]`: brush size

Presentation uses SDL's accelerated renderer (snappy, matches the C++ build);
set `SANDSIM_VK_RENDERER=software` to force a CPU blit if it flickers on your
driver.

## Notes

The interactive view presents through SDL's **software** renderer on purpose: an
accelerated (OpenGL) SDL renderer would fight the Vulkan compute for the same GPU
every frame, and that context thrash shows up as severe slowdown and flicker. The
compute + readback is ~0.5 ms/frame, so a CPU blit of one texture is plenty fast
and fully decoupled from the compute device. Each frame's step and host readback
are folded into a single command submission (one fence wait).

Cells/moved are `DEVICE_LOCAL` (VRAM) so the compute hits VRAM, not system RAM
over PCIe; the host-visible staging buffer is touched only when streaming. Note
that the per-frame cost here is dominated by the 16 ordered passes (a pipeline
barrier between each), not memory bandwidth — so on a GPU with resizable BAR the
placement matters little, but on one without it the device-local buffer is the
right call. The `RESULT` checksum still matches the C++ and OpenGL builds
bit-for-bit — see [`../tools/benchmark.sh`](../tools/benchmark.sh) and
[WORLD.md](../WORLD.md).
