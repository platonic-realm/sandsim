# sandsim — Vulkan (compute)

The multi-material streaming world on the GPU via a **Vulkan compute shader**
([`shaders/world.comp`](shaders/world.comp)), bit-identical to the C++ and
OpenGL builds.

Each frame is the same 16 disjoint sub-passes as [`cpp/simd_core.h`](../cpp/simd_core.h),
one dispatch each (with a pipeline barrier between); a thread is the **source**
of a move based purely on its `(x,y)` coordinates, so the in-place update is
race-free and reproduces the CPU result exactly. The live `4×4`-chunk window is a
**host-visible (mapped) storage buffer**, so the CPU streams chunks to/from disk
directly into it — the chunk↔disk logic is identical to the C++ build, no staging
copies. Consecutive frames between camera moves are recorded into one command
buffer and submitted together (one fence wait), so huge worlds stream while the
simulation runs on the GPU.

## Requirements

- The Vulkan SDK (headers, loader, and `glslc`)
- SDL2 (interactive window)
- A Vulkan-capable GPU and driver

## Build & run

```sh
make                                  # glslc shaders/world.comp -> .spv, then the app
./sandsim_world_vk                    # interactive: arrows pan, number keys paint
./sandsim_world_vk --bench 600 6 6    # headless streaming benchmark (one RESULT line)
```

The SPIR-V is located next to the executable, so the binary runs from anywhere
(e.g. the repo root via `tools/benchmark.sh`).

## Controls

- Arrows: pan the camera by a chunk
- `1` Wall · `2` Sand · `3` Water · `4` Gas · `0` Eraser
- Left mouse: paint

## Notes

The mapped buffer is `HOST_VISIBLE | HOST_COHERENT`, which is the simplest way to
let the CPU stream straight into the simulated grid; on a discrete GPU that
trades some compute bandwidth (cells are accessed over PCIe) for streaming
simplicity. The `RESULT` checksum still matches the C++ and OpenGL builds
bit-for-bit — see [`../tools/benchmark.sh`](../tools/benchmark.sh) and
[WORLD.md](../WORLD.md).
