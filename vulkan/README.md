# SandSim Vulkan Implementation

GPU-accelerated falling-sand simulation using a Vulkan compute shader.

## Requirements

- Vulkan SDK installed (for headers, loader, and `glslc`)
- SDL2
- A Vulkan-capable GPU and driver

## Build

From this `vulkan` directory, `make` (compiles the shader to SPIR-V, then the
app), or by hand:

```
glslc shaders/sand.comp -o shaders/sand.comp.spv
g++ -std=c++17 -O2 sandsim_vulkan_compute.cpp -o sandsim_vulkan_compute -lvulkan -lSDL2
```

## Run

```
./sandsim_vulkan_compute                  # interactive
./sandsim_vulkan_compute --bench 1000 400 300   # headless benchmark
```

## Controls

- Left mouse: add sand
- C: clear
- R: randomize

## Notes

- The simulation uses a double-buffered grid in a single storage buffer
  (src/dst halves). Each frame the host copies src→dst so particles "stay"
  unless the shader atomically claims a destination and clears the source in dst.
  Workgroup size is 16×16.
- `--bench` records every step into a single command buffer (device-side copy +
  pipeline barriers, one submit/fence), which is far faster than the naive
  per-step fence wait the interactive loop uses.
- A multi-buffer variant was removed: it simulated 16 independent grids at once
  via the dispatch z-dimension, which makes sense for filling idle SIMD lanes on
  a CPU but not on a GPU — a single grid already saturates the device, so it just
  did ~16× the work. See `cpp/` for the (CPU) SIMD discussion.
