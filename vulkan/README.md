# SandSim Vulkan Implementation

GPU-accelerated falling-sand simulation using Vulkan compute. Includes single-buffer and multi-buffer variants.

## Requirements

- Vulkan SDK installed (for headers, loader, and `glslc`)
- SDL2
- A Vulkan-capable GPU and driver

## Build

From this `vulkan` directory:

1) Compile shaders to SPIR-V

```
glslc shaders/sand.comp -o shaders/sand.comp.spv
glslc shaders/sand_mb.comp -o shaders/sand_mb.comp.spv
```

2) Build the apps

```
g++ -std=c++17 -O2 sandsim_vulkan_compute.cpp -o sandsim_vulkan_compute -lvulkan -lSDL2
g++ -std=c++17 -O2 sandsim_vulkan_compute_mb.cpp -o sandsim_vulkan_compute_mb -lvulkan -lSDL2
```

## Run

```
./sandsim_vulkan_compute
./sandsim_vulkan_compute_mb
```

## Controls

- Left mouse: add sand
- C: clear
- R: randomize
- MB only: Space cycles buffers; 0-9 jump to specific buffer index

## Notes

- The simulation uses a double-buffered grid in a single storage buffer (src/dst halves).
- Each frame, the host copies src→dst so particles “stay” unless the shader atomically claims a destination and clears the source in dst.
- Workgroup size is 16×16; the MB variant uses the dispatch z-dimension for the buffer index.


