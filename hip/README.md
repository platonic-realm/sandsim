# sandsim HIP Implementation

A GPU falling-sand simulation using a **HIP compute kernel**. It uses the same
double-buffered, atomic-claim model as the Vulkan/OpenGL versions: a device
buffer holds the grid twice (a src half and a dst half); each step the host
copies src→dst on the device and the kernel atomically claims destination cells
(`atomicCAS`). HIP is portable: it runs on AMD GPUs through ROCm and on NVIDIA
GPUs through HIP's CUDA back-end.

## Requirements

- `hipcc` (ROCm) — or a CUDA toolkit with HIP installed
- SDL2 (for the interactive window)
- A supported GPU

## Build

```sh
make                       # hipcc -std=c++17 -O2 ...
make GPU_ARCH=gfx1032      # optionally target a specific arch
```

## Run

```sh
./sandsim_hip                       # interactive SDL2 window (device->host copy per frame)
./sandsim_hip --bench 1000 400 300  # headless benchmark on the GPU
```

## Controls

- Left mouse drag: add sand
- `C`: clear
- `R`: randomize (~30% density)

## Benchmark

This is part of the `gpu` rule group. The reported `checksum` may vary run to
run (atomic contention is resolved by GPU scheduling order); the conserved
`sand` count is the deterministic check, and matches the OpenGL and Vulkan
back-ends for the same seed. See [BENCHMARKS.md](../BENCHMARKS.md).
