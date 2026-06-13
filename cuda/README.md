# sandsim CUDA Implementation

A GPU falling-sand simulation using a **CUDA compute kernel**. It is the same
double-buffered, atomic-claim model as the HIP version (`atomicCAS` to claim
destination cells), written against the CUDA runtime API. The kernel and host
loop are a near line-for-line mirror of [`hip/sandsim_hip.cpp`](../hip/sandsim_hip.cpp),
which is built and verified; only the `hip*` calls are swapped for `cuda*`.

> **Status: source-only.** The development host has no CUDA toolkit and no
> NVIDIA GPU, so this file was written to spec but not compiled there. Build it
> on a machine with the CUDA toolkit installed.

## Requirements

- CUDA toolkit (`nvcc`)
- SDL2 (for the interactive window)
- An NVIDIA GPU

## Build

```sh
make            # skips cleanly with a message if nvcc is not installed
make SM=86      # optionally target a specific compute capability (e.g. sm_86)
# or: nvcc -O2 -std=c++17 sandsim_cuda.cu -o sandsim_cuda $(pkg-config --libs sdl2)
```

## Run

```sh
./sandsim_cuda                       # interactive SDL2 window
./sandsim_cuda --bench 1000 400 300  # headless benchmark on the GPU
```

## Controls

- Left mouse drag: add sand
- `C`: clear
- `R`: randomize (~30% density)

## Benchmark

Part of the `gpu` rule group: the `checksum` may vary run to run; the conserved
`sand` count is the deterministic check. See [BENCHMARKS.md](../BENCHMARKS.md).
