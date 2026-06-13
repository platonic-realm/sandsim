# sandsim

sandsim is a collection of sand falling simulations implemented in various
programming languages and frameworks. It serves as a benchmark and learning
tool for comparing different approaches to optimizing grid-based simulations.

## Overview

The core simulation involves a 2D grid where sand particles fall due to gravity
and interact with each other. The basic rules are:

1. Sand falls straight down if the cell below is empty.
2. If blocked, sand tries to fall diagonally left or right.
3. If all paths are blocked, the sand stays in place.

The reference for these rules is [`cpp/sandsim_scalar_sb.cpp`](cpp/sandsim_scalar_sb.cpp).
Every CPU implementation reproduces it exactly, which lets the benchmark harness
verify that they all agree bit-for-bit (see [Benchmarking](#benchmarking)).

There is also a **multi-material (Noita-style) track** that adds solids, powder,
liquid, and gas with density-based interaction — see
[Materials](#materials-noita-style) below and [MATERIALS.md](MATERIALS.md).

## Implementations

| Implementation | Status | Notes |
|----------------|--------|-------|
| [C](c/)           | ✅ done        | Scalar, SDL2 window, headless `--bench` |
| [C++](cpp/)       | ✅ done        | scalar / SSE / AVX2 × sand / materials / world; `--bench` on `scalar_sb` |
| [Python](python/) | ✅ done        | Pygame window |
| [Rust](rust/)     | ✅ done        | `std`-only (SDL2 via FFI), headless `--bench` |
| [Zig](zig/)       | ✅ done        | SDL2 via `@cImport`, headless `--bench` |
| [OpenGL](opengl/) | ✅ done        | GL 4.3 compute shader, GPU; headless `--bench` |
| [Vulkan](vulkan/) | ✅ done        | Compute shader, GPU; headless `--bench` |
| [HIP](hip/)       | ✅ done        | Compute kernel (ROCm / AMD, or CUDA backend); headless `--bench` |

The GPU implementations (OpenGL, Vulkan, HIP) follow a shared model: a
double-buffered grid in GPU memory where each step the source half is copied to
the destination half and a compute shader/kernel atomically claims destination
cells. They form a distinct "gpu" rule group from the CPU scalar rule.

## Building

Each directory builds on its own (see its README), or use the root Makefile:

```sh
make all      # build every implementation whose toolchain is installed
make c        # build a single implementation (c rust zig cpp opengl hip vulkan)
make bench    # build the benchmarkable implementations and print a comparison table
make clean    # remove all build artifacts
```

`make all` detects the available toolchains and skips any that are missing,
printing a short note for each skip.

Common dependencies: a C/C++ compiler, SDL2 (CPU/HIP/Vulkan windows),
GLEW + GLFW (OpenGL), the Vulkan SDK + `glslc` (Vulkan), `cargo` (Rust),
`zig` (Zig), and `hipcc` (HIP).

## Benchmarking

Every implementation accepts a headless benchmark mode:

```sh
./<binary> --bench [steps=1000] [width=400] [height=300]
```

which skips all window/display setup, seeds the grid deterministically, times
only the update loop, and prints one machine-readable line:

```
RESULT impl=<name> rule=<scalar|gpu> width=W height=H steps=N \
       elapsed_ms=<f> mcells_per_s=<f> checksum=<hex16> [sand=<n>]
```

The seed and 64-bit FNV-1a checksum use identical fixed-width integer
arithmetic in every language, so all `rule=scalar` implementations produce the
**same checksum** — a strict cross-language correctness check. GPU
implementations resolve cell contention by scheduling order, so their checksum
can vary run to run; they instead report a conserved `sand` count.

Run the whole comparison with `make bench` (or `tools/bench.sh [steps] [w] [h]`),
which builds what it can, runs every available implementation, prints a table
sorted by throughput, and asserts that the scalar group agrees. See
[BENCHMARKS.md](BENCHMARKS.md) for methodology and sample results.

## Materials (Noita-style)

Beyond the single-material sand simulation, sandsim includes a **multi-material
engine** with solids (`WALL`), powder (`SAND`), liquid (`WATER`), and `GAS`.
Movement is a swap, so materials are conserved, and density decides interaction:
sand sinks through water, water sinks through gas, gas rises through air — giving
a natural gas/water/sand stack.

```sh
make materials            # build the material-capable implementations
cpp/sandsim_materials     # interactive: number keys pick a material, mouse paints
make bench-materials      # cross-check all five language ports agree
```

It is implemented identically in C, C++, Python, Rust, and Zig, and the five
agree on a bit-for-bit checksum and conserved per-material counts. See
[MATERIALS.md](MATERIALS.md) for the rules, controls, and verification details.

## Big world (streaming, Noita-style)

To simulate a world far larger than memory or the screen, sandsim adds a
**chunked, disk-streamed world** modeled on Noita's engine: the world is split
into 64×64 chunks, only a few "live boxes" around the camera are kept resident
and simulated, settled chunks **sleep**, and everything else is **saved to disk**
and reloaded on demand.

```sh
make world            # build the chunked-world implementations
cpp/sandsim_world     # interactive: WASD/arrows pan the camera, number keys paint
make bench-world      # deterministic cross-language streaming cross-check
```

It is implemented in C, C++, Python, Rust, and Zig (all five agree on a
whole-world checksum and conserve every material across the streaming
round-trip), plus **connected SIMD** variants
(`cpp/sandsim_world_sse.cpp` and `_avx.cpp`) that simulate one contiguous grid
with the single-grid SIMD technique (16/32 adjacent cells per instruction) and
the full materials rule — material flows across the whole region, no scalar
border pass. See [WORLD.md](WORLD.md) for the Noita research and the design.

## Getting Started

1. Clone this repository:

   ```sh
   git clone https://github.com/yourusername/sandsim.git
   ```

2. Build everything available and run the benchmark:

   ```sh
   cd sandsim
   make bench
   ```

3. Or follow the README in any implementation directory to build and run that
   version interactively.

## Contributing

Contributions are welcome! If you'd like to add an implementation in a new
language or framework, please:

1. Create a new directory for your implementation.
2. Include a README with build and run instructions.
3. Follow the basic simulation rules and, ideally, implement the `--bench`
   contract above so the new version joins the comparison harness.
4. Submit a pull request with a description of your implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgments

- Inspired by falling sand games (Noita) and cellular automata simulations.
