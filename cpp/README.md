# SandSim C++ Implementation

This directory contains the C++ implementation of the sandsim project, including scalar, SSE, and AVX versions of the sand falling simulation, in both single-buffer and multi-buffer variants.

## Versions

1. **Scalar Single Buffer Version (sandsim_scalar_sb.cpp)**: Basic implementation without SIMD optimizations.
2. **Scalar Multi-Buffer Version (sandsim_scalar_mb.cpp)**: Basic implementation with multi-buffer support.
3. **SSE Single Buffer Version (sandsim_sse_sb.cpp)**: SSE-optimized implementation with a single buffer.
4. **SSE Multi-Buffer Version (sandsim_sse_mb.cpp)**: SSE-optimized implementation with multi-buffer support.
5. **AVX2 Single Buffer Version (sandsim_avx_sb.cpp)**: AVX2-optimized implementation with a single buffer.
6. **AVX2 Multi-Buffer Version (sandsim_avx_mb.cpp)**: AVX2-optimized implementation with multi-buffer support.
7. **NEON Multi-Buffer Version (sandsim_neon_mb.cpp)**: ARM NEON-optimized implementation with multi-buffer support.

## Requirements

- C++17 compatible compiler (e.g., GCC 7+, Clang 5+, MSVC 2017+)
- SDL2 library for visualization
- For NEON version: ARM processor with NEON support

## Compiling the Project

You can compile the project using either GCC or Clang. Here are the commands for each version:

### Scalar Single Buffer Version

```
g++ -std=c++17 -O3 sandsim_scalar_sb.cpp -o sandsim_scalar_sb -lSDL2
```

### Scalar Multi-Buffer Version

```
g++ -std=c++17 -O3 sandsim_scalar_mb.cpp -o sandsim_scalar_mb -lSDL2
```

### SSE Single Buffer Version

```
g++ -std=c++17 -O3 -msse4.1 sandsim_sse_sb.cpp -o sandsim_sse_sb -lSDL2
```

### SSE Multi-Buffer Version

```
g++ -std=c++17 -O3 -msse4.1 sandsim_sse_mb.cpp -o sandsim_sse_mb -lSDL2
```

### AVX2 Single Buffer Version

```
g++ -std=c++17 -O3 -mavx2 sandsim_avx_sb.cpp -o sandsim_avx_sb -lSDL2
```

### AVX2 Multi-Buffer Version

```
g++ -std=c++17 -O3 -mavx2 sandsim_avx_mb.cpp -o sandsim_avx_mb -lSDL2
```

### NEON Multi-Buffer Version (for ARM processors)

```
g++ -std=c++17 -O3 -march=armv8-a+simd sandsim_neon_mb.cpp -o sandsim_neon_mb -lSDL2
```

## Running the Simulations

After compiling, you can run each version by executing the corresponding binary:

```
./sandsim_scalar_sb
./sandsim_scalar_mb
./sandsim_sse_sb
./sandsim_sse_mb
./sandsim_avx_sb
./sandsim_avx_mb
./sandsim_neon_mb
```

## Controls

- Left mouse click and drag: Add sand
- 'C' key: Clear the simulation
- 'R' key: Randomize the sand distribution
- Space bar: Cycle through buffers (multi-buffer versions only)
- Number keys 0-9: Switch to a specific buffer (multi-buffer versions only)

## Implementation Details

- The scalar versions use basic loop-based updates, with the multi-buffer version managing multiple simulations.
- The single buffer SIMD versions (SSE, AVX2) use SIMD instructions to process multiple particles in parallel within a single simulation.
- The multi-buffer SIMD versions (SSE, AVX2, NEON) simulate multiple sand buffers simultaneously, allowing for increased performance and the ability to switch between different simulations.
- Each sand particle is rendered as a 2x2 pixel block for better visibility.

## Performance Considerations

- The SIMD versions should generally offer better performance than the scalar versions, especially for larger simulation sizes.
- Single buffer versions process one simulation at a time, with SIMD versions parallelizing particle updates.
- Multi-buffer versions process multiple entire simulations in parallel:
  - Scalar multi-buffer version simulates multiple buffers sequentially
  - SSE version processes 16 buffers simultaneously
  - AVX2 version processes 32 buffers simultaneously
  - NEON version processes 16 buffers simultaneously

## Notes

- The NEON version is designed for ARM processors and will not compile on x86 systems.
- For best performance, compile with optimization flags and run on hardware that supports the corresponding SIMD instruction set.
- The multi-buffer versions allow for interesting effects by switching between buffers or viewing multiple simulations at once.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file in the root directory for details.