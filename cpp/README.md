# SandSim C++ Implementation

This directory contains the C++ implementation of the sandsim project, including scalar, SSE, and AVX2 versions of the sand falling simulation.

## Requirements

- C++17 compatible compiler (e.g., GCC 7+, Clang 5+, MSVC 2017+)
- SDL2 library for visualization (optional)

## Compiling the Project

You can compile the project using either GCC or Clang. Here are the commands for each version:

### AVX2 Version

GCC:
```
g++ -std=c++17 -O3 -march=native -mavx2 sandsim_avx2.cpp -o sandsim_avx2 -lSDL2
```

Clang:
```
clang++ -std=c++17 -O3 -march=native -mavx2 sandsim_avx2.cpp -o sandsim_avx2 -lSDL2
```



## Implementation Details

- `sandsim_scalar.cpp`: Basic implementation without SIMD optimizations.
- `sandsim_sse.cpp`: SSE-optimized implementation.
- `sandsim_avx2.cpp`: AVX2-optimized implementation.
 simulation mode.


## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file in the root directory for details.
