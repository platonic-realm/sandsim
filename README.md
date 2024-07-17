# sandsim

sandsim is a collection of sand falling simulations implemented in various programming languages and frameworks. It serves as a benchmark and learning tool for comparing different approaches to optimizing grid-based simulations.

## Overview

The core simulation involves a 2D grid where sand particles fall due to gravity and interact with each other. The basic rules are:

1. Sand falls straight down if the cell below is empty.
2. If blocked, sand tries to fall diagonally left or right.
3. If all paths are blocked, the sand stays in place.

## Implementations

Target implementations:

- C
- [C++](cpp/README.md)
- [Python](python/README.md)
- Rust
- Zig
- Mojo
- OpenGL (GPU acceleration)
- Vulkan (GPU acceleration)
- CUDA (GPU acceleration)
- HIP (GPU acceleration)

Each implementation is contained in its own directory and includes a README with specific instructions.

## Getting Started

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sandsim.git
   ```

2. Navigate to the implementation you're interested in:
   ```
   cd sandsim/cpp
   ```

3. Follow the README instructions in that directory to build and run the simulation.

## Contributing

Contributions are welcome! If you'd like to add an implementation in a new language or framework, please:

1. Create a new directory for your implementation.
2. Include a README with build and run instructions.
3. Ensure your code is well-commented and follows the basic simulation rules.
4. Submit a pull request with a description of your implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by falling sand games (Noita) and cellular automata simulations.
