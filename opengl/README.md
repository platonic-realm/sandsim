# sandsim OpenGL Implementation

A GPU falling-sand simulation using an **OpenGL 4.3 compute shader**. It follows
the same double-buffered, atomic-claim model as the Vulkan version: one SSBO
holds the grid twice (a src half and a dst half); each step the host copies
src→dst on the device and the compute shader atomically claims destination cells
(`atomicCompSwap`). Rendering is fully on the GPU — a fullscreen-triangle
fragment shader reads the grid SSBO and colors the screen. Shaders are embedded
in [`sandsim_gl.cpp`](sandsim_gl.cpp) as raw strings, so there are no external
shader files to locate at runtime.

## Requirements

- A C++17 compiler
- GLEW, GLFW, and an OpenGL 4.3-capable GPU/driver

## Build

```sh
make            # g++ ... $(pkg-config --cflags --libs glew glfw3 gl)
```

## Run

```sh
./sandsim_gl                       # interactive window
./sandsim_gl --bench 1000 400 300  # headless benchmark (creates a hidden context)
```

The headless `--bench` mode still needs a reachable GPU/display to create a GL
context. GLEW resolves entry points through GLX, so the program prefers GLFW's
X11 backend (which also works under XWayland) and tolerates the benign
`NO_GLX_DISPLAY` notice on EGL-backed contexts.

## Controls

- Left mouse drag: add sand
- `C`: clear
- `R`: randomize (~30% density)
- `Esc`: quit

## Benchmark

This is part of the `gpu` rule group. Because the compute shader resolves cell
contention by scheduling order, the reported `checksum` may vary run to run; the
conserved `sand` count is the deterministic check. See
[BENCHMARKS.md](../BENCHMARKS.md).
