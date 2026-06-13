# sandsim — C++ (SIMD)

The multi-material streaming world on the CPU, using the single-grid SIMD
technique (the lanes are adjacent cells of one connected grid).

**One binary, runtime SIMD dispatch.** The update lives in
[`simd_core.h`](simd_core.h), templated over the vector width. It is compiled
twice — `world_step_sse.cpp` with `-msse4.1` (16 lanes) and `world_step_avx.cpp`
with `-mavx2` (32 lanes) — and the host (`sandsim_world.cpp`) picks the widest
the running CPU supports via `__builtin_cpu_supports`. Both compute the same
result (the rule is order-independent and width-independent), so the choice is
purely performance. Set `SANDSIM_SIMD=sse|avx` to force one.

## Build & run

```sh
make
./sandsim_world                       # interactive (arrows pan, number keys paint)
./sandsim_world --res 1280x800 --scale 3 --sps 120   # window res / virtual-pixel size / physics rate
./sandsim_world --bench 600 6 6       # headless: whole-world checksum + conserved counts
./sandsim_world --ppm out.ppm 500     # render a snapshot
```

The interactive view renders each cell as a `scale × scale` virtual pixel; the
resident window is `winW/scale × winH/scale` cells. The physics rate (`--sps`,
steps/second) is decoupled from rendering, so it's the same wall-clock speed on
every backend. `--res`/`--scale`/`--sps` (or `SANDSIM_RES`/`SANDSIM_SCALE`/
`SANDSIM_SPS`) work the same on all three backends; defaults 1024x768, 2x2, 60.

The same `simd_core.h` rule is implemented by the GPU compute shaders, so the
C++, OpenGL, and Vulkan worlds produce a bit-identical world from the same seed.
