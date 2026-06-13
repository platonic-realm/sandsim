# Benchmarks

This document describes how sandsim measures and compares implementations, and
records a sample run. Reproduce it with `make bench` (or
`tools/bench.sh [steps] [width] [height]`).

## The `--bench` contract

Every implementation accepts:

```sh
./<binary> --bench [steps=1000] [width=400] [height=300]
```

In this mode it:

1. allocates a `width × height` grid (no window/GPU-display setup),
2. seeds each cell deterministically,
3. times **only** the update loop over `steps` iterations, and
4. prints exactly one line:

```
RESULT impl=<name> rule=<scalar|gpu> width=W height=H steps=N \
       elapsed_ms=<f> mcells_per_s=<f> checksum=<hex16> [sand=<n>]
```

`mcells_per_s = width * height * steps / elapsed_seconds / 1e6` — millions of
cell-updates per second.

### Deterministic seed and checksum

So that every language computes identical bits, both use fixed-width wraparound
arithmetic:

- **Seed** (~30% sand), `uint32` math:
  `h = x*374761393 + y*668265263; h = (h ^ (h>>13)) * 1274126177; sand = (h % 100) < 30`
- **Checksum** (FNV-1a over the grid, row-major, `uint64` math):
  `c = 14695981039346656037; for cell: c = (c ^ cell) * 1099511628211`

### Rule groups

- **scalar** — the canonical CPU rule from `cpp/sandsim_scalar_sb.cpp`. All
  scalar implementations (C++, C, Rust, Zig) must print the **same
  checksum**; the harness asserts this and fails if they diverge.
- **gpu** — the double-buffered, atomic-claim model used by OpenGL, Vulkan,
  and HIP. Cell contention is resolved by GPU scheduling order, so the
  exact final positions (and therefore the checksum) can vary run to run. The
  number of sand particles is conserved, so each GPU run instead reports a
  `sand` count, which is deterministic (here, 36021 for the default seed).

The scalar and gpu groups are **not** expected to share a checksum: the GPU rule
uses a parity-based left/right preference, whereas the scalar rule always tries
left before right.

## Sample results

Hardware: AMD Ryzen 7 6800HS (Zen 3+, 8 cores) with integrated Radeon 680M
(gfx1032-class, RDNA2), Arch Linux. Toolchains: gcc 16.1, rustc 1.96, zig 0.16,
hipcc 7.2 (ROCm), Mesa OpenGL/Vulkan. `steps=1000`, grid `400×300`.

| Implementation | Rule | Mcells/s | Checksum | Sand |
|----------------|------|---------:|----------|-----:|
| opengl         | gpu  | 14014.54 | (varies) | 36021 |
| hip            | gpu  | 11488.08 | (varies) | 36021 |
| vulkan         | gpu  |  4474.47 | (varies) | 36021 |
| c              | scalar |  1075.54 | 31128ca3d1fcadc6 | - |
| cpp_scalar_sb  | scalar |  1022.65 | 31128ca3d1fcadc6 | - |
| zig            | scalar |   990.33 | 31128ca3d1fcadc6 | - |
| rust           | scalar |   862.47 | 31128ca3d1fcadc6 | - |

`scalar-rule checksum agreement: PASS (4 implementations share 31128ca3d1fcadc6)`

### Notes and takeaways

- **The scalar implementations land within ~20% of one another.** They run the
  same memory-bound, branchy, single-threaded inner loop, so the language
  matters less than how the optimizer vectorizes it; the spread is real but
  small.
- **The GPU compute versions are ~10× faster** than the CPU versions despite
  doing more synchronization work, because thousands of cells update in
  parallel each step.
- **Batching the GPU work matters far more than the API.** The first cut of the
  Vulkan benchmark copied the grid on the *host* and waited on a fence after
  *every* step; it managed only ~38 Mcells/s — ~300× slower than OpenGL. Moving
  the copy onto the device (`vkCmdCopyBuffer`) and recording all steps into a
  single command buffer with pipeline barriers — one submit, one fence wait —
  brought it to ~4500 Mcells/s, a **~120× speedup**, putting it in the same
  league as the others. The remaining gap to OpenGL/HIP is mostly that the
  integrated GPU's host-visible buffer copy is slower than a device-local one.
  The HIP benchmark applies the same idea with an async stream and a
  single `StreamSynchronize`, and OpenGL caches its uniform locations out of the
  step loop. The lesson: keep work on the device and avoid a host round-trip per
  step.
- **All three runnable GPU back-ends conserve the same 36021 sand particles**,
  a good cross-API correctness signal even though their exact layouts differ.

Numbers are indicative of a single laptop and will vary with hardware, drivers,
thermal state, and grid size. The point of the harness is the *comparison* and
the *correctness check*, not any absolute figure.
