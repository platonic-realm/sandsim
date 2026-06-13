# sandsim Mojo Implementation

A Mojo port of the canonical scalar falling-sand simulation. The default mode is
a dependency-free ASCII animation in the terminal; the `--bench` mode is the
project's shared headless benchmark. A graphical window is intentionally out of
scope here (it would require a graphics binding that cannot be assumed).

> **Status: source-only.** The development host has no Mojo toolchain, so this
> file was written to spec but not compiled there. It targets the Mojo standard
> library and uses the same fixed-width integer arithmetic as the other ports,
> so its `--bench` checksum is intended to match them (`31128ca3d1fcadc6` at
> 1000 steps / 400×300).

## Requirements

- The Mojo toolchain (`mojo`)

## Run

```sh
mojo run sandsim.mojo                       # ASCII animation in the terminal
mojo run sandsim.mojo --bench 1000 400 300  # headless benchmark
```

(or compile with `mojo build sandsim.mojo`).

## Benchmark

`--bench` seeds the grid deterministically (matching the C/C++/Rust/Zig ports),
times the update loop, and prints a `RESULT` line in the scalar rule group. See
[BENCHMARKS.md](../BENCHMARKS.md).

## Note on Mojo versions

Mojo's syntax and standard library are still evolving. This file uses `inout`
parameters, `from sys import argv`, and `from time import perf_counter_ns`; on a
different Mojo release a few of these touch points may need adjusting, but the
simulation logic does not.
