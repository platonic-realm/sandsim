# sandsim

A multi-material falling-sand **world**: a huge grid of materials (sand, water,
lava, oil, acid, gas, rock and more) that is chunked and streamed to/from disk
around a camera, so the world can be far larger than memory. It generates as a
varied landscape — an open sky over rolling ground, with underground caverns,
clustered pools of every liquid and gas, lava welling up from the deep, and the
odd buried spring or cache of TNT — and then comes alive as the materials react.
There is one engine, with a single implementation per platform:

| Platform | Status | Notes |
|----------|--------|-------|
| [C++ (SIMD)](cpp/) | ✅ | One binary that picks AVX2 (32-wide) or SSE4.1 (16-wide) at runtime. |
| [OpenGL](opengl/)  | ✅ | GPU compute (4.3), disk-streamed. Bit-identical to the C++ build. |
| [Vulkan](vulkan/)  | ✅ | GPU compute, disk-streamed. Bit-identical to the C++ build. |

All three produce the **same world from the same seed** — `make benchmark`
builds them, asserts the checksums match, and prints a throughput table.

## The simulation

Materials: `EMPTY`, `WALL` (solid), `SAND` (powder), `WATER`, `GAS`, `OIL`,
`FIRE`, `LAVA`, `STEAM`, `WOOD`, `PLANT`, `ACID`, `SMOKE`, `GLASS`, `ICE`, `SPRING`,
`TNT`, `ASH` (powder), `VOLCANO`, `VOID`, `MUD`, `VIRUS`, `SPARK`. Movement is a density swap
— heaviest to lightest is `SAND > LAVA > ACID > WATER > OIL > air > GAS > FIRE`, with
`STEAM`/`SMOKE` the lightest — so sand sinks through lava, acid sinks below water,
oil floats on water, and gas/fire/steam/smoke rise. `ASH` falls and piles like
sand. `WALL`, `WOOD`, `PLANT`, `GLASS`, `ICE`, `SPRING`, `TNT`, `VOLCANO`, `VOID`,
`MUD`, and `VIRUS` are solids that don't move.
On top of movement there are reactions, all order-independent and bit-identical on
CPU and GPU:

- `FIRE` rises like flame and **burns out over time** (a deterministic per-cell,
  frame-varying transform — the same hash on CPU and GPU), some of it wisping into
  `SMOKE` that rises and then fades, and a little settling as `ASH` — a grey powder
  that falls and piles up — so fires billow and leave soot behind.
- `FIRE`/`LAVA` **ignite `OIL` and `GAS`** they touch: dab fire (or pour lava) into
  an oil pool or a gas pocket and it combusts, the flame front racing through the
  fuel one layer per frame before burning away — so an underground gas pocket lit by
  a lava vein goes up in a sheet of flame.
- `WOOD` is a **flammable solid** — build a structure and set it alight; it
  catches *slowly* (frame-hashed, so timber smoulders where oil whooshes) and
  burns down.
- `PLANT` **grows**: an empty cell touching both `PLANT` and `WATER` sprouts more
  plant, so vines creep along waterlines from a seed. It burns fast like dry
  grass. (This is the engine's first material-*creating* rule.)
- `ACID` is a heavy corrosive liquid that **dissolves** the solids it touches
  (`WALL`, `SAND`, `WOOD`, `PLANT` → gone) and slowly **evaporates** — and it
  **flash-boils to `SMOKE`** the instant it meets `FIRE` or `LAVA` — so a splash of
  it bores through a structure and then runs out, faster still if it hits something
  hot.
- `GLASS` is **made by melting `SAND` in `LAVA`** — drop sand into a lava pool and
  it sets into an inert, fireproof, acid-proof solid you can build with.
- `MUD` is **wet earth**: `SAND` that touches `WATER` packs into it, so shores and
  riverbanks turn muddy — and it **bakes back to `SAND`** next to `FIRE` or `LAVA`,
  a little wet/dry cycle (water makes mud, heat dries it out again).
- `ICE` is a two-way **phase** solid: it **melts back to `WATER`** wherever it
  touches `FIRE` or `LAVA`, and it also **freezes the `WATER` it touches** into
  more ice — a slow cold front that creeps across a still pool. Both are
  frame-hashed, and because melting is faster than freezing, heat and cold settle
  into an equilibrium: a pond ices over, but a torch held to it melts a hole that
  the meltwater fills, and ice dropped on lava melts and quenches it to stone.
- `SPRING` is an inert solid that **sources `WATER`** — it never moves or depletes,
  but the empty cells around it well up with water, so it's an **endless fountain**.
  Where plant only grows where water already is, a spring needs nothing but space,
  so it keeps the world alive: rivers keep flowing, reservoirs refill, frozen ponds
  thaw back, vines keep creeping (plant needs water), and a spring set by lava is a
  perpetual steam engine.
- `VOLCANO` is the spring's **hot twin**: an inert vent that never depletes but
  **sources `LAVA`** into the empty cells around it, so it's an endless lava
  fountain. Because lava drives every heat reaction, a volcano is a self-sustaining
  chaos engine — its flow ignites oil and gas, fuses sand to glass, melts ice,
  flashes water to steam, and sets off TNT. The generated world buries a few deep
  down, so caverns near them slowly flood with lava.
- `VOID` is the **sink** to those sources: a black hole that **consumes whatever it
  touches** down to `EMPTY` and never depletes, so with gravity feeding it, material
  flows in and vanishes — a drain. Only `WALL` resists it, so you box a void in with
  stone; everything else (even lava, glass, or another source) is swallowed. Pair a
  `VOLCANO` with a `VOID` and you have an endless lava-disposal pit.
- `VIRUS` is a **self-propagating infection**: it converts the cells it touches into
  more virus and **burns itself out** to `EMPTY` over time, so it spreads as an
  expanding wave that leaves wasteland behind. `WALL` contains it and `FIRE`/`LAVA`
  **cauterise** it — so you fight a plague with a firebreak. Dab one cell into a lush
  cavern and watch it eat the place, then collapse.
- `SPARK` is **electricity**: drop one into `WATER` and a bright charge **arcs
  through the whole pool** in a single pulse, flashing the water to `STEAM` (a rising
  cloud the water cycle later rains back) and **igniting any `GAS` or `OIL`** it
  reaches — so electrocuting a flooded gas cavern sets the whole thing off. The pulse
  boils away its own conductor, so it sweeps once and dies rather than lingering.
- `TNT` is an **explosive**: touch it with `FIRE` or `LAVA` and it detonates,
  bursting into a ball of `FIRE` that consumes the soft stuff around it (sand, oil,
  wood, plant, gas) and **chain-detonates neighbouring `TNT`** — so a packed block
  goes off as a detonation wave rolling one ring per frame. `WALL`, `GLASS`, and
  `WATER` shrug it off, so a blast stops at a stone wall or fizzles at a waterline
  (and the water then quenches the flames). Lay a `TNT` fuse to a powder keg.
- **Water meets hot:** `WATER` touching `FIRE` or `LAVA` flashes to `STEAM` — so
  water **puts fires out** — while the fire is quenched and the lava freezes to
  stone (`WALL`). The `STEAM` then rises and **condenses back to `WATER`**, a
  little boil → rise → rain water cycle.

Fire and lava **shimmer** as they're drawn (an animated, render-only flicker — it
doesn't touch the simulation). Paint with the mouse and pick a material from the
on-screen palette (or keys `0`-`9`, `P` plant, `A` acid, `M` smoke, `G` glass, `I` ice, `S` spring, `T` tnt, `H` ash, `V` volcano, `X` void, `D` mud, `Z` virus, `E` spark); `[` / `]` size the brush. The palette
**wraps into a grid** so every material stays on-screen and clickable, and is the
same on all three backends; every rule — movement, the time-varying transforms, and
the neighbour reactions — is bit-identical across CPU SIMD, OpenGL, and Vulkan.

In the interactive view the **whole local world keeps simulating**, not just the
part you can see: the arrows scroll a **viewport** over a world twice its size in
each direction, all of which stays live, so panning reveals surroundings that have
gone on bubbling, burning and growing while off-screen rather than chunks frozen
where you left them. (The disk-streamed *world-larger-than-memory* is what `--bench`
demonstrates; the interactive view trades unlimited size for a fully-alive sandbox.)

The update rule is **order-independent**: each frame is a fixed sequence of
sub-passes, and within a pass every move is between a disjoint pair of cells
(vertical moves split by row parity, diagonal/horizontal by column parity). That
makes it a pure function of the previous frame — which is what lets the
massively-parallel GPU backends reproduce the CPU result **bit-for-bit**. See
[WORLD.md](WORLD.md) for the design and the Noita techniques it adapts.

## Build & run

```sh
make            # build all platforms
make cpp        # just the C++ SIMD world
./cpp/sandsim_world                       # interactive: arrows pan, number keys paint
./cpp/sandsim_world --res 1280x800 --scale 3   # window resolution + virtual-pixel size
./cpp/sandsim_world --bench 600 6 6       # headless: whole-world checksum + material counts
SANDSIM_SIMD=sse ./cpp/sandsim_world --bench 600 6 6   # force SSE (default: widest the CPU has)
make benchmark  # build all three, verify identical output, print a throughput table
```

The interactive view renders each cell as a **virtual pixel** of `scale × scale`
screen pixels, so the viewport is `winW/scale × winH/scale` cells (the simulated
world is twice that in each direction, and all of it runs). The
**physics rate is decoupled from rendering** (a fixed-timestep accumulator keyed
to real time), so the simulation runs at the same wall-clock speed on every
backend, whatever the frame rate. All configurable the same way on all three —
`--res WxH` / `--scale N` / `--sps STEPS_PER_SEC`, or `SANDSIM_RES` /
`SANDSIM_SCALE` / `SANDSIM_SPS` (default **1024×768, 2×2, 60 steps/s**).

Dependencies: a C++17 compiler + SDL2; GLEW + GLFW (OpenGL); the Vulkan SDK +
`glslc` (Vulkan).

## License

MIT — see [LICENSE](LICENSE).
