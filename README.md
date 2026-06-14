# sandsim

A multi-material falling-sand **world**: a huge grid of materials (sand, water,
lava, oil, acid, gas, rock and more) that is chunked and streamed to/from disk
around a camera, so the world can be far larger than memory. It generates as a
varied landscape — an open sky over rolling ground with **snow-capped peaks**, and
underground caverns packed with clustered pools of every liquid and gas, **coal
seams, salt deposits and silvery mercury pools**, lava welling up from the deep, and
the odd buried spring or cache of TNT — and then comes alive as the materials react
(coal catches where lava finds it, snow melts at the warm edges, mercury pools at the
bottom).
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
`TNT`, `ASH` (powder), `VOLCANO`, `VOID`, `MUD`, `VIRUS`, `SPARK`, `OBSIDIAN`, `SALT`,
`SNOW` (powder), `MERCURY`, `GUNPOWDER` (powder), `THERMITE` (powder), `FROST`, `WISP`, `COAL` (powder), `EMBER` (powder), `CLONER`, `CRYSTAL`, `ANTIMATTER`. Movement is a
density swap — heaviest to lightest is `MERCURY > SAND > LAVA > ACID > WATER > OIL >
SNOW > air > GAS > FIRE`, with `STEAM`/`SMOKE` the lightest — so sand sinks through
lava, acid sinks below water, oil floats on water, and gas/fire/steam/smoke rise.
`ASH`, `GUNPOWDER`, `THERMITE`, `COAL` and `EMBER` fall and pile like sand;
`SNOW` is lighter than every liquid, so it falls through air but **floats on water
and oil**; `MERCURY` is the heaviest of all, so **everything floats on it**, and `WISP` is the lightest, so **it rises through everything** (even liquids). `WALL`,
`WOOD`, `PLANT`, `GLASS`, `ICE`, `SPRING`, `TNT`, `VOLCANO`, `VOID`, `MUD`, `VIRUS`,
`OBSIDIAN`, `SALT`, `CLONER`, `CRYSTAL`, and `ANTIMATTER` are solids that don't move.
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
  the meltwater fills, and ice dropped on lava melts and quenches it to obsidian.
- `SALT` is a **de-icer**: the `ICE` it touches **melts to `WATER` with no heat at
  all**, and the salt then **dissolves away** in that meltwater — so a sprinkle on a
  frozen pond thaws a patch and vanishes. A finite amount of salt melts a finite
  amount of ice, the cold counterpart to a torch.
- `SNOW` is a **light powder** — lighter than every liquid, so it falls through air
  but **floats** on water and oil instead of sinking (sand and the rest pour straight
  through it). Like ice, it **melts to `WATER` next to `FIRE` or `LAVA`**, so a
  snowdrift slumps into a puddle at the first spark.
- `MERCURY` is the opposite extreme: the **heaviest** material, a liquid metal that
  sinks below everything and pools at the very bottom, so sand, water and oil all
  **float on top of it** (the mirror of snow). It's also **toxic** — the `PLANT` it
  touches withers to nothing — so a silver puddle is both a curiosity and a hazard.
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
- `GUNPOWDER` is **TNT you can pour** — a black powder that falls and piles like
  sand, so you can run it into trails, cracks and heaps, then **detonate it with a
  spark**: it blasts and chain-detonates through itself exactly like TNT. Pour a
  fuse of it from a torch to a buried cache and watch the line race away.
- `THERMITE` is a **powder that burns through stone** — pile it like sand, light it
  with `FIRE`/`LAVA`, and it ignites so hot it **melts the solid `WALL`, `GLASS`,
  `OBSIDIAN`, `SAND` or `WOOD` it touches into molten `LAVA`** before combusting away.
  The burn chains through the pile one ring per frame, so a heap dumped on a stone
  floor eats a glowing cavity straight down through it — the only thing in the world
  that can melt the otherwise-indestructible `WALL`/`GLASS`/`OBSIDIAN`. The lava it
  leaves then drives every other heat reaction.
- `FROST` is **a fire made of cold** — the mirror of fire's spread. Drop a seed into a
  pool and a **freezing wave races out across the `WATER`, crackling the whole pond
  over into `ICE`** one cell per frame, leaving a solid sheet behind. It **withers the
  `PLANT` it touches** (a killing frost) and is itself **melted back to `WATER` by
  `FIRE`/`LAVA`** — so a torch carves a hole in advancing ice, and the meltwater
  flashes to steam on the lava. Where fire needs fuel and burns out, frost needs water
  and freezes solid; throw heat at a frost front to stop it dead.
- `WISP` (a will-o'-the-wisp of marsh gas) is the **lightest thing in the world** — the
  exact inverse of `MERCURY`. Where everything floats *on* mercury, wisp rises *through*
  everything: it **bubbles up through water, oil, acid, even lava and mercury** to
  collect against the ceiling, the first material that rises through a liquid instead of
  being trapped under it. And it's **flammable** — `FIRE`, `LAVA` or a `SPARK` lights it —
  so a bubble climbing through a flooded cavern flashes the instant it breaks the surface
  into a flame, and a pocket gathered under a stone roof goes off like a gas main.
- `COAL` is a **pourable fuel** you pile like sand — and the world's first *lasting* fire.
  Light a heap with `FIRE` or `LAVA` and it catches into `EMBER`, glowing burning coal
  that **spreads through the pile** (ember to coal, so it creeps along no matter which way
  the loose flames drift), **radiates `FIRE` into the gaps around it** — setting light to
  any oil, gas or wood nearby — and slowly **burns down to `ASH`**. Where oil and gas
  flash away in a single frame, a coal bed smoulders for a long while, so it's the way to
  keep a forge hot: a steady heat source to boil water, melt ice or cook off `THERMITE`
  long after the match is gone.
- `CLONER` is a **duplicator** — the generic source. Where a `SPRING` only ever wells
  water and a `VOLCANO` only lava, a cloner wells **whatever you feed it**: it copies the
  material sitting directly on top of it into the empty cell directly below, without ever
  using up the original. Drip one drop of anything into it — lava, acid, mercury, even
  snow — and it becomes an **endless faucet** of that material, so you can plumb a whole
  contraption from a single seed (a `VOLCANO` of your choosing). Stack it over a `VOID`
  for an infinite source-and-drain.
- `CRYSTAL` is a **growing mineral** — drop a seed and it **branches out into the empty
  space around it like a gem or coral**, the first thing in the world that grows into bare
  air (where `PLANT` needs a waterline and `FROST`/`VIRUS` consume a host). The trick is
  that a cell only crystallises when **exactly one** of its neighbours is already crystal,
  so any gap flanked by two arms locks shut — the growth stays a delicate fractal dendrite
  instead of flooding solid. Grow a glittering thicket to fill a cavern, then bury it,
  smash a `VOLCANO` through it, or hand it to a `CLONER`.
- `ANTIMATTER` is a **disintegration charge** — the one thing that destroys *anything*.
  It sits inert in a vacuum, but the instant it touches matter it **annihilates it to
  nothing in a flash of `FIRE`**, eating clean through even `WALL`, `LAVA` and `WATER` —
  the materials that stop fire, acid and every explosion. A blob consumes itself and the
  matter around it from the outside in over a few frames, carving out a cavity its own
  size and then burning out (it's never created, only spent, so it always stops). Lay a
  line of it to bore a tunnel through solid rock, or drop a chunk to simply delete a hole
  in the world.
- **Water meets hot:** `WATER` touching `FIRE` or `LAVA` flashes to `STEAM` — so
  water **puts fires out** — while the fire is quenched and the lava forges into
  `OBSIDIAN`, the glassy black volcanic rock (an inert, fire/acid/blast-proof solid
  you can farm wherever lava meets water). The `STEAM` then rises and **condenses
  back to `WATER`**, a little boil → rise → rain water cycle.

Fire and lava **shimmer** as they're drawn (an animated, render-only flicker — it
doesn't touch the simulation). Paint with the mouse and pick a material from the
on-screen palette (or keys `0`-`9`, `P` plant, `A` acid, `M` smoke, `G` glass, `I` ice, `S` spring, `T` tnt, `H` ash, `V` volcano, `X` void, `D` mud, `Z` virus, `E` spark, `O` obsidian, `L` salt, `N` snow, `Q` mercury, `B` gunpowder, `K` thermite, `F` frost, `W` wisp, `C` coal, `R` ember, `U` cloner, `Y` crystal, `J` antimatter); `[` / `]` size the brush. The palette
**wraps into a grid** so every material stays on-screen and clickable, and is the
same on all three backends; every rule — movement, the time-varying transforms, and
the neighbour reactions — is bit-identical across CPU SIMD, OpenGL, and Vulkan.

In the interactive view the simulated area is a little **larger than what you can
see** — the viewport plus a one-chunk **live border** all around it, all of it
stepping every frame. So panning with the arrows reveals an edge that has gone on
bubbling, burning and growing rather than chunks frozen where you left them, without
paying to simulate the whole surroundings (simulating a much larger region made the
CPU crawl). For a bigger, livelier picture at the same cost, raise `--scale` (bigger
virtual pixels = fewer cells to simulate). The disk-streamed
*world-larger-than-memory* is what `--bench` demonstrates.

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
`SANDSIM_SCALE` / `SANDSIM_SPS` (default **1024×768, 3×3, 60 steps/s**).

Dependencies: a C++17 compiler + SDL2; GLEW + GLFW (OpenGL); the Vulkan SDK +
`glslc` (Vulkan).

## License

MIT — see [LICENSE](LICENSE).
