# Big world: chunked streaming (Noita-style)

A single screen-sized grid of materials (sand, water, gas, walls) is easy. To
simulate a world much larger than memory or the screen — the way
[Noita](https://en.wikipedia.org/wiki/Noita_(video_game)) does — sandsim uses a
**chunked, disk-streamed world**: only a small "live window" of the world is kept
in memory and simulated/rendered at a time, and the rest is saved to disk and
reloaded on demand. The same engine runs on the CPU (SIMD) and the GPU
(OpenGL/Vulkan compute), bit-for-bit identical.

## How Noita does it (research)

From the 2019 GDC talk *Exploring the Tech and Design of Noita* and the Noita
wiki:

- **Two chunk granularities.** The world is divided into **64×64-pixel chunks**
  for *simulation*, and larger **512×512 chunks** for *disk streaming*.
- **Dirty rectangles.** Each 64×64 chunk keeps a **dirty rectangle** — the
  bounding box of pixels that might still move. Only pixels inside it are
  simulated; the rect shrinks as things settle. A chunk whose dirty rect is
  empty is **asleep** and skipped entirely. A pixel that moves into a sleeping
  chunk **wakes** it.
- **"Updated this frame" flag.** When a pixel moves it is flagged so the scan
  doesn't move it again the same frame (a pixel that fell into a not-yet-visited
  cell would otherwise be processed twice).
- **Streaming to disk.** Only a small number of chunks around the player are
  kept resident (Noita's `STREAMING_CHUNK_TARGET` defaults to ~12). Chunks that
  drift out of range are serialized to disk and freed; chunks coming into range
  are read back, or generated if never visited. `StreamingKeepAliveComponent`
  keeps the chunks under an entity loaded.
- **Multithreading.** Non-adjacent chunks are updated in parallel. Because a
  pixel only ever touches its immediate neighborhood, chunks that are at least
  one chunk apart can be processed concurrently without conflicts — a
  checkerboard of update passes.

Sources:
[GDC talk notes](https://braindump.jethro.dev/posts/gdc_vault_exploring_the_tech_and_design_of_noita/),
[80.lv writeup](https://80.lv/articles/noita-a-game-based-on-falling-sand-simulation),
[Noita wiki: StreamingKeepAliveComponent](https://noita.wiki.gg/wiki/Documentation:_StreamingKeepAliveComponent).

## Our adaptation

The same ideas, simplified so the **one** engine can run on the CPU and on the
GPU and produce a **bit-identical** world.

- **Materials** = `EMPTY`, `WALL`, `SAND`, `WATER`, `GAS`, `OIL`, `FIRE`, `LAVA`,
  `STEAM`, `WOOD`, `PLANT`, `ACID`, `SMOKE`, `GLASS`, `ICE`, `SPRING`, `TNT`, `ASH`, `VOLCANO`, `VOID`, `MUD`, `VIRUS`, `SPARK`, `OBSIDIAN`, `SALT`, `SNOW`, `MERCURY`, `GUNPOWDER`, `THERMITE`, `FROST`, `WISP`, `COAL`, `EMBER`. Movement is a pure density swap (heavy→light:
  `MERCURY > SAND > LAVA > ACID > WATER > OIL > SNOW > air > GAS > FIRE`, `STEAM` light, `WISP` lightest of all). On top of it
  sit the reactions, each kept order-independent so the GPU reproduces them
  exactly. The density extremes are deliberately *one-sided* and cheap: `MERCURY` is
  heaviest, so it only ever sinks (its own target set lists everything; no other
  material needs to know about it), and `WISP` is the exact inverse — lightest, so it
  only ever rises, bubbling up *through* every liquid and gas (the first material that
  rises through a liquid instead of being trapped beneath it) to gather at the ceiling,
  again touching only its own target set. The reactions:
  - **time-varying transforms** — `FIRE` burning out to `EMPTY` and `STEAM`
    condensing back to `WATER` are per-cell passes that are pure functions of
    `(x, y, frame)` (no neighbour reads), so the GPU computes the identical hash.
  - **ignition** (`FIRE`/`LAVA` igniting the `OIL`, `GAS`, `PLANT` and flammable `WISP`
    it touches, and smouldering `WOOD`) and **things-meet-hot** (`WATER` → `STEAM`, `ACID` →
    `SMOKE`, `FIRE` quenched, `LAVA` → `OBSIDIAN`) are neighbour-based, which is normally
    order-*dependent* (CPU-sequential ≠ GPU-parallel). Each is made order-independent
    with **two snapshot passes through the `moved` scratch buffer** (free after the
    movement step): pass 1 reads the grid and marks the cells to transform, pass 2
    applies the marks — so every pass reads one buffer and writes another. Steam thus
    completes a boil → rise → condense water cycle.
  - **growth** — `PLANT` next to `WATER` sprouts into adjacent empty cells (a
    frame-hashed mark/apply pair where each empty cell decides from a snapshot),
    the first rule that *creates* material rather than moving or transforming it.
  - **glassmaking** — `SAND` touching `LAVA` sets into `GLASS`, an inert solid
    (another mark/apply snapshot pair), so sand poured into a lava pool fuses.
  - **mud** — a wet/dry cycle folded into one mark/apply pass: `SAND` next to
    `WATER` packs into `MUD`, and `MUD` next to `FIRE`/`LAVA` bakes back to `SAND`
    (pass 1 marks which way each cell goes, pass 2 applies). Mud forms naturally
    along the generated world's shores, so the benchmark exercises it directly.
  - **melting & freezing** — `ICE` is a two-way phase solid: it thaws to `WATER`
    next to `FIRE`/`LAVA`, and freezes the `WATER` it touches into more `ICE` (both
    frame-hashed mark/apply pairs). Freezing is slower than melting, so a pond ices
    over as a creeping cold front while heat melts holes back into it — the two
    reach an equilibrium. The meltwater also feeds the existing water rules, so ice
    dropped on lava melts and then quenches the lava to obsidian.
  - **corrosion** — `ACID` dissolves the solids it touches (`WALL`/`SAND`/`WOOD`/
    `PLANT` → `EMPTY`) and slowly evaporates; same two-pass snapshot shape.
  - **sourcing** — a `SPRING` solid wells `WATER`, and a `VOLCANO` solid wells
    `LAVA`, up into the empty cells around it (the empty cell decides from a
    snapshot, like plant growth). Neither depletes, so they're endless generators —
    the rules that create mass from nothing and keep long-running worlds in motion
    instead of settling (a volcano in particular feeds every heat reaction). The
    generated world buries a few of each, so the streaming benchmark covers them.
    Their opposite is the `VOID` **sink** — a black hole that consumes any neighbour
    (mark/apply snapshot; everything but `WALL` and `VOID` is cleared to `EMPTY`),
    so material flows in under gravity and vanishes. It's paint-only (kept out of
    the generated world), verified bit-identical by seeding it into `worldgen.h`.
  - **detonation** — `TNT` (and pourable `GUNPOWDER`) touched by `FIRE`/`LAVA` bursts into `FIRE` across its
    8-neighbourhood and chain-detonates adjacent `TNT`. The two passes are unusual:
    pass 1 marks the *detonators* (TNT next to something hot) into the scratch
    buffer; pass 2 turns a cell to `FIRE` if it is a detonator **or** a blastable
    cell next to one — so pass 2 reads the pass-1 marks of its neighbours (a stable
    snapshot) plus its own grid cell, writing only itself. That keeps it order-
    independent and GPU-identical even though the blast reaches a full ring outward,
    and the wave then advances one ring per frame as the new fire reaches more TNT.
  - **thermite** — `THERMITE` is the same detonation *shape* turned into a slow burn-
    through: pass 1 marks every `THERMITE` cell touching `FIRE`/`LAVA` (the cells that
    ignite this frame); pass 2 turns each marked cell into `FIRE` and melts every
    adjacent meltable solid (`WALL`/`GLASS`/`OBSIDIAN`/`SAND`/`WOOD`) into `LAVA` —
    reading the pass-1 marks of its 4 neighbours plus its own cell, so it stays order-
    independent. Because the ignition chains one ring per frame and leaves molten lava
    behind, a pile poured on stone eats a cavity straight through it — the only rule
    that destroys the otherwise-indestructible `WALL`/`GLASS`/`OBSIDIAN`. Paint-only,
    verified by a unit test plus a `worldgen.h` chamber that agrees bit-for-bit.
  - **frost** — `FROST` is the cold mirror of fire's spread: one combined mark/apply
    pass marks each cell `1` (`WATER` next to frost → `FROST`, the advancing edge),
    `3` (a `FROST` cell → `ICE`, crystallising — it's only ever the leading edge),
    `4` (`PLANT` next to frost → `EMPTY`, a killing frost) or `5` (`FROST` next to
    `FIRE`/`LAVA` → `WATER`, melted), then applies. The advance is *deterministic*
    (every water cell touching frost freezes), so the freeze front never starves the
    way a probabilistic one would — it sweeps a whole connected pool to ice at one
    cell per frame and self-terminates where the water runs out. The melt path is the
    interesting one: a frost front hitting lava turns back to water, which the earlier
    **quench** pass flashes to steam while forging the lava to obsidian — so the
    boundary *resolves to a fixed point* rather than oscillating frost↔water forever
    (verified: a full-pipeline chamber freezes solid and stops changing by frame 38).
    Paint-only, verified bit-identical via a `worldgen.h` chamber.
  - **smouldering** — `COAL` is a pourable fuel that burns *slowly* rather than flashing.
    One combined mark/apply pass marks each cell `1` (a `COAL` touching `FIRE`/`LAVA` or an
    `EMBER`, so it catches → `EMBER`) or `2` (a cell that is an `EMBER` this frame); then
    pass 2 turns the `1`s into `EMBER`, ages each `EMBER` down to `ASH` on a slow frame-hash
    (else it keeps burning), and turns an `EMPTY` cell next to an `EMBER` (pass-1 mark `2`)
    into `FIRE` on a faster frame-hash. The propagation is the point: embers spread
    **ember-to-coal through the pile itself**, so the fire creeps along regardless of which
    way the loose flames drift up — unlike `WOOD`, which only burns where the flame happens
    to reach. The radiated `FIRE` lights neighbouring oil/gas/wood, and because each ember
    only ages to ash on a slow hash, a coal bed is a *lasting* heat source. Paint-only,
    verified by a unit test (a lit pile spreads to ember and burns down to ash) plus a
    `worldgen.h` chamber that agrees bit-for-bit.
  - **infection** — `VIRUS` self-propagates: one combined mark/apply pass marks each
    cell `1` (a consumable neighbour of a virus, so it gets infected) or `2` (a virus
    that burns out or is cauterised by `FIRE`/`LAVA`, so it dies to `EMPTY`), then
    applies. Because spread outpaces decay it expands as a wave that leaves emptiness
    behind, contained by `WALL`. Paint-only, verified by seeding it into `worldgen.h`.
  - **electricity** — `SPARK` conducts through `WATER`: one combined mark/apply pass
    marks each cell `1` (water next to a spark → spark), `2` (a spark next to water →
    `STEAM`, boiling off), `3` (an isolated spark → `EMPTY`) or `4` (`GAS`/`OIL` next
    to a spark → `FIRE`). Boiling the water it passes — rather than handing it back —
    is what makes the pulse sweep a pool once and terminate instead of oscillating
    spark↔water forever. Paint-only, verified by a static unit test plus a walled
    `worldgen.h` chamber where it agrees bit-for-bit and ignites ~1900 gas cells.
  - **de-icing** — `SALT` is the cold counterpart to a torch: one combined mark/apply
    pass marks each cell `1` (salt touching water → `EMPTY`, dissolved) or `2` (ice
    touching salt → `WATER`, melted with no heat), then applies. Both frame-hashed, so
    a finite sprinkle melts a finite patch of ice and then dissolves into the
    meltwater. Paint-only; verified by a static unit test and a `worldgen.h` seed.
  - **poisoning** — `MERCURY` (the heaviest material, a liquid metal everything
    floats on) is toxic: a `PLANT` cell touching it withers to `EMPTY` (frame-hashed
    mark/apply). The density tier is the cheap end to add — nothing is heavier than
    mercury, so no other material's targets change; it only needed its own
    `belowMerc` (sand + everything below) and DOWN/HORIZ eligibility. Paint-only;
    movement verified with a standalone step test (mercury sinks below all),
    poisoning with a unit test, bit-identity with a `worldgen.h` seed.

  All reaction passes are gated by a per-world flag set when a reactive material
  is present, so a world of only sand/water/rock pays nothing for them. The
  generated world *is* full of lava, oil, acid, plant and the rest, so the
  streaming `--bench` run exercises the entire reaction set every step — which
  turns the benchmark's checksum-equality assertion into a continuous proof that
  all of it stays bit-identical across CPU SIMD, OpenGL and Vulkan (it does).
- **Chunk** = `CHUNK × CHUNK` cells (`CHUNK = 64`) of material ids. The world is
  `wbox × hbox` chunks; chunks live on disk and are generated the first time
  they're needed from a single deterministic, hash-based seed shared by all three
  backends (`worldgen.h`) — an open sky over rolling ground, a sandy crust, and an
  underground of rock veined with caverns and clustered pools of every liquid and
  gas, lava welling up from the deep, and the rare buried spring, TNT cache or
  deep lava-spewing volcano.
- **Live window.** A `gw × gh` box of chunks around the camera is kept resident in
  **one contiguous grid** with a `PAD`-cell `WALL` border (the border keeps
  material from falling into the void and gives the GPU/SIMD offset accesses a
  safe halo). The window is sized at runtime — the interactive view derives it
  from the display (`winW/scale × winH/scale` cells; default 1024×768 @ 2× →
  `8 × 6` chunks), while `--bench` fixes it at `4 × 4` so its cross-backend
  checksums are a stable reference. As the camera moves, chunks that leave the
  window are **saved to disk and evicted**; chunks that enter are **loaded from
  disk**, or **generated** if never visited.
- **Simulation.** The whole live window is stepped each frame by the
  order-independent rule below. A per-cell **moved flag** (reset each frame) gives
  one-move-per-frame priority.

### The order-independent rule

The reason all three backends agree bit-for-bit is that the update is a **pure
function of the previous frame**. A naive "scan and move" rule is
order-dependent (a cell's result depends on whether its neighbor was already
processed this frame) and so can't be reproduced by a massively parallel GPU.
Instead each frame is a fixed sequence of **16 disjoint sub-passes**
([`cpp/simd_core.h`](cpp/simd_core.h)), and within a pass every move is between a
*disjoint* pair of cells, so no two moves touch the same cell:

- **down / down-left / down-right** (sand, water) and **up / up-left / up-right**
  (gas): vertical moves are split by **row parity**, diagonals by **column
  parity**, so sources and targets never collide.
- **horizontal-left / -right** (water, gas): a same-row swap chains neighbor to
  neighbor, so it is split into **even/odd column phases** — even sources move
  into odd targets, then odd into even.

On the **CPU** this is SIMD: the lanes are 16 (SSE) or 32 (AVX2) **adjacent cells
of one contiguous grid**, so material flows freely across the whole region (not
independent boxes — SIMD lanes don't communicate). The width is chosen at
runtime; both widths compute the same result. See
[cpp/README.md](cpp/README.md).

On the **GPU** (OpenGL / Vulkan) the same 16 passes are 16 compute dispatches.
The key to making the in-place update race-free: a thread decides whether its
cell is the **source** of a move purely from its `(x,y)` coordinates (the
parity / boundary pattern), never from cell content — so non-source threads
touch nothing and only one thread ever writes a given cell. The horizontal
boundary skip that the SIMD cross-lane shift implies is replicated on the GPU
(`cx % 16`), so the GPU and CPU agree exactly.

### GPU streaming

The GPU keeps the live window in a device buffer where the step runs, and
streams chunks to/from disk just like the CPU so the world can be far larger
than VRAM. OpenGL keeps a CPU shadow and syncs the buffer only on camera moves;
Vulkan maps a host-visible buffer the CPU streams into directly. Either way the
per-frame simulation stays on the GPU, and the same seed + same camera path
gives the same world.

## Verifying it

Every backend has a headless `--bench [steps] [wbox] [hbox]` that builds a
fixed, finite, wall-bordered world, runs a deterministic camera sweep (forcing
most chunks out to disk and back), and prints one `RESULT` line: a checksum over
the **entire** world (resident + on-disk chunks, in chunk order), the
per-material counts, and disk I/O.

Two properties are checked:

1. **Conservation.** The world is finite and wall-bordered, so every material
   count is invariant for the whole run even as chunks are evicted and reloaded —
   proving the streaming round-trip is lossless (`conserved=yes`).
2. **Bit-identical backends.** C++ SIMD, OpenGL, and Vulkan use the same seed,
   camera path, and order-independent rule, so they produce the **same
   checksum**. `make benchmark` runs all three and fails loudly on any mismatch.

A window of `GW × GH` chunks (`--bench N 4 4`) is fully resident — no eviction —
which is the simplest bit-identical check; larger boxes exercise streaming.

## Running

```sh
make                   # build cpp + opengl + vulkan
cpp/sandsim_world      # interactive: arrows pan the camera, number keys paint
cpp/sandsim_world --res 1280x800 --scale 3   # window resolution + virtual-pixel size
make benchmark         # build all three, assert identical checksum, print a table

# render a snapshot of the live window to an image:
cpp/sandsim_world --ppm world.ppm 600 && magick world.ppm world.png
```
