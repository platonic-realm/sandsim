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
  `STEAM`, `WOOD`, `PLANT`, `ACID`, `SMOKE`, `GLASS`. Movement is a pure density swap (heavy→light:
  `SAND > LAVA > WATER > OIL > air > GAS > FIRE`, `STEAM` lightest). On top of it
  sit the reactions, each kept order-independent so the GPU reproduces them
  exactly:
  - **time-varying transforms** — `FIRE` burning out to `EMPTY` and `STEAM`
    condensing back to `WATER` are per-cell passes that are pure functions of
    `(x, y, frame)` (no neighbour reads), so the GPU computes the identical hash.
  - **ignition** (`FIRE`/`LAVA` → `OIL`) and **water-meets-hot** (`WATER` →
    `STEAM`, `FIRE` quenched, `LAVA` → stone) are neighbour-based, which is
    normally order-*dependent* (CPU-sequential ≠ GPU-parallel). Each is made
    order-independent with **two snapshot passes through the `moved` scratch
    buffer** (free after the movement step): pass 1 reads the grid and marks the
    cells to transform, pass 2 applies the marks — so every pass reads one buffer
    and writes another. Steam thus completes a boil → rise → condense water cycle.
  - **growth** — `PLANT` next to `WATER` sprouts into adjacent empty cells (a
    frame-hashed mark/apply pair where each empty cell decides from a snapshot),
    the first rule that *creates* material rather than moving or transforming it.
  - **glassmaking** — `SAND` touching `LAVA` sets into `GLASS`, an inert solid
    (another mark/apply snapshot pair), so sand poured into a lava pool fuses.
  - **corrosion** — `ACID` dissolves the solids it touches (`WALL`/`SAND`/`WOOD`/
    `PLANT` → `EMPTY`) and slowly evaporates; same two-pass snapshot shape.

  All reaction passes are gated by a per-world flag set when fire/lava enters, so
  fire-free worlds (like the `--bench` seed) skip them and the cross-backend
  reference checksums are unchanged. Each rule was verified bit-identical by
  temporarily seeding the reactive materials across all three backends.
- **Chunk** = `CHUNK × CHUNK` cells (`CHUNK = 64`) of material ids. The world is
  `wbox × hbox` chunks; chunks live on disk and are generated (from a
  deterministic seed) the first time they're needed.
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
