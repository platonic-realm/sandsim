# Big world: chunked streaming (Noita-style)

The [materials engine](MATERIALS.md) simulates one screen-sized grid. To
simulate a world much larger than memory or the screen — the way
[Noita](https://en.wikipedia.org/wiki/Noita_(video_game)) does — sandsim adds a
**chunked, disk-streamed world**: only a few "live boxes" of the world are kept
in memory and simulated/rendered at a time, and the rest is saved to disk and
reloaded on demand.

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

The same ideas, simplified so they can be implemented identically in several
languages and verified deterministically.

- **Chunk** = `CHUNK × CHUNK` cells (`CHUNK = 64`) of material ids, plus a
  **dirty rectangle** (the sub-box to simulate next frame) and an `awake` flag.
  An asleep chunk (empty dirty rect) is skipped.
- **World** = a sparse map of resident chunks keyed by chunk coordinate, plus a
  **disk directory**. Cells are addressed in global coordinates; a read of an
  unresident chunk returns `WALL` (so material never falls into the void).
- **Camera / live region.** A box of chunks around the camera is kept resident
  — the "live boxes." Each step:
  - chunks that fall outside the live region (plus a margin) are **saved to
    disk and evicted**;
  - chunks that enter the live region are **loaded from disk**, or **generated**
    from a deterministic function if they have never existed.
- **Simulation.** Awake resident chunks are processed in a fixed order
  (bottom chunks first; within a chunk, bottom-to-top) using the materials rule
  via global cell access. A per-cell **moved flag** (reset each frame, spanning
  chunk borders) stops any cell being stepped twice and makes the chunk
  processing order safe. A move near a chunk edge **wakes** the neighbor.
- **Sleeping.** After a frame, each chunk's next dirty rect is the bounding box
  of cells that moved, grown by one cell. A chunk with no moves goes to sleep;
  painting or an incoming particle wakes it again.

The dirty rect is kept at sub-chunk granularity (a bounding box within the
chunk). True per-pixel dirty rects and multithreaded checkerboard passes are
noted as further refinements.

### What the SIMD variant does

`cpp/sandsim_world_simd.cpp` reuses the **multi-buffer SIMD** technique from
`cpp/sandsim_sse_mb.cpp` / `sandsim_avx_mb.cpp`: the interleaved layout
`cells[(y*W + x)*LANES + lane]` lets a single vector load read the same cell
across `LANES` boxes at once, so one SIMD instruction advances `LANES` live
boxes in parallel. Here the lanes are **live boxes of the streamed world**
rather than independent windows. Box interiors advance in lockstep across the
lanes; a scalar pass resolves the one-cell flow across box borders, and the same
streaming/eviction keeps only a working set of boxes resident.

## Verifying it

Each implementation has a headless `--bench` that builds a fixed, finite,
wall-bordered world (a few chunks), then runs a deterministic camera sweep that
forces most chunks out to disk and back while the simulation runs. It prints one
`RESULT` line with a checksum over the **entire** world (resident + on-disk
chunks, in chunk order) and the per-material counts.

Two properties are checked:

1. **Conservation.** Because the world is finite and wall-bordered, every
   material count is invariant for the whole run — even as chunks are evicted to
   disk and reloaded. This proves the streaming round-trip is lossless.
2. **Cross-language agreement.** All ports use identical fixed-width integer
   arithmetic, the same generation, the same camera path, and the same update
   order, so they produce the **same checksum**. `make bench-world` runs them
   all and asserts agreement.

Note that with streaming, chunks that are evicted while still active **freeze**
until reloaded — the intended Noita-like behavior — so the streamed result is
its own deterministic thing, not identical to simulating the whole world
resident at once.

## Running

```sh
make world            # build the chunked-world implementations
cpp/sandsim_world     # interactive: WASD/arrows pan the camera, number keys paint
make bench-world      # deterministic cross-language streaming cross-check

# render the whole streamed world (resident + on-disk chunks) to an image:
cpp/sandsim_world --ppm world.ppm 600 6 6 && magick world.ppm world.png
```
