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
  `STEAM`, `WOOD`, `PLANT`, `ACID`, `SMOKE`, `GLASS`, `ICE`, `SPRING`, `TNT`, `ASH`, `VOLCANO`, `VOID`, `MUD`, `VIRUS`, `SPARK`, `OBSIDIAN`, `SALT`, `SNOW`, `MERCURY`, `GUNPOWDER`, `THERMITE`, `FROST`, `WISP`, `COAL`, `EMBER`, `CLONER`, `CRYSTAL`, `ANTIMATTER`, `MOSS`, `FUMES`, `WIRE`, `EHEAD`, `ETAIL`, `IGNITER`, `SENSOR`, `LIFE`, `GEYSER`, `LYE`, `SODIUM`, `CORAL`, `PHOSPHORUS`, `CEMENT`, `CHLORINE`, `BATTERY`, `FUSE`, `CRYO`. Movement is a pure density swap (heavy→light:
  `MERCURY > SAND > LAVA > ACID > WATER > OIL > SNOW > air > GAS > FIRE`, `STEAM` light, `WISP` lightest of all). On top of it
  sit the reactions, each kept order-independent so the GPU reproduces them
  exactly. The density extremes are deliberately *one-sided* and cheap: `MERCURY` is
  heaviest, so it only ever sinks (its own target set lists everything; no other
  material needs to know about it), and `WISP` is the exact inverse — lightest, so it
  only ever rises, bubbling up *through* every liquid and gas (the first material that
  rises through a liquid instead of being trapped beneath it) to gather at the ceiling,
  again touching only its own target set. `FUMES` are the odd gas out: a heavy vapour in
  the `SNOW` density tier (heavier than air, floats on every liquid) that *sinks* and pools
  in the low ground while spreading flat like a gas — the mirror of `GAS`, which rises. The
  reactions:
  - **time-varying transforms** — `FIRE` burning out to `EMPTY` and `STEAM`
    condensing back to `WATER` are per-cell passes that are pure functions of
    `(x, y, frame)` (no neighbour reads), so the GPU computes the identical hash.
  - **ignition** (`FIRE`/`LAVA` igniting the `OIL`, `GAS`, `PLANT`, `MOSS` and flammable
    `WISP`/`FUMES` it touches, and smouldering `WOOD`; a `SPARK` lights the same fuels) and
    **things-meet-hot** (`WATER` → `STEAM`, `ACID` →
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
  - **duplication** — `CLONER` copies the material directly above it into the empty cell
    directly below, so it's a faucet of whatever you feed it (`SPRING`/`VOLCANO` well a
    *fixed* material; the cloner wells *any*). This is the first pass to carry a **material
    id** through the scratch buffer rather than a bare flag: pass 1 stores, for each
    `CLONER`, the cloneable material sitting above it (skipping `EMPTY`/`WALL`/`CLONER`, so
    it can't extrude structure or replicate itself); pass 2 fills any `EMPTY` cell whose
    upper neighbour is such a loaded cloner with that stored id. Pass 2 reads only the
    pass-1 scratch of its neighbour plus its own grid cell — never a neighbour's *live* grid
    cell — so there is no read/write race and it stays order-independent / GPU-identical.
    The source on top is only read, never consumed, so one drop is an endless supply.
    Paint-only, verified by a unit test plus a `worldgen.h` chamber that agrees bit-for-bit.
  - **crystallisation** — `CRYSTAL` grows dendritically into bare air: the first growth
    rule that needs neither a waterline (`PLANT`) nor a host to consume (`FROST`/`VIRUS`).
    A frame-hashed mark/apply pair where each `EMPTY` cell decides from a snapshot, but the
    condition is the classic dendrite rule: it crystallises only when **exactly one** of
    its *eight* neighbours is already crystal. A tip extends where a single arm reaches,
    while any gap flanked by two arms (>=2 crystal neighbours) locks and never fills, so the
    growth stays a delicate branching gem instead of a solid flood (verified: a 600-frame
    seed grows to a few hundred cells with **zero** cells having >=3 orthogonal crystal
    neighbours). Paint-only, verified bit-identical via a `worldgen.h` chamber.
  - **annihilation** — `ANTIMATTER` disintegrates any *matter* it touches, the same
    two-pass detonation *shape* as `TNT`/`THERMITE` but with no trigger and no survivors:
    pass 1 marks every `ANTIMATTER` cell that has a non-`EMPTY`, non-`ANTIMATTER` neighbour;
    pass 2 turns each marked cell to `FIRE` (the energy release) and clears every matter
    cell next to one to `EMPTY` -- so it eats through `WALL`, `LAVA`, `WATER`, *anything*.
    The crucial property is that antimatter is **never created, only consumed**: matter goes
    to `EMPTY`, antimatter goes to `FIRE`, so its count strictly decreases and the reaction
    always terminates (making the eaten matter into more antimatter would be a non-stopping
    world-eater). A solid blob therefore peels to fire from the outside in over a few frames
    -- the exposed inner layers annihilate against the fire their own surface just made --
    carving a clean cavity its own size. Paint-only, verified by a unit test (a blob buried
    in `WALL` eats its shell and burns itself out) plus a `worldgen.h` chamber that agrees
    bit-for-bit.
  - **overgrowth** — `MOSS` is the third growth mode: where `PLANT` needs a waterline and
    `CRYSTAL` branches into open air, moss coats *surfaces*. A frame-hashed mark/apply pair
    where an `EMPTY` cell becomes `MOSS` only when it is adjacent to existing moss **and** to
    a stone/timber anchor (`WALL`/`OBSIDIAN`/`GLASS`/`WOOD`) -- so it spreads only along the
    thin skin of empty cells hugging a surface, greening walls and climbing them like ivy
    without filling open space (the anchor is deliberately *not* moss itself, or it would
    flood). It joins the **ignition** pass as instant fuel, so a torch clears it off in a
    flash. Paint-only, verified by a unit test (moss creeps a floor and climbs a wall, every
    cell still touching stone -- it never floats free) plus a bit-identical `worldgen.h` chamber.
  - **wireworld** — `WIRE`/`EHEAD`/`ETAIL` are a whole [Wireworld](https://en.wikipedia.org/wiki/Wireworld)
    cellular automaton, so you can build digital logic (diodes, gates, clocks) out of copper
    wire carrying electrons. Unlike every other rule it is a *synchronous CA* -- a cell's next
    state depends only on the current states of its neighbours -- which is exactly what the
    two-pass snapshot computes, except here the scratch buffer carries the **next material id**
    (like the cloner) rather than a flag: pass 1 writes each cell's next state (`EHEAD`→`ETAIL`,
    `ETAIL`→`WIRE`, and a `WIRE` → `EHEAD` iff 1 or 2 of its 8 neighbours are heads; `0` means
    "not a wire cell"), pass 2 applies it. The 1-or-2-heads rule is the whole of digital logic:
    a head races forward leaving a tail that stops it backing up, a fork that meets 3 heads
    stays quiet (a gate), and a loop of wire clocks forever. Paint-only, verified by a unit test
    (a pulse travels one cell per frame the correct way; a wire loop circulates an electron for
    60 frames) plus a `worldgen.h` clock-loop chamber that agrees bit-for-bit across all three.
  - **ignition output** — `IGNITER` bridges the Wireworld circuits back to the physical sim: it
    runs right after the wireworld pass and, by the same two-pass snapshot, marks every
    `IGNITER` next to an electron head and then turns the `EMPTY` cells next to a marked one
    into `FIRE`. The timing is the subtle part -- it detects the `EHEAD` *after* the CA pass,
    which is exactly when the head has propagated into the wire cell touching the igniter, so a
    pulse arriving fires a one-shot flame burst (verified by a unit test: an electron crossing
    the igniter produces a fire burst, and the igniter is inert with no pulse). That makes a
    clock a timed detonator and a gate a triggered one -- a circuit can light `FUMES`, set off
    `GUNPOWDER`/`TNT` or burn a structure on cue. Verified bit-identical via a `worldgen.h`
    clock-driven-igniter-over-gunpowder chamber.
  - **sensing** — `SENSOR` is the opposite bridge, physical → digital, completing the I/O: it
    marks every sensor that has a *detectable* neighbour (anything that isn't empty/wall/the
    circuitry itself) and turns the `WIRE` next to a marked sensor into an electron head -- so a
    flood, a lava flow or a piling powder injects a signal the logic can act on. Because the
    sensor *creates* electrons, it's the one exception to the presence-gate's "never created by
    another reaction" rule, so the wireworld pass is also gated on a sensor being present (else
    a sensor-made electron wouldn't propagate). Together `SENSOR` (in), `WIRE`/`EHEAD`/`ETAIL`
    (logic) and `IGNITER` (out) are a full sense→decide→act machine. Verified by a unit test
    (water touching a sensor injects an electron that travels the wire; inert when untouched)
    plus a `worldgen.h` chamber wiring a water-tripped sensor through to an igniter over
    gunpowder, bit-identical across all three.
  - **game of life** — `LIFE` is [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life),
    a second synchronous CA in the sandbox: a `LIFE` cell with 2 or 3 live neighbours of 8
    survives, an `EMPTY` cell with exactly 3 is born, else death/stay-empty. Same two-pass
    snapshot as wireworld -- pass 1 marks each `LIFE`-or-`EMPTY` cell's fate (`1`=live, `2`=empty,
    `0`=leave, which keeps every *other* material untouched), pass 2 applies. The leave-other-
    materials-alone rule is what makes it collide with the physics: falling sand and water aren't
    empty, so they block births and mow down patterns -- a glider sailing through a sand stream
    is a deterministic clash of a cellular automaton and a falling-sand world. Paint-only,
    verified by a unit test (a blinker oscillates period-2, a block is a still life, a glider
    translates by (1,1) every 4 steps) plus a bit-identical `worldgen.h` glider+blinker chamber.
  - **eruption** — `GEYSER` is the first *rhythmic* source: where `SPRING`/`VOLCANO` well their
    material every frame, a geyser only erupts during a window of a global cycle (`frame % 150 <
    30`), puffing `STEAM` into the empty cells around it on a frame-hash, then falling dormant.
    The whole pass is gated on the eruption window -- when dormant it writes nothing (the CPU
    early-returns leaving the grid *and* scratch untouched; the GPU writes a blank scratch and
    applies nothing -- identical because the geyser is the last pass, so scratch is never read
    after it). The steam rises and condenses back to water through the existing cycle, so a vent
    drives a slow heartbeat of steam and rain. The global frame window is the same `frame`-driven
    determinism the time-varying transforms use, so it stays bit-identical (verified: a unit test
    shows it erupts exactly the 60-of-300 expected frames and emits *nothing* in all 240 dormant
    frames, plus a `worldgen.h` geyser chamber agrees across all three over multiple cycles).
  - **neutralisation** — `LYE` is the caustic counterpart of `ACID`, and where the two touch
    they cancel into products: acid + base → salt + water, so a 2-pass mark/apply turns each
    `ACID` cell adjacent to lye into `WATER` and each `LYE` cell adjacent to acid into `SALT`.
    It's the first reaction where *two reactive materials destroy one another* rather than one
    consuming the other -- both inputs are spent, both outputs are pre-existing materials. Because
    lye *creates* `SALT`, it breaks the "no gated input is made by another reaction" invariant, so
    the salt-cycle pass is gated on `present[SALT] || present[LYE]` (else the freshly minted brine
    would never dissolve). Verified: a unit test (an acid pool drizzled with lye neutralises at the
    interface, acid→water and lye→salt, conserved) plus a `worldgen.h` lye-on-acid chamber that
    agrees bit-for-bit across CPU SIMD, OpenGL and Vulkan over 300 frames.
  - **water-explosion** — `SODIUM` is a sand-density alkali-metal powder and the only explosive
    that `WATER` *triggers* rather than quenches. A 2-pass snapshot (passes 62/63) marks each
    `SODIUM` cell touching `WATER` or something hot, then turns every marked cell into `FIRE` and
    flashes each `WATER` beside a marked cell to `STEAM` (the exothermic *2Na + 2H₂O → 2NaOH + H₂*
    reaction, the hydrogen igniting). Sodium is only ever *consumed* (never created by any rule),
    so its count strictly decreases and the chain terminates; the `FIRE` it makes is hot, so a
    pile self-propagates one ring per frame, and the steam rejoins the boil→rise→rain water cycle.
    Like `TNT`/`GUNPOWDER` it is inert until triggered, so it is *not* in the `hasReactive` set —
    its trigger (`WATER`/`FIRE`) already is. Verified: a unit test (sodium+water→fire+steam,
    sodium+heat→fire, dry sodium stays inert, count strictly decreases) and a movement test
    (rests exactly like `SAND`, sinks through water) plus a `worldgen.h` sodium-on-water chamber
    bit-identical across all three backends over 300 frames.
  - **reef growth** — `CORAL` is the first growth bound to a *liquid* substrate: where `CRYSTAL`
    grows into air, `PLANT` along the waterline and `MOSS` over stone, coral spreads through
    `WATER`. A 2-pass snapshot (passes 64/65) marks each `WATER` cell with *exactly one* coral
    neighbour-of-eight (a frame-hash gating the rate, so reefs branch dendritically and cells
    flanked by two arms lock instead of flooding — the same trick `CRYSTAL` uses), and also marks
    each `CORAL` touching `FIRE`/`LAVA` for bleaching; pass 2 turns the first into `CORAL` and the
    second into `ASH`. Coral consumes the water it grows into, so a reef is bounded by its pool
    and terminates. Both its triggers — the `WATER` it grows into and the `FIRE`/`LAVA` that
    bleaches it — are already reactive, so (like `SODIUM`/`LYE`) coral is *not* in `hasReactive`;
    the `present[CORAL]` gate suffices and is safe (coral is only ever created by its own pass).
    Verified: a unit test (a seed branches through a pool without flooding, stays inert in air
    with no water, bleaches to ash beside fire, deterministic) plus a `worldgen.h` flooded-chamber
    reef bit-identical across all three backends over 300 frames.
  - **air-ignition** — `PHOSPHORUS` is the mirror of `SODIUM`: where sodium ignites on contact
    with `WATER`, white phosphorus ignites on contact with *air*. A 2-pass snapshot (passes 66/67)
    marks each `PHOSPHORUS` cell that is next to `FIRE`/`LAVA` (an instant catch) or next to an
    `EMPTY` cell *and* the frame-hash fires (a brief spontaneous delay); pass 2 turns each marked
    cell into `FIRE`. A grain walled in by water or solids has no empty neighbour and stays inert,
    so a cache survives underwater until the pool drains; a pile in air burns from its exposed
    surface inward as the fire it makes catches the next grain. **Unlike `SODIUM`/`LYE`, its
    spontaneous trigger — `EMPTY` air — is *not* reactive, so `PHOSPHORUS` must itself be in the
    `hasReactive` set (like `VIRUS`/`CRYSTAL`), or a phosphorus-only pocket would never run its
    pass on a backend that skips reactions; `present[PHOSPHORUS]` then gates it and is safe (it is
    only ever painted, never created by a rule).** Verified: a unit test (air-exposed grain
    ignites, submerged grain stable over 500 frames, instant catch beside lava, a pile burns
    surface-first, deterministic), a movement test (rests exactly like `SAND`, sinks through
    water), and two `worldgen.h` chambers — a dry phosphorus pile in air (isolating the
    `hasReactive` edit) and a half-submerged cache — each bit-identical across all three backends.
  - **curing** — `CEMENT` is the first *building* material: a sand-density powder that hardens into
    `WALL`. A 2-pass snapshot (passes 68/69) marks each `CEMENT` cell that is *supported* (the cell
    directly below is not empty) and whose frame-hash fires, then turns it to `WALL`; a grain still
    falling through air (empty below) is never marked, so it can't freeze in mid-air, and a settled
    pile cures from the supported cells upward like drying concrete. The product is the existing
    `WALL`, so no new material is needed, and it is the counterpart to the wall-eaters
    (`ACID`/`THERMITE`/`ANTIMATTER`). Like `PHOSPHORUS`, its trigger is *not* a reactive material
    (it cures on its own timer), so `CEMENT` must itself be in the `hasReactive` set, or a
    cement-only pile would never cure on a backend that skips reactions; `present[CEMENT]` then
    gates it and is safe (only ever painted). Verified: a unit test (a supported grain cures, an
    unsupported grain stays loose over 1000 frames, a resting column cures fully, deterministic), a
    movement test (rests exactly like `SAND`, sinks through water), and a cement-only `worldgen.h`
    chamber — isolating the `hasReactive` edit — bit-identical across all three backends.
  - **toxic gas** — `CHLORINE` is a heavy green gas wired to move exactly like `FUMES` (it sinks
    and pools), and the first material with gas-phase chemistry. A 2-pass snapshot (passes 70/71)
    marks each cell's fate: a `CHLORINE` (or `SODIUM`) cell beside the other becomes `SALT`
    (2Na + Cl₂ → 2NaCl — closing a loop with two existing materials); a `CHLORINE` beside `PLANT`,
    `MOSS` or `CORAL` is spent bleaching that cell to `EMPTY`; and any stray chlorine disperses to
    `EMPTY` on a frame-hash. Chlorine is only ever consumed (never created), so it terminates.
    Because it makes `SALT`, the salt-cycle pass is gated on `present[SALT] || present[LYE] ||
    present[CHLORINE]` (the same fix `LYE` needed), and because it disperses on its own timer it
    is in the `hasReactive` set (like `PHOSPHORUS`/`CEMENT`). Adding a new gas is the most
    invasive movement change — chlorine's id had to join every `canEnter`/eligibility site where
    `FUMES` appears, across `simd_core.h` and both GPU shaders — so it was verified hardest: the
    default bench is unchanged, a WALL-bordered CPU test confirms chlorine moves *bit-identically*
    to `FUMES` over 150 frames, a reaction unit test covers Na→salt/bleach/disperse, and two
    `worldgen.h` chambers (chlorine-only, and chlorine + sodium + plant) agree across all three
    backends.
  - **power source** — `BATTERY` is the missing autonomous source for the Wireworld kit (input
    `SENSOR`, logic `WIRE`/`EHEAD`/`ETAIL`, output `IGNITER` — but nothing that *starts* a signal).
    A 2-pass snapshot (passes 72/73) marks each `WIRE` cell touching a battery on a pulse frame
    (`frame % 12 == 0`, the same global-frame determinism geyser uses), then lights it to `EHEAD`;
    on the other eleven frames it writes nothing. Because it manufactures electrons, it joins the
    wireworld gate (`present[EHEAD] || present[ETAIL] || present[SENSOR] || present[BATTERY]`) —
    the same fix `SENSOR` needed, since otherwise an injected head would never propagate — and
    because it acts on its own clock with no reactive trigger it is in the `hasReactive` set (like
    `PHOSPHORUS`/`CEMENT`/`CHLORINE`). It runs after the wireworld pass, so an injected head begins
    travelling the next frame; the result is a clean clock that lets circuits run hands-free
    (a self-firing igniter, a ring oscillator). Verified: a unit test (a wire beside a battery
    lights to a head on pulse frames and only then, the battery persists and converts only wire,
    exactly two pulses in 24 frames, deterministic) plus a battery+wire `worldgen.h` chamber —
    isolating *both* the `hasReactive` and wireworld-gate edits at once — bit-identical across all
    three backends.
  - **fuse** — `FUSE` is a detonator cord that burns along itself at a crisp one cell per frame, so
    explosions can be *timed and routed*. It uses a second material, `BURNFUSE`, as the travelling
    tip. A 2-pass snapshot (passes 74/75) marks each `FUSE` cell that catches — from an adjacent
    `BURNFUSE`, `FIRE`, `LAVA` or `EMBER` — and each `BURNFUSE` cell (which always burns out); pass 2
    turns the first into `BURNFUSE` and the second into `FIRE`. Because the tip lives exactly one
    frame before becoming fire, the burn advances exactly one cell per frame and leaves a short
    fading trail of flame, and that fire detonates or ignites whatever the cord meets — so the
    detonation reuses the existing fire mechanics with no special-casing of `TNT`/`GUNPOWDER`/etc.
    `FUSE` is inert until lit, so (like `TNT`) it is *not* in `hasReactive`; `BURNFUSE` burns out on
    its own, so it *is* (the reaction is gated on `present[FUSE] || present[BURNFUSE]`, and the
    `present[]` latch keeps it running until every tip has burned out). Verified: a unit test (a cord
    catches from fire, the tip advances exactly one cell per frame, it leaves a fire trail, an unlit
    cord stays inert, lava lights it too, deterministic) plus two `worldgen.h` chambers — a
    `BURNFUSE`-only block isolating the `hasReactive` edit, and a full fuse → `TNT` contraption —
    bit-identical across all three backends.
  - **cryo** — `CRYO` (cryogenic coolant) is the first *cold* liquid, the pourable mirror of
    `LAVA`. Its movement is wired to be identical to `OIL` (a light liquid that floats on water), so
    a new liquid tier was added without a bespoke density: `CRYO`'s id joins every `canEnter`/
    eligibility site where `OIL` appears, across `simd_core.h` and both GPU shaders. Its reaction
    (a 2-pass snapshot, passes 76/77) marks each cell's fate — `WATER` next to cryo flash-freezes to
    `ICE`, `FIRE` next to cryo is snuffed to `EMPTY`, `LAVA` next to cryo is chilled to `OBSIDIAN`
    (no `STEAM`, unlike a water quench), and any cryo next to `FIRE`/`LAVA` or whose frame-hash fires
    boils off to `EMPTY` — then applies. Cryo is only ever consumed, so it terminates, and because it
    evaporates on its own timer it is in the `hasReactive` set (like `PHOSPHORUS`/`CEMENT`/
    `CHLORINE`). Adding a liquid is the most invasive movement change, so it was verified hardest:
    the default bench is unchanged, a WALL-bordered CPU test confirms `CRYO` moves *bit-identically*
    to `OIL` over 150 frames, a reaction unit test covers freeze/snuff/chill/boil-off, and two
    `worldgen.h` chambers (cryo-only, and cryo + water + lava) agree across all three backends.
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

  **Presence gating.** Each reaction is skipped unless its trigger material can
  actually be in the live grid — a `present[]` latch set when a material is loaded
  or painted. The safety argument is that every *gated* reaction's input is a
  material that is never created by *another* reaction (only by itself, or never),
  so while its flag is false the pass is a guaranteed **no-op**; skipping it can't
  change the result, which keeps all three backends bit-identical (the always-on
  core — fire/ignite/quench/plant/glass/melt/freeze/mud — handles the materials
  that *are* created mid-frame). It's set-and-never-cleared, so a stale flag only
  costs an extra no-op pass, never correctness. In a world of just sand/water/rock
  the dozen-plus paint-only reactions (virus, frost, crystal, antimatter, moss,
  wireworld, …) cost nothing; the win is real — gating roughly **doubled** the CPU
  benchmark and sped the GPU backends up by tens of percent, with the cross-backend
  checksum unchanged (verified against the un-gated build, both for the default
  world and a chamber seeded with every paint-only reaction at once).

  The generated world *is* full of lava, oil, acid, plant, tnt, coal, salt and the
  rest, so the streaming `--bench` run still exercises most of the reaction set every
  step — which turns the benchmark's checksum-equality assertion into a continuous
  proof that all of it stays bit-identical across CPU SIMD, OpenGL and Vulkan (it does).
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
