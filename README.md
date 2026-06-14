# sandsim

A multi-material falling-sand **world**: a huge grid of materials (sand, water,
lava, oil, acid, gas, rock and more) that is chunked and streamed to/from disk
around a camera, so the world can be far larger than memory. It generates as a
varied landscape — an open sky over rolling ground with **snow-capped peaks**, and
underground caverns packed with clustered pools of every liquid and gas, **coal
seams, salt deposits and silvery mercury pools**, lava welling up from the deep, the
odd buried spring or cache of TNT, **pockets of bubbling marsh gas and geothermal
geyser vents that pulse steam on a cycle** — and then comes alive as the materials
react (coal catches where lava finds it, snow melts at the warm edges, mercury pools
at the bottom, geysers gush steam that rains back, marsh gas flares near a lava vein).
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
`SNOW` (powder), `MERCURY`, `GUNPOWDER` (powder), `THERMITE` (powder), `FROST`, `WISP`, `COAL` (powder), `EMBER` (powder), `CLONER`, `CRYSTAL`, `ANTIMATTER`, `MOSS`, `FUMES`, `WIRE`, `EHEAD`, `ETAIL`, `IGNITER`, `SENSOR`, `LIFE`, `GEYSER`, `LYE` (powder), `SODIUM` (powder), `CORAL`, `PHOSPHORUS` (powder), `CEMENT` (powder), `CHLORINE` (gas), `BATTERY`, `FUSE`, `CRYO` (liquid), `LAMP`, `PETRIFY`, `FIREWORK`, `LEVITON` (powder), `SPROUT`, `BELT`, `MAGNET`, `IRON` (powder), `NITRO` (liquid), `RUST` (powder), `SEED` (powder), `LASER`, `BEAM`, `ICICLE`. Movement is a
density swap — heaviest to lightest is `MERCURY > SAND > LAVA > ACID > WATER > OIL >
SNOW > air > GAS > FIRE`, with `STEAM`/`SMOKE` the lightest — so sand sinks through
lava, acid sinks below water, oil floats on water, and gas/fire/steam/smoke rise (and `FUMES` and `CHLORINE` are the odd gases that **sink** -- heavy vapours that pool in the low ground).
`ASH`, `GUNPOWDER`, `THERMITE`, `COAL`, `EMBER`, `LYE`, `SODIUM`, `PHOSPHORUS`, `CEMENT`, `IRON`, `RUST` and `SEED` fall and pile like sand;
`SNOW` is lighter than every liquid, so it falls through air but **floats on water
and oil**; `MERCURY` is the heaviest of all, so **everything floats on it**, and `WISP` is the lightest, so **it rises through everything** (even liquids). `WALL`,
`WOOD`, `PLANT`, `GLASS`, `ICE`, `SPRING`, `TNT`, `VOLCANO`, `VOID`, `MUD`, `VIRUS`,
`OBSIDIAN`, `SALT`, `CLONER`, `CRYSTAL`, `CORAL`, `ANTIMATTER`, `MOSS`, `FUSE`, `PETRIFY`, `BELT`, `MAGNET`, `LASER`, `BEAM`, `ICICLE`, and the `WIRE`/`EHEAD`/`ETAIL`/`IGNITER`/`SENSOR`/`BATTERY`/`LAMP` circuitry, Conway `LIFE` cells and `GEYSER` vents are solids that don't move.
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
- `MOSS` is a **creeping overgrowth** — the third way things grow. Where `PLANT` needs a
  waterline and `CRYSTAL` branches into open air, moss **coats solid surfaces**: it spreads
  along the thin skin of empty cells hugging `WALL`, `OBSIDIAN`, `GLASS` or `WOOD`, so it
  greens over walls and timber and climbs them like ivy on a ruin — without ever filling
  open space (it has nothing to cling to there). It's **flammable**, so a torch clears it
  off in a flash; left alone, it slowly reclaims any stonework you build.
- `FUMES` are a **heavy flammable vapour** — the exact mirror of `GAS`. Where gas rises to
  the ceiling, fumes **sink through the air and pool in the low ground**, spreading out flat
  across cavern floors (they're the `SNOW` density tier, so they **float on every liquid**
  and heavier things sink straight through them). And like gas they're **flammable** — a
  `SPARK` or a stray flame flashes the whole pocket at once — so a pit that quietly fills
  with fumes is a bomb waiting for a trigger. The invisible-danger counterpart to a gas
  cloud overhead.
- `WIRE` + `EHEAD` + `ETAIL` are a whole **[Wireworld](https://en.wikipedia.org/wiki/Wireworld)
  computer** bolted onto the sandbox — the digital counterpart to the analog `SPARK`. Lay
  copper `WIRE` and drop an electron (a bright head `EHEAD` chased by a dim tail `ETAIL`) onto
  it, and the electron **races along the wire** by the classic rules: a head becomes a tail,
  a tail becomes wire, and a bare wire turns into a head exactly when **one or two** of its
  eight neighbours are heads. That one rule gives you everything — **diodes, AND/OR/XOR gates,
  clocks, memory** — so you can build working logic, even a clock that ticks forever round a
  loop, right in the middle of the falling sand. (It's a synchronous cellular automaton, which
  is exactly what the engine's two-pass snapshot computes.)
- `IGNITER` is the **bridge from those circuits to the physical world** — a spark plug for
  your logic. It sits inert until a Wireworld electron reaches it, then **spits `FIRE` into
  the cells around it**. So wire a clock to one for a timed detonator, or a gate for a
  triggered trap: a circuit can now light `FUMES`, set off `GUNPOWDER`/`TNT`, fire a cannon
  or burn down a structure on a pulse. The output side of everything `WIRE` can compute.
- `SENSOR` is the **other bridge — physical to digital**, the input that completes the loop.
  It's inert until any *real* material (a liquid, a powder, fire — anything that isn't empty
  space, wall or circuitry) touches it, at which point it **fires an electron into the `WIRE`
  beside it**. So a circuit can finally *read* the world: put a sensor at a water line for a
  flood alarm, in a cavern roof for a lava detector, or under a hopper to count what piles up.
  With `SENSOR` (input), `WIRE`/`EHEAD`/`ETAIL` (logic) and `IGNITER` (output) you can build a
  whole **machine that senses, decides and acts** — feedback loops, automatic traps, contraptions.
- `LIFE` is **[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)** —
  the *other* famous cellular automaton, sharing the sandbox with Wireworld. A live cell with two
  or three live neighbours survives, an empty cell with exactly three is born, everything else
  dies. Paint a **glider** and watch it sail across the screen, a **blinker** to tick, a glider
  gun to spew them forever. The twist is that it shares the grid with the physics: falling sand
  and water **block births** where they land and **mow down** patterns, so a stream of sand
  through a glider gun is a chaotic, deterministic collision of a cellular automaton and a
  falling-sand world.
- `GEYSER` is a **geothermal vent on a timer** — the first *rhythmic* thing in the world. Where
  a `SPRING` trickles water and a `VOLCANO` oozes lava forever, a geyser holds its breath and
  then **erupts in a burst of `STEAM`**, again and again on a cycle. The plume rises, condenses
  through the water cycle and rains back down, so a buried geyser drives a slow heartbeat of
  steam and rain — pulsing, alive, never quite the same. Bury one under a pool and watch it
  gush.
- `LYE` is a **caustic powder — the chemical opposite of `ACID`.** It falls and piles like
  sand, but where lye and acid touch they **neutralise each other into products**: acid + base
  → salt + water, so the `ACID` is spent to `WATER` and the `LYE` to `SALT`. It's the first
  reaction where *two reactive materials cancel one another* rather than one consuming the
  other — drizzle lye onto an acid pool and watch the etching stop as the brine settles out.
- `SODIUM` is a **soft alkali-metal powder — the one explosive that water *sets off* instead of
  putting out.** It falls and piles like sand and sits inert while dry, but the instant it
  touches `WATER` (or any flame) it **flares to `FIRE` and flashes the touching water to
  `STEAM`** — the real *2Na + 2H₂O → 2NaOH + H₂* reaction is sharply exothermic and the
  hydrogen it frees ignites. The fire it makes is itself hot, so a sodium pile chain-reacts
  outward one ring per frame, and the steam rises and rains back through the water cycle.
  Pour it freely, then add a single drop of water and stand back. (The chemical sibling of
  `ACID`/`LYE`/`SALT` — and a violent counterpart to the inert `SAND` it resembles.)
- `CORAL` is a **living reef that grows dendritically — underwater.** Where `CRYSTAL`
  crystallises into bare air, `PLANT` creeps along the waterline and `MOSS` coats stone,
  coral spreads through `WATER` *itself*: a water cell turns to coral when exactly one of its
  eight neighbours is coral, so a single seed dropped in a pool **branches into a reef**,
  consuming the water as it goes (the one-neighbour rule keeps it lacy instead of flooding
  solid). It's alive, so heat **bleaches** it — a coral touching `FIRE` or `LAVA` dies to
  `ASH`. The first growth bound to a liquid substrate; seed a flooded cavern and watch it fill.
- `PHOSPHORUS` is the **mirror image of `SODIUM` — where sodium bursts into flame on contact
  with *water*, white phosphorus ignites on contact with *air*.** So you have to keep it
  **submerged**: a grain touching an empty cell spontaneously catches `FIRE` (a frame-hash gives
  it a brief, shimmering delay), while a grain walled in by water or solids sits perfectly
  stable. It catches instantly from any flame too, so a pile flares from its exposed surface
  inward. Store a cache underwater, then drain the pool — and the whole lot goes up at once.
  (Sodium and phosphorus make a neat pair: one fears water, the other needs it.)
- `CEMENT` is the **first building material — a pourable powder that hardens into stone.** Pour
  it like sand to fill a mould or cavity, and once a grain comes to **rest on something** it
  cures: a frame-hash slowly turns it to `WALL`, so a settled pile sets from the supported cells
  upward like drying concrete (a grain still falling through air never freezes in mid-air). It's
  the counterpart to the destroyers — where `ACID`, `THERMITE` and `ANTIMATTER` eat `WALL` away,
  cement pours new wall back, so you can carve *and* rebuild the world.
- `CHLORINE` is a **heavy green toxic gas** that sinks and pools along the ground like `FUMES`.
  Its signature is real chemistry: where chlorine meets `SODIUM` the two combine into `SALT`
  (2Na + Cl₂ → 2NaCl — literally the reaction that makes table salt, closing a loop between
  two materials already in the world). It's also poisonous to life — a chlorine cell touching
  `PLANT`, `MOSS` or `CORAL` **bleaches it away** — and any stray gas slowly disperses, so the
  cloud thins instead of lasting forever. Flood a cavern with it, then drop sodium in to watch
  it crystallise into brine.
- `BATTERY` is a **power source for circuits — the first material that *generates* signals on
  its own.** The Wireworld kit already had an input (`SENSOR`), logic (`WIRE`/`EHEAD`/`ETAIL`)
  and an output (`IGNITER`), but no autonomous source: every circuit had to be hand-pulsed. Lay
  a battery against a wire and, on a steady clock, it **injects an electron pulse** into the
  adjacent wire — so circuits finally **run hands-free**. Wire a battery → wire → igniter next to
  a cache of `GUNPOWDER` and you've built a self-firing cannon; loop it through some logic and
  you've got a ring oscillator or a blinking display. Where a `SENSOR` waits for the world to
  poke it, a battery needs nothing at all.
- `FUSE` is a **detonator cord — the classic way to *time* an explosion.** A length of fuse is
  inert until lit, then it burns along itself at a crisp **one cell per frame**, so you can route
  a long winding cord from a safe corner to a cache of `TNT` and light the far end. The burning
  tip leaves a short trail of `FIRE` that detonates or ignites whatever the cord runs into — so a
  fuse pairs with *every* explosive (`TNT`, `GUNPOWDER`, `SODIUM`, `PHOSPHORUS`) and any flammable.
  Light it with a `FIRE`/`LAVA`/`EMBER` touch, or wire it to an `IGNITER` for a circuit-triggered
  charge: a battery → wire → igniter → fuse → bomb is a fully automatic, hands-free demolition.
- `CRYO` is **cryogenic coolant (liquid nitrogen) — the first *cold* liquid and the pourable
  counterpart to `LAVA`.** It flows and pools like a light liquid (it floats on water the way
  `OIL` does), and it is fiercely cold: `WATER` it touches **flash-freezes to `ICE`**, `FIRE` it
  touches is **snuffed out**, and `LAVA` it touches is chilled straight to `OBSIDIAN` (without the
  `STEAM` a water dousing makes). Being volatile it boils away on its own, so a pour is temporary.
  Flood a lake to skate across it, freeze a moat solid, or quench a lava flow cleanly — the cold
  mirror of everything fire does.
- `LAMP` is a **circuit-driven light — the Wireworld kit's visual output.** The kit already had an
  input (`SENSOR`), logic, a power source (`BATTERY`) and a physical output (`IGNITER`), but no way
  to *see* a signal. A lamp is a dark bulb that **glows whenever an electron (`EHEAD`/`ETAIL`)
  passes a cell next to it** and dims the instant the pulse leaves. It never touches the circuit —
  it only watches — so a row of lamps beside a wire **lights in sequence as a pulse runs past (a
  marquee)**, and a battery-clocked wire makes a lamp **blink**. Build glowing signs, bar displays,
  running lights — animated art driven by your circuits.
- `PETRIFY` is a **creeping stone-curse — medusa for the sandbox.** Drop it on anything *living*
  — `PLANT`, `WOOD`, `MOSS`, `CORAL` — and the curse **sweeps through the connected greenery one
  cell per frame, turning life to stone** and leaving a frozen `OBSIDIAN` statue in its wake (the
  wavefront is the curse, the trail is rock). It's finite — living matter is consumed and every
  curse-cell settles — so a petrification always burns out, leaving you a permanent obsidian
  sculpture of whatever you turned. Petrify a forest, a moss-clad wall, or a whole coral reef.
- `FIREWORK` is a **self-launching rocket — the one material whose motion is driven by a
  *reaction* instead of gravity.** Paint a firework and it **climbs straight up one cell per
  frame** (it rises dead straight, not dispersing like a gas, because the rocket pass moves it
  cell by cell) until a timer — or a ceiling — sets it off, when it **bursts into a ball of
  `FIRE`.** Spray a handful for a **fountain of sparks popping into fireballs at different
  heights**; sit one under a `CLONER` for an endless show, or light a `FUSE` to a stash for a
  timed finale. (Bury it and it pops at once — give your rockets open sky.)
- `LEVITON` is **anti-gravity dust — the exact mirror of `SAND`.** It falls *up*: a grain rises
  and **piles into inverted dunes against the ceiling**, sliding up the sides of the heap just as
  sand slides down a mound (it rises and slides diagonally up but doesn't spread flat, so it
  builds shapes rather than dispersing like a gas). It's lighter than everything, so heavier
  materials **sink straight through it** — drop sand on a leviton cloud and they pass clean
  through each other, one falling, one rising. Pour it under an overhang to fill it from below.
- `SPROUT` is a **growing tree.** Paint a sprout at ground level and a tip **climbs upward,
  laying down a `WOOD` trunk** behind it, until it tops out and **unfurls a leafy `PLANT`
  crown** — plant a row and a whole forest rises, each trunk a slightly different height. The
  grown tree is ordinary wood and plant, so it does everything they do: it **burns**, it
  **petrifies to stone**, and its leaves **creep further near water**. (It uses the same
  reaction-driven climb as `FIREWORK`, but builds instead of bursting.) Sit one under a `CLONER`
  to plant an endless orchard.
- `BELT` is a **conveyor belt — the start of an automation layer.** It's a static machine that
  **carries the loose material resting on it sideways (to the right):** lay a row of belts and any
  loose stuff on top — sand, ore, even a trickle of water — **marches along** and piles up where
  the line ends or a wall blocks it. Feed a belt with a `CLONER` for an endless supply, route the
  stream into a furnace of `LAVA`, drop it off a ledge to sort by where it lands — build factories.
  The belt never moves; it's the *riders* that travel, claimed one cell at a time so they shuffle
  along single-file.
- `MAGNET` and `IRON` are **magnetism, at last.** `IRON` is a heavy steel powder that pours and
  piles like sand — until it touches a `MAGNET`, when it **magnetises and clings**. The field
  carries through the clinging iron, so a grain that sticks magnetises the next, and a poured heap
  **accretes into chains and clumps reaching out from the magnet**. Pour iron over a magnet, or
  run a stream of it there on a `BELT`, and it collects itself into the lodestone — a magnetic
  ore-sorter. (Loose iron away from any magnet is just a heavy grey powder you can pour and pile.)
- `RUST` is **iron's foundry cycle.** Leave `IRON` sitting in `WATER` or `ACID` and it slowly
  **corrodes to `RUST`** — a crumbly orange powder that pours and piles just like the iron it
  came from. Hold rust to `FIRE` or `LAVA` and it **smelts back to `IRON`**, good as new. So a
  damp dump of iron rots to rust over time, and a furnace reclaims it: pour iron into a flooded
  pit to watch it rust away, then rake the rust through a lava channel to forge it back — an
  endlessly recyclable metal. (The corrosion is gradual and probabilistic, so heaps rust from
  the wet edges inward rather than all at once.)
- `SEED` is **a forest in a grain.** Scatter seeds like sand and they pour and pile; wherever one
  **comes to rest on solid ground with `WATER` in reach it germinates** into a `SPROUT`, which
  then climbs on its own and grows a whole tree (`WOOD` trunk, `PLANT` canopy). So rain a handful
  of seeds across a wet valley floor and watch a forest erupt — and since the result is ordinary
  wood and leaves, it burns, petrifies and spreads exactly like any other planting. (A seed with
  no water, or one still falling through the air, just sits there as a grain until it lands somewhere
  damp.)
- `LASER` and `BEAM` are **a cutting ray.** A `LASER` is a fixed emitter that fires a `BEAM`
  straight to its right; the beam **extends one cell per frame** through open space, holds as a
  continuous red line for as long as the emitter feeds it, **burns the flammables it strikes**
  (`WOOD`, `PLANT`, `OIL`, `GAS`, `WISP`, `MOSS`, `FUMES`, `COAL`, `SEED`) into `FIRE`, and is
  **stopped cold by anything solid or wet**. Lay one across a cavern as a tripwire that torches a
  forest or an oil slick, aim it at a `FUSE` to set off a charge, or block it with a wall to switch
  it off. (The beam doesn't fall — it's pure energy — and cut the emitter and the ray winks out.)
- `ICICLE` is **dripstone — the downward mirror of a growing tree.** Where a `SPROUT` climbs
  *up* laying a wood trunk, an icicle grows *down*: paint a tip on the underside of a ceiling or
  an overhang and it **descends one cell per frame, leaving a hanging spear of `ICE` behind it**,
  until it reaches a floor or tapers off and the tip itself freezes solid. Because the body is
  ordinary ice it glistens, **melts back to water near `FIRE`/`LAVA`** and thaws against `SALT`,
  so a cave roof can grow a fringe of icicles that drip away the moment the heat rises. (Hang a
  row of them for a frozen portcullis, or grow one over a fire to watch it melt.)
- `NITRO` is a **flowing liquid explosive — the one that *floods*.** Where `TNT` is a static block
  and `GUNPOWDER` a loose pile, nitro is a water-density liquid: **pour it into a fortress's cracks
  or flood a whole chamber**, then touch it with any flame and the entire connected pool
  **chain-detonates** into fire. It shares the TNT/gunpowder blast system, so nitro and TNT set
  each other off — wick a `FUSE` to a flooded vault, or drip `LAVA` in from above, and bring the
  walls down from the inside. (Volatile: keep it well clear of lava and fire until you mean it.)
- **Water meets hot:** `WATER` touching `FIRE` or `LAVA` flashes to `STEAM` — so
  water **puts fires out** — while the fire is quenched and the lava forges into
  `OBSIDIAN`, the glassy black volcanic rock (an inert, fire/acid/blast-proof solid
  you can farm wherever lava meets water). The `STEAM` then rises and **condenses
  back to `WATER`**, a little boil → rise → rain water cycle.

Fire and lava **shimmer** as they're drawn (an animated, render-only flicker — it
doesn't touch the simulation), and emissive materials — fire, lava, embers, sparks,
lit lamps, lasers and beams, fireworks, burning fuses, electron heads — cast a soft
**additive bloom/glow** into their surroundings, so a flame lights up the cave around
it and a laser leaves a luminous streak. (Both effects are render-only in the CPU/SDL
viewer; the simulation is untouched.) Paint with the mouse and pick a material from the
on-screen palette (or keys `0`-`9`, `P` plant, `A` acid, `M` smoke, `G` glass, `I` ice, `S` spring, `T` tnt, `H` ash, `V` volcano, `X` void, `D` mud, `Z` virus, `E` spark, `O` obsidian, `L` salt, `N` snow, `Q` mercury, `B` gunpowder, `K` thermite, `F` frost, `W` wisp, `C` coal, `R` ember, `U` cloner, `Y` crystal, `J` antimatter, `;` moss, `,` fumes, `.` wire, `/` electron-head, `'` electron-tail, `-` igniter, `=` sensor, `\` life, ``` geyser; `LYE` and any newer material are palette-click only, as every single-character key is now taken). **Left-click paints, right-click erases, middle-click eyedrops** (picks the material under the cursor), the **mouse wheel** (or `[` / `]`) sizes the brush, `SPACE` pauses/resumes, `TAB` single-steps one frame while paused (watch a reaction unfold frame by frame), and `DEL`/`BACKSPACE` clears the canvas to empty air. The palette
**wraps into a grid** so every material stays on-screen and clickable, and is the
same on all three backends; every rule — movement, the time-varying transforms, and
the neighbour reactions — is bit-identical across CPU SIMD, OpenGL, and Vulkan.

The palette in the CPU/SDL viewer is **grouped into labelled categories** — Tools, Solids,
Powders, Liquids, Gases, Fire & Heat, Explosives, Life & Growth, Circuits, Machines and
Sources & Magic — with a thin category-coloured accent under each swatch so related
materials cluster and read at a glance instead of scattering in add-order.

The CPU/SDL viewer also draws a small **HUD** (rendered with a built-in bitmap font, so
no font dependency): a **bottom info bar** showing the selected material's name and
**category**, the brush size, FPS and the key controls; a **hover tooltip** that names
(and categorises) whatever is under the cursor — any palette swatch or any cell in the
world, so the 70-material palette is finally legible — a **circular brush outline** that previews exactly what you'll paint;
a live **FPS** readout; and a centred **PAUSED** banner. (UI/UX polish like this is
per-viewer and need not be bit-identical across backends — only the simulation is.)

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
