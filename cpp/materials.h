// Material ids shared by the baseline host code and the SIMD step TUs.
#pragma once
#include <cstdint>
#include <cstddef>

// Density order (heavy -> light): SAND > LAVA > WATER > OIL > air > GAS > FIRE,
// with STEAM the lightest (rises). OIL floats on water; FIRE rises and burns out;
// LAVA is a heavy molten liquid that sets oil ablaze. Water meeting fire/lava
// flashes to STEAM, which rises and condenses back to WATER -> a little water cycle.
// WOOD is a flammable SOLID: it never moves (absent from every density group) and
// blocks like WALL, but fire/lava set it alight (slowly). PLANT is a flammable
// solid too, but it GROWS into empty space wherever it meets WATER. ACID is a
// heavy corrosive liquid that dissolves the solids it touches and evaporates.
// SMOKE is the lightest gas: some of the flame that burns out becomes smoke,
// which rises and fades away. GLASS is an inert solid -- it's made by melting
// SAND in LAVA, and it resists fire and acid. ICE is a solid too, but it MELTS
// back to WATER wherever it touches FIRE or LAVA, and FREEZES the WATER it touches
// into more ICE -- a two-way phase. SPRING is an inert solid that never moves or
// depletes but SOURCES WATER into the empty cells around it: an endless fountain.
// TNT is an explosive solid: FIRE/LAVA detonates it into a burst of FIRE that
// blasts the soft cells around it and chain-detonates neighbouring TNT. ASH is a
// light-grey powder (it falls and piles like SAND) left behind as some FIRE burns
// out -- the soot of a dying flame. VOLCANO is SPRING's hot twin: an inert vent
// that never depletes but SOURCES LAVA into the empty cells around it. VOID is the
// sink to those sources: a black hole that CONSUMES everything around it to EMPTY
// (only WALL contains it), never depleting. MUD is wet earth: SAND that touches
// WATER packs into it, and it bakes back to SAND next to FIRE/LAVA -- a little
// wet/dry cycle that muddies shores. VIRUS is a self-propagating infection: it
// converts the cells it touches into more VIRUS, burns itself out to EMPTY over
// time (so it spreads as a wave), and is cauterised by FIRE/LAVA -- WALL contains it.
// SPARK is electricity: it arcs through WATER as a one-shot pulse and ignites the
// GAS/OIL it passes. OBSIDIAN is the glassy black rock LAVA forges into when WATER
// quenches it (instead of plain stone) -- an inert, fire/acid/blast-proof solid.
// SALT is a de-icer: it MELTS the ICE it touches to WATER (no heat needed) and then
// DISSOLVES away in that water. SNOW is a light powder -- lighter than every liquid,
// so it falls through air but floats on water and oil -- that melts to WATER near
// FIRE/LAVA. MERCURY is the opposite end: the HEAVIEST material, a liquid metal that
// everything else floats on, and it's toxic -- it kills the PLANT it touches.
// GUNPOWDER is a pourable explosive: a black powder (it falls and piles like SAND)
// that detonates like TNT when FIRE/LAVA touches it, chaining through itself.
// THERMITE is an incendiary powder (it falls and piles like SAND too): once FIRE/LAVA
// lights it, it burns hot enough to MELT the solid WALL/GLASS/OBSIDIAN/SAND/WOOD it
// touches into molten LAVA, then combusts to FIRE -- so a pile eats a cavity through
// stone the way nothing else can. The ignition chains through the powder one ring/frame.
// FROST is the cold mirror of FIRE/VIRUS: a creeping freeze front, painted as a seed,
// that races across WATER turning it to ICE (seeding more FROST a step ahead so the
// wave advances and leaves an ice trail), WITHERS the PLANT it touches, and is MELTED
// back to WATER by FIRE/LAVA -- so heat stops a frost front the way water stops a fire.
// WISP (will-o'-the-wisp / marsh gas) is the LIGHTEST material -- the inverse of MERCURY:
// where everything floats on mercury, wisp rises through everything. It BUBBLES UP
// through every liquid (water, oil, acid, lava, mercury) and gas to collect at the
// ceiling, and it's FLAMMABLE -- FIRE/LAVA/SPARK ignite it -- so a flammable bubble
// rising through water flashes when it surfaces into a flame.
// COAL is a pourable fuel powder (it falls and piles like SAND): lit by FIRE/LAVA it
// catches into EMBER -- glowing burning coal -- which spreads ember-to-coal through a
// pile, RADIATES FIRE into the empty cells around it (igniting nearby fuel) for a while,
// and finally burns down to ASH. So a heap of coal lit at one corner becomes a lasting
// campfire/forge, unlike oil/gas which flash away in a frame. EMBER also falls like sand.
// CLONER is an inert solid duplicator: it copies whatever material sits directly on top
// of it into the empty cell directly below, so feeding it one drop of ANY material turns
// it into an endless faucet of that material (the generic source -- SPRING/VOLCANO well a
// fixed material; CLONER wells whatever you give it). It is the first rule to carry a
// material id through the scratch buffer rather than a bare flag.
// CRYSTAL is an inert mineral that GROWS dendritically into empty space -- the first rule
// to grow into bare air (plant creeps along waterlines, frost/virus consume a host). An
// empty cell crystallises only when EXACTLY ONE neighbour is already crystal, so it
// branches like a real gem/coral instead of flooding solid.
// ANTIMATTER is a disintegration charge: inert in a vacuum, but it DISINTEGRATES any
// matter it touches to EMPTY -- even WALL/LAVA/WATER, which stop fire and explosions --
// flashing to FIRE as it is consumed. A blob annihilates itself and the matter around it
// in a bright flash (peeling outside-in over a few frames), carving a clean cavity its own
// size through anything; it is never created, only consumed, so it always burns itself out.
// MOSS is a creeping overgrowth that coats SOLID SURFACES -- a third growth mode distinct
// from PLANT (needs a waterline) and CRYSTAL (branches into open air): it spreads only along
// the empty cells hugging a WALL/OBSIDIAN/GLASS/WOOD surface, so it greens walls and timber
// like ivy on a ruin without filling open space. It's flammable.
// FUMES is a heavy flammable vapour -- the mirror of GAS: where gas rises to the ceiling,
// fumes SINK through air and POOL in low ground and caverns (same density tier as SNOW:
// heavier than air, floats on every liquid) while spreading out flat like a gas. Then a
// SPARK or flame flashes the whole pocket -- so a pit that fills with fumes is a bomb.
// WIRE / EHEAD / ETAIL are a Wireworld cellular automaton bolted onto the sim: copper WIRE
// carries electrons (an EHEAD "head" chased by an ETAIL "tail") by the classic rules, so you
// can build working logic -- diodes, AND/OR/XOR gates, clocks -- the digital counterpart to
// the analog SPARK. They are inert solids; only the CA rule moves the electrons along.
// IGNITER is the bridge from those circuits to the physical world: an inert solid that does
// nothing until a Wireworld electron (EHEAD) reaches it, then spits FIRE into the empty cells
// around it. Wire a clock or a gate to one and you have a logic-controlled detonator.
// SENSOR is the opposite bridge -- physical to digital: an inert solid that fires an electron
// (EHEAD) into the WIRE next to it whenever any real material (a liquid, powder, fire, ...)
// touches it, so a circuit can READ the world. Sensor + wire logic + igniter = a machine that
// senses, decides and acts (a flood alarm, a heat trigger, a feedback loop).
// LIFE is Conway's Game of Life -- a second famous cellular automaton sharing the sandbox with
// Wireworld. A live cell with 2 or 3 live neighbours survives, an empty cell with exactly 3 is
// born; everything else dies/stays empty. Gliders and oscillators weave through the falling
// sand (which blocks births where it lands), a CA colliding with a physics sim.
// GEYSER is a geothermal vent -- the first RHYTHMIC source. Where SPRING/VOLCANO well their
// material constantly, a geyser ERUPTS on a cycle: for a window of every period it puffs STEAM
// into the empty cells around it (a rising plume the water cycle later rains back), then falls
// dormant. So the world gets periodic gushes instead of a steady trickle.
// LYE is a caustic powder (it falls and piles like SAND) -- the chemical counter to ACID. Where
// the two meet they NEUTRALISE each other: the acid is spent to harmless WATER and the lye to
// SALT (acid + base -> salt + water), so a sprinkle of lye stops an acid spill dead. The first
// rule where two reactive materials cancel each other into products.
enum Material : uint8_t {
    EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7,
    STEAM = 8, WOOD = 9, PLANT = 10, ACID = 11, SMOKE = 12, GLASS = 13, ICE = 14,
    SPRING = 15, TNT = 16, ASH = 17, VOLCANO = 18, VOID = 19, MUD = 20, VIRUS = 21,
    SPARK = 22, OBSIDIAN = 23, SALT = 24, SNOW = 25, MERCURY = 26, GUNPOWDER = 27,
    THERMITE = 28, FROST = 29, WISP = 30, COAL = 31, EMBER = 32, CLONER = 33, CRYSTAL = 34,
    ANTIMATTER = 35, MOSS = 36, FUMES = 37, WIRE = 38, EHEAD = 39, ETAIL = 40, IGNITER = 41,
    SENSOR = 42, LIFE = 43, GEYSER = 44, LYE = 45, SODIUM = 46, CORAL = 47, PHOSPHORUS = 48,
    CEMENT = 49, CHLORINE = 50, BATTERY = 51, FUSE = 52, BURNFUSE = 53, CRYO = 54,
    LAMP = 55, LAMPLIT = 56, PETRIFY = 57,
    MATERIAL_COUNT = 58
};

// Fire burn-out: a per-cell, time-varying transform that is a PURE function of
// (x, y, frame) -- no neighbour reads -- so it stays order-independent and is
// bit-identical on CPU SIMD and the GPU compute backends (which compute the same
// hash). ~FIRE_DECAY/256 of the flame winks out each frame.
static constexpr uint32_t FIRE_DECAY = 12;     // of 256 -> avg flame life ~21 frames
static constexpr uint32_t SMOKE_FROM_FIRE = 4; // burn-outs with hash < this leave SMOKE
static constexpr uint32_t ASH_FROM_FIRE = 6;   // burn-outs with hash in [SMOKE,this) leave ASH (soot)
static constexpr uint32_t SMOKE_FADE = 16;     // of 256 -> smoke wisps last ~16 frames
static constexpr uint32_t STEAM_CONDENSE = 5;  // of 256 -> steam lasts ~50 frames, rises first

inline uint32_t fireHash(int x, int y, uint32_t frame) {
    return ((uint32_t)x * 167u + (uint32_t)y * 101u + frame * 131u) & 0xFFu;
}
inline bool fireBurnsOut(int x, int y, uint32_t frame) { return fireHash(x, y, frame) < FIRE_DECAY; }
inline bool smokeFades(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 73u + (uint32_t)y * 179u + frame * 149u) & 0xFFu;
    return h < SMOKE_FADE;
}
inline bool steamCondenses(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 193u + (uint32_t)y * 97u + frame * 111u) & 0xFFu;
    return h < STEAM_CONDENSE;
}
static constexpr uint32_t WOOD_IGNITE = 28;    // of 256/frame -> wood smoulders, slower than oil
inline bool woodCatches(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 149u + (uint32_t)y * 83u + frame * 157u) & 0xFFu;
    return h < WOOD_IGNITE;
}
static constexpr uint32_t PLANT_GROW = 14;     // of 256/frame -> vines creep, not explode
inline bool plantGrows(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 113u + (uint32_t)y * 191u + frame * 71u) & 0xFFu;
    return h < PLANT_GROW;
}
static constexpr uint32_t ACID_EVAP = 3;       // of 256/frame -> acid slowly used up
static constexpr uint32_t ACID_EAT  = 22;      // of 256/frame -> dissolves solids it touches
inline bool acidEvaporates(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 211u + (uint32_t)y * 137u + frame * 59u) & 0xFFu;
    return h < ACID_EVAP;
}
inline bool acidEats(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 53u + (uint32_t)y * 199u + frame * 89u) & 0xFFu;
    return h < ACID_EAT;
}
inline bool acidDissolves(uint8_t m) {         // which solids acid corrodes (not air/fluids)
    return m == WALL || m == SAND || m == WOOD || m == PLANT;
}
static constexpr uint32_t ICE_MELT = 18;       // of 256/frame -> ice near heat thaws in ~14 frames
inline bool iceMelts(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 127u + (uint32_t)y * 163u + frame * 41u) & 0xFFu;
    return h < ICE_MELT;
}
static constexpr uint32_t ICE_FREEZE = 4;      // of 256/frame -> water touching ice creeps to ice (slow front)
inline bool iceFreezes(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 181u + (uint32_t)y * 67u + frame * 103u) & 0xFFu;
    return h < ICE_FREEZE;
}
static constexpr uint32_t SPRING_FLOW = 20;    // of 256/frame -> empty cells by a spring well up with water
inline bool springFlows(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 89u + (uint32_t)y * 223u + frame * 47u) & 0xFFu;
    return h < SPRING_FLOW;
}
static constexpr uint32_t VOLCANO_FLOW = 16;   // of 256/frame -> empty cells by a vent ooze lava
inline bool volcanoFlows(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 109u + (uint32_t)y * 241u + frame * 67u) & 0xFFu;
    return h < VOLCANO_FLOW;
}
static constexpr uint32_t MUD_FORM = 10;       // of 256/frame -> sand touching water packs to mud
inline bool mudForms(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 157u + (uint32_t)y * 97u + frame * 61u) & 0xFFu;
    return h < MUD_FORM;
}
static constexpr uint32_t MUD_BAKE = 14;       // of 256/frame -> mud by fire/lava bakes back to sand
inline bool mudBakes(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 83u + (uint32_t)y * 173u + frame * 109u) & 0xFFu;
    return h < MUD_BAKE;
}
static constexpr uint32_t VIRUS_SPREAD = 36;   // of 256/frame -> infection eats a neighbour
inline bool virusSpreads(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 191u + (uint32_t)y * 131u + frame * 73u) & 0xFFu;
    return h < VIRUS_SPREAD;
}
static constexpr uint32_t VIRUS_DECAY = 16;    // of 256/frame -> virus burns out (~16 frame life)
inline bool virusDecays(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 107u + (uint32_t)y * 199u + frame * 149u) & 0xFFu;
    return h < VIRUS_DECAY;
}
// What the infection can convert: anything but the void, the walls that contain it,
// the fire/lava that cauterise it, and itself.
inline bool virusEats(uint8_t m) {
    return m != EMPTY && m != WALL && m != VIRUS && m != FIRE && m != LAVA && m != VOID;
}

// Per-cell time-varying transforms over the live interior (scalar; identical on
// every SIMD width and the GPU). FIRE burns out to EMPTY; STEAM condenses back to
// WATER. Gated by the world on "is anything reactive present?" so plain worlds
// (e.g. the benchmark) pay nothing.
inline void decayFire(uint8_t* grid, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i];
            if (c == FIRE) {
                uint32_t h = fireHash(x, y, frame);
                if (h < FIRE_DECAY)                                       // flame -> smoke / ash / empty
                    grid[i] = (h < SMOKE_FROM_FIRE) ? SMOKE : (h < ASH_FROM_FIRE) ? ASH : EMPTY;
            }
            else if (c == SMOKE && smokeFades(x, y, frame)) grid[i] = EMPTY;
            else if (c == STEAM && steamCondenses(x, y, frame)) grid[i] = WATER;
            else if (c == ACID && acidEvaporates(x, y, frame)) grid[i] = EMPTY;
        }
}

// Ignition: OIL touching FIRE (4-neighbour) catches and becomes FIRE, so flame
// spreads through fuel one layer per frame. Done in two per-cell passes through a
// scratch buffer (the `moved` flags, free after the movement step): pass 1 reads
// a consistent snapshot of the grid and records intent; pass 2 applies it. Each
// pass reads one buffer and writes another, so it is order-independent and the
// GPU reproduces it bit-for-bit.
inline bool isHot(uint8_t m) { return m == FIRE || m == LAVA; }   // ignites adjacent fuel

// OIL and WOOD catch fire from an adjacent FIRE/LAVA. OIL ignites instantly; WOOD
// smoulders (a per-cell frame hash gates it, so it burns slower). Two-pass
// snapshot via the scratch buffer keeps it order-independent / GPU-identical.
inline void igniteFire(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i];
            bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
            bool ign = ((c == OIL || c == PLANT || c == GAS || c == WISP || c == MOSS || c == FUMES) && hot)   // oil, plant, gas, wisp, moss & fumes: instant
                    || (c == WOOD && hot && woodCatches(x, y, frame));                                       // wood: slow
            scratch[i] = ign ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = FIRE;
        }
}

// Things meet hot: at an interface with FIRE or LAVA, WATER flashes to STEAM and
// ACID boils off to SMOKE, while the FIRE is quenched to EMPTY and the LAVA freezes
// to WALL (stone) wherever WATER touches it. One two-pass snapshot (the `moved`
// scratch): pass 1 marks every reacting cell at such an interface, pass 2 transforms
// each by its own type -- order-independent, GPU-identical.
inline void quench(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    auto nb = [&](size_t i, uint8_t m) {
        return grid[i-1]==m || grid[i+1]==m || grid[i-SW]==m || grid[i+SW]==m;
    };
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i];
            bool react = false;
            if (c == WATER)      react = nb(i, FIRE) || nb(i, LAVA);
            else if (c == ACID)  react = nb(i, FIRE) || nb(i, LAVA);   // acid boils off
            else if (c == FIRE)  react = nb(i, WATER);
            else if (c == LAVA)  react = nb(i, WATER);
            scratch[i] = react ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (!scratch[i]) continue;
            uint8_t c = grid[i];                                       // water->steam, acid->smoke,
            grid[i] = (c == WATER) ? STEAM : (c == ACID) ? SMOKE       // fire->empty, lava->obsidian
                    : (c == FIRE) ? EMPTY : OBSIDIAN;
        }
}

// Plant growth: an EMPTY cell with both a PLANT neighbour and a WATER neighbour
// sprouts PLANT (frame-hashed so vines creep). Each empty cell decides from a
// snapshot and writes only itself, so it's order-independent and GPU-identical.
// Self-limiting: once plant lines a waterline, the remaining empty cells no
// longer touch water.
inline void growPlant(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    auto nb = [&](size_t i, uint8_t m) {
        return grid[i-1]==m || grid[i+1]==m || grid[i-SW]==m || grid[i+SW]==m;
    };
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            scratch[i] = (grid[i] == EMPTY && nb(i, PLANT) && nb(i, WATER) && plantGrows(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = PLANT;
        }
}

// Acid corrosion: a dissolvable SOLID touching ACID is eaten away to EMPTY
// (frame-hashed, so it bores through gradually). Two-pass snapshot via the
// scratch buffer -> order-independent and GPU-identical. The interior-only sweep
// never touches the padding border, so the WALL frame can't be eaten through.
inline void dissolveAcid(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    auto nb = [&](size_t i, uint8_t m) {
        return grid[i-1]==m || grid[i+1]==m || grid[i-SW]==m || grid[i+SW]==m;
    };
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            scratch[i] = (acidDissolves(grid[i]) && nb(i, ACID) && acidEats(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = EMPTY;
        }
}

// Glassmaking: SAND touching LAVA melts to GLASS (an inert solid). One two-pass
// snapshot through the scratch buffer -> order-independent, GPU-identical.
inline void makeGlass(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            scratch[i] = (grid[i] == SAND &&
                          (grid[i-1]==LAVA || grid[i+1]==LAVA || grid[i-SW]==LAVA || grid[i+SW]==LAVA)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = GLASS;
        }
}

// Melting: ICE (and SNOW, the light powder) touching FIRE or LAVA thaws back to
// WATER (frame-hashed, so it melts over several frames). Two-pass snapshot through
// the scratch buffer -> order-independent, GPU-identical. The inverse of glassmaking,
// and it feeds the existing water rules: ice dropped on lava melts, and that water
// then quenches the lava to obsidian.
inline void meltIce(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
            scratch[i] = ((grid[i] == ICE || grid[i] == SNOW) && hot && iceMelts(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = WATER;
        }
}

// Freezing: WATER touching ICE slowly turns to ICE (frame-hashed, so a cold front
// creeps through a still pool). The counterpart to meltIce -- ice spreads through
// water but retreats wherever fire/lava thaws it, so heat and cold reach a little
// equilibrium. Two-pass snapshot through the scratch buffer -> order-independent,
// GPU-identical.
inline void freezeWater(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool ice = grid[i-1]==ICE || grid[i+1]==ICE || grid[i-SW]==ICE || grid[i+SW]==ICE;
            scratch[i] = (grid[i] == WATER && ice && iceFreezes(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = ICE;
        }
}

// Spring (source): an EMPTY cell touching a SPRING wells up with WATER (frame-
// hashed, so it bubbles out at a steady rate). The empty cell decides from a
// snapshot and writes only itself -- like plant growth -- so it stays order-
// independent and GPU-identical. The SPRING itself is a permanent solid: it never
// moves and never depletes, so it is an endless fountain. This is the engine's
// first rule that creates material from nothing (so it does not conserve mass --
// only reachable when a spring is actually placed, never in the benchmark seed).
inline void emitSpring(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool src = grid[i-1]==SPRING || grid[i+1]==SPRING || grid[i-SW]==SPRING || grid[i+SW]==SPRING;
            scratch[i] = (grid[i] == EMPTY && src && springFlows(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = WATER;
        }
}

// Volcano (lava source): an EMPTY cell touching a VOLCANO wells up with LAVA --
// the hot mirror of emitSpring (same order-independent mark/apply snapshot). The
// VOLCANO never moves or depletes, so it is a perpetual lava vent, and the lava it
// oozes then drives every heat reaction: ignition, glassmaking, steam, melt, blast.
inline void emitVolcano(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool src = grid[i-1]==VOLCANO || grid[i+1]==VOLCANO || grid[i-SW]==VOLCANO || grid[i+SW]==VOLCANO;
            scratch[i] = (grid[i] == EMPTY && src && volcanoFlows(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = LAVA;
        }
}

// Void (sink): a black hole that swallows whatever it touches. Any cell next to a
// VOID -- except WALL (which contains it) and another VOID -- is consumed to EMPTY.
// The VOID never moves or depletes, so with gravity feeding it, material flows in
// and vanishes: a drain. Two-pass snapshot (mark consumed cells, then clear them),
// so it's order-independent and GPU-identical -- the sink twin of the spring/volcano
// sources.
inline void consumeVoid(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i];
            bool nearVoid = grid[i-1]==VOID || grid[i+1]==VOID || grid[i-SW]==VOID || grid[i+SW]==VOID;
            scratch[i] = (c != EMPTY && c != WALL && c != VOID && nearVoid) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = EMPTY;
        }
}

// Mud: a little wet/dry cycle done as ONE two-pass snapshot. SAND touching WATER
// packs into MUD; MUD touching FIRE/LAVA bakes back to SAND. Pass 1 marks each cell
// with which way it goes (1 = wet to mud, 2 = bake to sand), pass 2 applies -- so
// it's order-independent and GPU-identical. Frame-hashed, so shores muddy and bake
// gradually rather than all at once.
inline void mudCycle(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == SAND) {
                bool water = grid[i-1]==WATER || grid[i+1]==WATER || grid[i-SW]==WATER || grid[i+SW]==WATER;
                if (water && mudForms(x, y, frame)) r = 1;
            } else if (c == MUD) {
                bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
                if (hot && mudBakes(x, y, frame)) r = 2;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = MUD;
            else if (scratch[i] == 2) grid[i] = SAND;
        }
}

// Virus: a self-propagating infection, one combined two-pass snapshot. A VIRUS cell
// dies to EMPTY if it burns out (frame hash) or touches FIRE/LAVA (cauterised); any
// other consumable cell next to a VIRUS turns into VIRUS. Because new infection
// outpaces decay, it spreads as an expanding wave that leaves emptiness behind and
// stops at WALL. Pass 1 marks each cell (1 = infect, 2 = die), pass 2 applies, so
// it's order-independent and GPU-identical.
inline void spreadVirus(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == VIRUS) {
                bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
                if (hot || virusDecays(x, y, frame)) r = 2;
            } else if (virusEats(c)) {
                bool nv = grid[i-1]==VIRUS || grid[i+1]==VIRUS || grid[i-SW]==VIRUS || grid[i+SW]==VIRUS;
                if (nv && virusSpreads(x, y, frame)) r = 1;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = VIRUS;
            else if (scratch[i] == 2) grid[i] = EMPTY;
        }
}

// Spark: electricity arcing through water. Every WATER cell next to a SPARK flashes
// to SPARK; every SPARK then boils off to STEAM (or fizzles to EMPTY if it has no
// water around it), so a bright pulse sweeps a connected pool once -- converting it
// to a rising cloud of steam -- and dies when the water runs out. The GAS and OIL it
// touches are ignited to FIRE. One combined two-pass snapshot (mark 1=conduct,
// 2=boil to steam, 3=fizzle, 4=ignite; then apply), order-independent and
// GPU-identical. Boiling (not returning to water) is what makes it terminate instead
// of oscillating forever.
inline void arcSpark(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            bool nWater = grid[i-1]==WATER || grid[i+1]==WATER || grid[i-SW]==WATER || grid[i+SW]==WATER;
            bool nSpark = grid[i-1]==SPARK || grid[i+1]==SPARK || grid[i-SW]==SPARK || grid[i+SW]==SPARK;
            if (c == SPARK)      r = nWater ? 2 : 3;
            else if (c == WATER) { if (nSpark) r = 1; }
            else if (c == GAS || c == OIL || c == WISP || c == FUMES) { if (nSpark) r = 4; }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t v = scratch[i];
            if (v == 1) grid[i] = SPARK;
            else if (v == 2) grid[i] = STEAM;
            else if (v == 3) grid[i] = EMPTY;
            else if (v == 4) grid[i] = FIRE;
        }
}

// Frost: a creeping cold front -- the cold mirror of FIRE/VIRUS. Painted as a seed it
// races across WATER, freezing every water cell it touches into more FROST one step
// ahead, so a freezing wave radiates out at one cell per frame and crackles a whole
// connected pond over with ice. Each FROST cell crystallises to ICE the next frame
// (it's only ever the advancing leading edge), so the front leaves a solid ice trail
// and self-terminates where the water runs out -- monotonic (water only ever turns to
// frost then ice), so it always settles. FROST also WITHERS the PLANT it touches (a
// killing frost) and is itself MELTED back to WATER by FIRE/LAVA -- so heat stops a
// frost front the way water stops a fire (and the meltwater then flashes to steam on
// the lava in the earlier quench pass, so the boundary resolves rather than oscillating).
// One combined two-pass snapshot (mark 1=water->frost advance, 3=frost->ice crystallise,
// 4=plant->empty wither, 5=frost->water melt; then apply), order-independent / GPU-identical.
inline void spreadFrost(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            bool nFrost = grid[i-1]==FROST || grid[i+1]==FROST || grid[i-SW]==FROST || grid[i+SW]==FROST;
            if (c == FROST) {
                bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
                r = hot ? 5 : 3;
            } else if (c == WATER) {
                if (nFrost) r = 1;
            } else if (c == PLANT) {
                if (nFrost) r = 4;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t v = scratch[i];
            if (v == 1) grid[i] = FROST;
            else if (v == 3) grid[i] = ICE;
            else if (v == 4) grid[i] = EMPTY;
            else if (v == 5) grid[i] = WATER;
        }
}

static constexpr uint32_t COAL_FLAME = 64;     // of 256/frame -> an EMBER sets an adjacent empty cell alight
inline bool coalFlames(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 97u + (uint32_t)y * 149u + frame * 73u) & 0xFFu;
    return h < COAL_FLAME;
}
static constexpr uint32_t COAL_ASH = 8;        // of 256/frame -> a burning EMBER ages down to ASH (avg ~32-frame burn)
inline bool coalAshes(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 181u + (uint32_t)y * 61u + frame * 139u) & 0xFFu;
    return h < COAL_ASH;
}

// Coal/ember: a pourable fuel. COAL is inert until FIRE/LAVA (or an existing EMBER)
// touches it, when it catches into EMBER -- glowing burning coal. An EMBER spreads to
// the COAL it touches (so the fire creeps through a pile ember-to-coal, independent of
// which way the loose flames drift), RADIATES FIRE into the empty cells around it
// (steadily, frame-hashed -- this is what ignites nearby oil/gas/wood) and slowly ages
// down to ASH, so the burn is long but finite. One combined two-pass snapshot (mark
// 1=coal catches -> ember, 2=this cell is an ember; then apply: 1->EMBER, 2->ASH-or-stay,
// and EMPTY next to an ember -> FIRE), order-independent and GPU-identical.
inline void smoulderCoal(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == COAL) {
                bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
                bool ember = grid[i-1]==EMBER || grid[i+1]==EMBER || grid[i-SW]==EMBER || grid[i+SW]==EMBER;
                if (hot || ember) r = 1;
            } else if (c == EMBER) {
                r = 2;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t v = scratch[i];
            if (v == 1) grid[i] = EMBER;
            else if (v == 2) { if (coalAshes(x, y, frame)) grid[i] = ASH; }   // else stays EMBER
            else if (grid[i] == EMPTY) {
                bool litN = scratch[i-1]==2 || scratch[i+1]==2 || scratch[i-SW]==2 || scratch[i+SW]==2;
                if (litN && coalFlames(x, y, frame)) grid[i] = FIRE;
            }
        }
}

// Cloner: an inert duplicator. It copies the material directly ABOVE it into the empty
// cell directly BELOW it, so a single drop of anything on top becomes an endless faucet
// of that material (the top cell is only read, never consumed). Two-pass snapshot, but
// the scratch carries a MATERIAL ID rather than a flag: pass 1 stores, for each CLONER,
// the cloneable material sitting above it (skip EMPTY/WALL/CLONER, so it can't extrude
// structure or replicate itself into grey goo); pass 2 fills any EMPTY cell whose upper
// neighbour is such a loaded cloner with that stored material. Pass 2 reads only the
// pass-1 scratch of its neighbour plus its own grid cell -- never a neighbour's live grid
// cell -- so it stays order-independent and GPU-identical (no read/write race).
inline void cloneMaterial(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t r = 0;
            if (grid[i] == CLONER) {
                uint8_t above = grid[i-SW];
                if (above != EMPTY && above != WALL && above != CLONER) r = above;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (grid[i] == EMPTY && scratch[i-SW] != 0) grid[i] = scratch[i-SW];
        }
}

static constexpr uint32_t CRYSTAL_GROW = 10;   // of 256/frame -> dendrites creep slowly
inline bool crystalGrows(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 127u + (uint32_t)y * 89u + frame * 167u) & 0xFFu;
    return h < CRYSTAL_GROW;
}

// Crystal: an inert mineral that GROWS dendritically into empty space -- the first rule
// to grow into bare air rather than along a waterline (plant) or by consuming a host
// (frost/virus). An EMPTY cell crystallises only when EXACTLY ONE of its EIGHT neighbours
// is already crystal (frame-hashed for a slow creep): a tip extends where a single arm
// reaches, but a cell flanked by two arms (>=2 crystal neighbours, orthogonal or diagonal)
// locks and won't fill -- the classic dendrite rule, so the growth stays a delicate
// branching gem instead of a solid flood. Two snapshot passes (mark the eligible empties,
// then apply), order-independent and GPU-identical -- like plant growth, but counting
// neighbours instead of needing water.
inline void growCrystal(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t r = 0;
            if (grid[i] == EMPTY) {
                int n = (grid[i-1]==CRYSTAL) + (grid[i+1]==CRYSTAL) + (grid[i-SW]==CRYSTAL) + (grid[i+SW]==CRYSTAL)
                      + (grid[i-SW-1]==CRYSTAL) + (grid[i-SW+1]==CRYSTAL) + (grid[i+SW-1]==CRYSTAL) + (grid[i+SW+1]==CRYSTAL);
                if (n == 1 && crystalGrows(x, y, frame)) r = 1;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = CRYSTAL;
        }
}

// Is this cell "matter" that antimatter annihilates? Anything that isn't empty space or
// more antimatter -- so antimatter eats WALL, LAVA, WATER and every other material alike.
inline bool isMatter(uint8_t m) { return m != EMPTY && m != ANTIMATTER; }

// Antimatter: disintegrates any MATTER it touches. Inert in a vacuum, but the instant it
// meets anything that isn't empty it annihilates -- a self-consuming disintegrator that
// eats a growing cavity through ANYTHING (even WALL/LAVA/WATER, which stop fire and
// explosions), flashing to FIRE as the energy is released and leaving a clean hole behind.
// Two-pass snapshot like detonateTnt: pass 1 marks every ANTIMATTER cell that touches
// matter; pass 2 turns each marked cell to FIRE (consuming the antimatter) and clears
// every matter cell next to one to EMPTY. Pass 2 reads the pass-1 marks plus its own grid
// cell, writing only itself, so it is order-independent and GPU-identical. Antimatter is
// never created, only consumed, so its count strictly decreases and it always terminates --
// a solid blob peels to fire from the outside in over a few frames (the exposed inner
// layers annihilate against the fire their own surface just made), leaving a clean cavity.
inline void annihilate(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool m = isMatter(grid[i-1]) || isMatter(grid[i+1]) || isMatter(grid[i-SW]) || isMatter(grid[i+SW]);
            scratch[i] = (grid[i] == ANTIMATTER && m) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) { grid[i] = FIRE; continue; }
            bool nearAnnih = scratch[i-1] || scratch[i+1] || scratch[i-SW] || scratch[i+SW];
            if (nearAnnih && isMatter(grid[i])) grid[i] = EMPTY;
        }
}

static constexpr uint32_t MOSS_GROW = 7;       // of 256/frame -> moss creeps slowly over stone
inline bool mossGrows(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 139u + (uint32_t)y * 113u + frame * 61u) & 0xFFu;
    return h < MOSS_GROW;
}
// What moss can cling to: the inert stone/timber surfaces (NOT moss itself, or it would
// fill open space instead of hugging a surface).
inline bool mossAnchor(uint8_t m) { return m == WALL || m == OBSIDIAN || m == GLASS || m == WOOD; }

// Moss: a creeping overgrowth that coats solid surfaces -- the third growth mode (PLANT
// needs a waterline, CRYSTAL branches into open air, MOSS hugs a surface). An EMPTY cell
// turns to MOSS when it is adjacent to existing MOSS *and* adjacent to a stone/wood anchor
// (frame-hashed for a slow creep), so moss spreads only along the thin layer of empty cells
// against a wall or timber -- greening structures without filling open space. A frame-hashed
// mark/apply pair (each empty cell decides from a snapshot), order-independent / GPU-identical.
inline void growMoss(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t r = 0;
            if (grid[i] == EMPTY) {
                bool nMoss = grid[i-1]==MOSS || grid[i+1]==MOSS || grid[i-SW]==MOSS || grid[i+SW]==MOSS;
                bool nAnchor = mossAnchor(grid[i-1]) || mossAnchor(grid[i+1]) || mossAnchor(grid[i-SW]) || mossAnchor(grid[i+SW]);
                if (nMoss && nAnchor && mossGrows(x, y, frame)) r = 1;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = MOSS;
        }
}

// Wireworld: a tiny synchronous cellular automaton layered on the sim, so copper WIRE can
// carry electrons and you can build working logic -- diodes, gates, clocks. An electron is
// a HEAD (EHEAD) chased by a TAIL (ETAIL). The classic rules: HEAD -> TAIL, TAIL -> WIRE, and
// WIRE -> HEAD exactly when 1 or 2 of its 8 neighbours are heads (so a pulse races forward,
// the tail stops it doubling back, and a junction that sees 3 heads stays quiet -- the basis
// of gates). Being a synchronous CA it uses the two-pass snapshot with the scratch buffer
// carrying the NEXT material id (like the cloner): pass 1 computes every cell's next state
// from the grid snapshot, pass 2 applies it. scratch == 0 means "not a wire cell, leave it"
// (wire cells never become EMPTY, so 0 is a safe sentinel). Order-independent / GPU-identical.
inline void wireWorld(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == EHEAD) r = ETAIL;
            else if (c == ETAIL) r = WIRE;
            else if (c == WIRE) {
                int h = (grid[i-1]==EHEAD) + (grid[i+1]==EHEAD) + (grid[i-SW]==EHEAD) + (grid[i+SW]==EHEAD)
                      + (grid[i-SW-1]==EHEAD) + (grid[i-SW+1]==EHEAD) + (grid[i+SW-1]==EHEAD) + (grid[i+SW+1]==EHEAD);
                r = (h == 1 || h == 2) ? EHEAD : WIRE;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = scratch[i];
        }
}

// Igniter: the bridge from Wireworld to the physical world. An inert solid that does nothing
// until an electron head (EHEAD) reaches a cell next to it, at which point it spits FIRE into
// the empty cells around it -- a logic-controlled spark plug, so a clock or a gate can fire a
// timed/triggered detonator into gunpowder, fumes, oil or TNT. It runs right after the
// Wireworld pass, so it sees the head exactly as the CA advances it into the adjacent wire.
// Two-pass snapshot (mark each IGNITER next to a head, then turn the EMPTY cells next to a
// marked igniter into FIRE), order-independent and GPU-identical -- a one-shot volcano of fire.
inline void fireIgniter(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool e = grid[i-1]==EHEAD || grid[i+1]==EHEAD || grid[i-SW]==EHEAD || grid[i+SW]==EHEAD;
            scratch[i] = (grid[i] == IGNITER && e) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (grid[i] == EMPTY) {
                bool nIgn = scratch[i-1] || scratch[i+1] || scratch[i-SW] || scratch[i+SW];
                if (nIgn) grid[i] = FIRE;
            }
        }
}

// What a SENSOR detects: any "real" material that flows up against it -- i.e. anything that
// isn't empty space, structural WALL, or part of the circuitry itself.
inline bool detectable(uint8_t m) {
    return m != EMPTY && m != WALL && m != WIRE && m != EHEAD && m != ETAIL && m != SENSOR && m != IGNITER;
}

// Sensor: the physical->digital bridge. An inert solid that, whenever a detectable material is
// next to it, lights the WIRE next to it into an electron head (EHEAD) -- so the Wireworld
// circuit can sense the world (water rising, lava arriving, a powder piling up) and react.
// Two-pass snapshot: pass 1 marks every SENSOR with a detectable neighbour, pass 2 turns a
// WIRE next to a marked sensor into EHEAD. The injected head is propagated by the wireworld
// pass on the following frame; while the trigger persists the sensor re-fires periodically
// (the wire cycles head->tail->wire before it can be relit), so a steady touch is a clock.
inline void senseWorld(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool d = detectable(grid[i-1]) || detectable(grid[i+1]) || detectable(grid[i-SW]) || detectable(grid[i+SW]);
            scratch[i] = (grid[i] == SENSOR && d) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (grid[i] == WIRE) {
                bool nSensor = scratch[i-1] || scratch[i+1] || scratch[i-SW] || scratch[i+SW];
                if (nSensor) grid[i] = EHEAD;
            }
        }
}

// Conway's Game of Life: a second cellular automaton in the sandbox. A LIFE cell with 2 or 3
// live neighbours (of 8) survives, else dies to EMPTY; an EMPTY cell with exactly 3 live
// neighbours is born. Synchronous CA, so the two-pass snapshot computes it: pass 1 marks each
// LIFE-or-EMPTY cell's fate (1 = live next, 2 = empty next, 0 = leave it -- which keeps any
// other material untouched, so it blocks births and gliders thread through the falling world),
// pass 2 applies. Order-independent / GPU-identical.
inline void conwayLife(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == LIFE || c == EMPTY) {
                int n = (grid[i-1]==LIFE) + (grid[i+1]==LIFE) + (grid[i-SW]==LIFE) + (grid[i+SW]==LIFE)
                      + (grid[i-SW-1]==LIFE) + (grid[i-SW+1]==LIFE) + (grid[i+SW-1]==LIFE) + (grid[i+SW+1]==LIFE);
                if (c == LIFE) r = (n == 2 || n == 3) ? 1 : 2;
                else           r = (n == 3) ? 1 : 0;   // empty: born only on exactly 3 live neighbours
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = LIFE;
            else if (scratch[i] == 2) grid[i] = EMPTY;
        }
}

static constexpr uint32_t GEYSER_PERIOD = 150;  // frames per eruption cycle
static constexpr uint32_t GEYSER_DUTY   = 30;   // of those, frames it is erupting (then dormant)
static constexpr uint32_t GEYSER_SPRAY  = 90;   // of 256/frame -> density of the steam puff while erupting
inline bool geyserErupts(uint32_t frame) { return (frame % GEYSER_PERIOD) < GEYSER_DUTY; }
inline bool geyserSprays(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 151u + (uint32_t)y * 47u + frame * 199u) & 0xFFu;
    return h < GEYSER_SPRAY;
}

// Geyser: a periodic geothermal vent. The whole function is gated on the global eruption
// window (so it does nothing -- touches nothing -- while dormant, keeping it bit-identical to
// the GPU, which writes a blank scratch and applies nothing). While erupting it puffs STEAM
// into the empty cells around it on a frame-hash for texture; the steam rises and condenses
// back to water through the existing cycle, so a geyser gushes and rains on a rhythm. Same
// order-independent mark/apply snapshot as emitSpring.
inline void eruptGeyser(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    if (!geyserErupts(frame)) return;   // dormant: leave the grid (and scratch) untouched
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool src = grid[i-1]==GEYSER || grid[i+1]==GEYSER || grid[i-SW]==GEYSER || grid[i+SW]==GEYSER;
            scratch[i] = (grid[i] == EMPTY && src && geyserSprays(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = STEAM;
        }
}

// Lye: the chemical counter to acid. Where LYE and ACID touch they neutralise -- acid + base
// -> salt + water -- so the ACID is spent to WATER and the LYE to SALT, both existing materials.
// Two-pass snapshot: pass 1 marks each LYE that touches acid (-> 1, becomes SALT) and each ACID
// that touches lye (-> 2, becomes WATER); pass 2 applies. Order-independent / GPU-identical.
inline void neutraliseLye(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == LYE) {
                if (grid[i-1]==ACID || grid[i+1]==ACID || grid[i-SW]==ACID || grid[i+SW]==ACID) r = 1;
            } else if (c == ACID) {
                if (grid[i-1]==LYE || grid[i+1]==LYE || grid[i-SW]==LYE || grid[i+SW]==LYE) r = 2;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = SALT;
            else if (scratch[i] == 2) grid[i] = WATER;
        }
}

// Sodium: a soft alkali-metal powder (it falls and piles like SAND) -- the one explosive
// that WATER sets off instead of quenching. Where SODIUM touches WATER (or anything hot)
// it flares to FIRE and flashes the touching water to STEAM: the real 2Na + 2H2O -> 2NaOH
// + H2 reaction is sharply exothermic and the liberated hydrogen ignites. Two-pass snapshot
// like detonateTnt: pass 1 marks every SODIUM cell touching water or fire/lava; pass 2 turns
// each marked cell into FIRE and boils each WATER beside a marked cell to STEAM. The FIRE it
// makes is itself hot, so a pile chain-reacts outward one ring per frame. The chemistry
// counterpart to the inert SAND it resembles, and a sibling to ACID/LYE/SALT.
inline void reactSodium(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool trig = grid[i-1]==WATER || grid[i+1]==WATER || grid[i-SW]==WATER || grid[i+SW]==WATER
                     || isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
            scratch[i] = (grid[i] == SODIUM && trig) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) { grid[i] = FIRE; continue; }
            if (grid[i] == WATER &&
                (scratch[i-1] || scratch[i+1] || scratch[i-SW] || scratch[i+SW]))
                grid[i] = STEAM;
        }
}

// Coral grows slowly, branching like a crystal -- but UNDERWATER. Where CRYSTAL
// crystallises into bare air and PLANT creeps along the waterline, CORAL spreads
// through WATER itself: a water cell turns to coral when EXACTLY ONE of its eight
// neighbours is coral (a frame-hash gates the rate, so reefs branch dendritically
// instead of flooding solid -- cells flanked by two arms lock, exactly as crystal
// does). It is a living reef, so heat bleaches it: a coral cell touching FIRE/LAVA
// dies to ASH. The first growth bound to a LIQUID substrate -- seed one grain in a
// pool and a branching reef fills it, consuming the water as it goes.
static constexpr uint32_t CORAL_GROW = 8;      // of 256/frame -> reefs branch slowly
inline bool coralGrows(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 151u + (uint32_t)y * 101u + frame * 181u) & 0xFFu;
    return h < CORAL_GROW;
}
inline void growCoral(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t r = 0;
            if (grid[i] == WATER) {
                int n = (grid[i-1]==CORAL) + (grid[i+1]==CORAL) + (grid[i-SW]==CORAL) + (grid[i+SW]==CORAL)
                      + (grid[i-SW-1]==CORAL) + (grid[i-SW+1]==CORAL) + (grid[i+SW-1]==CORAL) + (grid[i+SW+1]==CORAL);
                if (n == 1 && coralGrows(x, y, frame)) r = 1;          // grow into water
            } else if (grid[i] == CORAL) {
                if (isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW])) r = 2;  // bleach to ash
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = CORAL;
            else if (scratch[i] == 2) grid[i] = ASH;
        }
}

// Phosphorus: a waxy powder (it falls and piles like SAND) that is the mirror image
// of SODIUM -- where sodium bursts into flame on contact with WATER, white phosphorus
// ignites on contact with AIR. So you must keep it SUBMERGED: a grain touching an
// empty cell spontaneously catches FIRE (a frame-hash gives it a brief, shimmering
// delay), while a grain walled in by water or solids sits perfectly stable. It also
// catches instantly from any flame, so a pile flares from its exposed surface inward.
// Two-pass snapshot: pass 1 marks each PHOSPHORUS cell that is next to fire/lava (an
// instant catch) or next to air (and the frame-hash fires); pass 2 turns each marked
// cell into FIRE. Store it underwater and drain the pool to set the whole cache alight.
static constexpr uint32_t PHOS_IGNITE = 50;    // of 256/frame -> air-exposed grains light within a few frames
inline bool phosphorusBurns(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 199u + (uint32_t)y * 113u + frame * 173u) & 0xFFu;
    return h < PHOS_IGNITE;
}
inline void ignitePhosphorus(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t r = 0;
            if (grid[i] == PHOSPHORUS) {
                bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
                bool air = grid[i-1]==EMPTY || grid[i+1]==EMPTY || grid[i-SW]==EMPTY || grid[i+SW]==EMPTY;
                if (hot || (air && phosphorusBurns(x, y, frame))) r = 1;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = FIRE;
        }
}

// Cement: a pourable construction powder (it falls and piles like SAND). The first
// BUILDING material -- once a grain comes to rest *on something* (the cell below is
// not empty), it cures: a frame-hash slowly turns it to WALL (stone), so a poured
// pile settles into its mould and then sets, hardening from the supported cells
// upward like drying concrete. A grain still falling through air (nothing beneath it)
// stays loose, so it never freezes in mid-air. The counterpart to the destroyers
// (ACID/THERMITE/ANTIMATTER eat WALL; cement pours new WALL back). Two-pass snapshot:
// pass 1 marks each supported CEMENT whose frame-hash fires; pass 2 turns it to WALL.
static constexpr uint32_t CEMENT_SET = 6;      // of 256/frame -> a resting pile cures over ~1-2 s
inline bool cementSets(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 61u + (uint32_t)y * 157u + frame * 97u) & 0xFFu;
    return h < CEMENT_SET;
}
inline void hardenCement(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            // supported = the cell below is occupied (settled, not falling through air)
            scratch[i] = (grid[i] == CEMENT && grid[i+SW] != EMPTY && cementSets(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = WALL;
        }
}

// Chlorine: a heavy green toxic gas (it sinks and pools like FUMES). Its signature is
// real chemistry -- where CHLORINE meets SODIUM the two combine into SALT (2Na + Cl2 ->
// 2NaCl, the very reaction that makes table salt), closing a loop between two materials
// already in the world. It is also poisonous to living things: a CHLORINE cell touching
// PLANT, MOSS or CORAL bleaches it away (and is spent doing so), and any stray chlorine
// slowly disperses on a frame-hash, so toxic clouds thin out instead of lasting forever.
// Two-pass snapshot: pass 1 marks each cell's fate (1 = becomes SALT, 2 = becomes EMPTY);
// pass 2 applies. Order-independent / GPU-identical.
static constexpr uint32_t CHLORINE_FADE = 4;   // of 256/frame -> a cloud thins over ~1 s
inline bool chlorineFades(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 83u + (uint32_t)y * 211u + frame * 67u) & 0xFFu;
    return h < CHLORINE_FADE;
}
inline bool chlorineKills(uint8_t m) { return m == PLANT || m == MOSS || m == CORAL; }
inline void reactChlorine(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            uint8_t n0 = grid[i-1], n1 = grid[i+1], n2 = grid[i-SW], n3 = grid[i+SW];
            if (c == CHLORINE) {
                if (n0==SODIUM || n1==SODIUM || n2==SODIUM || n3==SODIUM) r = 1;           // -> SALT
                else if (chlorineKills(n0)||chlorineKills(n1)||chlorineKills(n2)||chlorineKills(n3)) r = 2;  // spent bleaching
                else if (chlorineFades(x, y, frame)) r = 2;                                // disperse
            } else if (c == SODIUM) {
                if (n0==CHLORINE || n1==CHLORINE || n2==CHLORINE || n3==CHLORINE) r = 1;   // -> SALT
            } else if (chlorineKills(c)) {
                if (n0==CHLORINE || n1==CHLORINE || n2==CHLORINE || n3==CHLORINE) r = 2;   // bleached away
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = SALT;
            else if (scratch[i] == 2) grid[i] = EMPTY;
        }
}

// Battery: a power source for WIREWORLD circuits and the first material to GENERATE
// signals on its own. On a periodic global clock it injects an electron head (EHEAD)
// into every WIRE cell touching it, so a battery laid against a wire network drives a
// steady train of pulses -- the missing autonomous SOURCE that lets circuits run
// hands-free (a clock for a timed IGNITER, a ring oscillator, a blinking display).
// Where a SENSOR needs the world to poke it, a battery needs nothing. The pulse window
// uses the global frame counter (same determinism as fire-decay/geyser), so it stays
// bit-identical across backends. Two-pass snapshot: pass 1 marks each WIRE touching a
// battery on a pulse frame; pass 2 lights it to EHEAD. Runs after the wireworld pass,
// so an injected head begins propagating on the next frame.
static constexpr uint32_t BATTERY_PERIOD = 12;   // a one-frame pulse every 12 frames
inline void emitBattery(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    bool pulse = (frame % BATTERY_PERIOD) == 0u;
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            scratch[i] = (pulse && grid[i] == WIRE &&
                          (grid[i-1]==BATTERY || grid[i+1]==BATTERY || grid[i-SW]==BATTERY || grid[i+SW]==BATTERY)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = EHEAD;
        }
}

// Fuse: a detonator cord. A length of FUSE is inert until lit -- then it burns along
// itself at a crisp one cell per frame, so you can route and TIME an explosion: lay a
// long winding fuse from a safe corner to a cache of TNT and light the far end. The
// burning tip is a second material, BURNFUSE, which lives a single frame before turning
// to FIRE, so the burn advances exactly one cell per frame and leaves a short fading
// trail of flame that detonates or ignites whatever the cord runs into -- the existing
// fire mechanics do the rest, no special-casing. FUSE catches from an adjacent burning
// tip or from FIRE/LAVA/EMBER, so a torch, a fleck of lava or an IGNITER's flame all
// light it. Two-pass snapshot: pass 1 marks each FUSE that catches (-> BURNFUSE) and
// each BURNFUSE that burns out (-> FIRE); pass 2 applies. Order-independent / GPU-identical.
inline bool litsFuse(uint8_t m) { return m == FIRE || m == LAVA || m == EMBER || m == BURNFUSE; }
inline void burnFuse(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == FUSE) {
                if (litsFuse(grid[i-1]) || litsFuse(grid[i+1]) || litsFuse(grid[i-SW]) || litsFuse(grid[i+SW])) r = 1;  // catches
            } else if (c == BURNFUSE) {
                r = 2;  // the tip lives one frame, then burns out to FIRE
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = BURNFUSE;
            else if (scratch[i] == 2) grid[i] = FIRE;
        }
}

// Cryo: a cryogenic coolant (liquid nitrogen) -- the first COLD liquid, the pourable
// counterpart to LAVA. It flows and pools like a light liquid (it floats on water, the
// way OIL does), and it is fiercely cold: WATER it touches flash-freezes to ICE, FIRE it
// touches is snuffed to EMPTY, and LAVA it touches is chilled straight to OBSIDIAN. Being
// volatile it boils away on its own -- a slow frame-hash evaporates it, and any cryo next
// to FIRE/LAVA boils off at once. Pour it over a lake to skate across, or quench a lava
// flow without the steam a water dousing would make. Two-pass snapshot: pass 1 marks each
// cell's fate (1 = freeze to ICE, 2 = vanish to EMPTY, 3 = chill to OBSIDIAN); pass 2
// applies. Order-independent / GPU-identical. Only ever consumed, so it terminates.
static constexpr uint32_t CRYO_EVAP = 3;       // of 256/frame -> a cryo pool boils off over ~1-2 s
inline bool cryoEvaporates(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 109u + (uint32_t)y * 233u + frame * 47u) & 0xFFu;
    return h < CRYO_EVAP;
}
inline void reactCryo(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            uint8_t n0 = grid[i-1], n1 = grid[i+1], n2 = grid[i-SW], n3 = grid[i+SW];
            bool nearCryo = n0==CRYO || n1==CRYO || n2==CRYO || n3==CRYO;
            if (c == CRYO) {
                bool hot = isHot(n0) || isHot(n1) || isHot(n2) || isHot(n3);   // FIRE or LAVA
                if (hot || cryoEvaporates(x, y, frame)) r = 2;                 // boils off
            } else if (c == WATER) {
                if (nearCryo) r = 1;                                           // flash-freeze -> ICE
            } else if (c == FIRE) {
                if (nearCryo) r = 2;                                           // snuffed cold -> EMPTY
            } else if (c == LAVA) {
                if (nearCryo) r = 3;                                           // chilled -> OBSIDIAN
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = ICE;
            else if (scratch[i] == 2) grid[i] = EMPTY;
            else if (scratch[i] == 3) grid[i] = OBSIDIAN;
        }
}

// Lamp: a circuit-driven light -- the WIREWORLD kit's visual OUTPUT (it already had the
// input SENSOR, the WIRE/EHEAD/ETAIL logic, the physical output IGNITER and the power
// source BATTERY, but no way to SEE a signal). A LAMP is a dark bulb that glows (becomes
// LAMPLIT) whenever an electron -- an EHEAD or ETAIL -- passes a cell next to it, and dims
// back the instant the pulse leaves. It never touches the circuit (electrons travel on
// WIRE; the lamp only watches), so a row of lamps beside a wire lights in sequence as a
// pulse runs past -- a marquee -- and a battery-clocked wire makes a lamp blink. Build
// glowing signs, bar displays, running lights. Two-pass snapshot: pass 1 marks each LAMP a
// passing electron lights and each LAMPLIT no longer beside one (which dims); pass 2 applies.
inline void lampLogic(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            bool nearE = grid[i-1]==EHEAD || grid[i+1]==EHEAD || grid[i-SW]==EHEAD || grid[i+SW]==EHEAD
                      || grid[i-1]==ETAIL || grid[i+1]==ETAIL || grid[i-SW]==ETAIL || grid[i+SW]==ETAIL;
            if (c == LAMP) { if (nearE) r = 1; }            // lit by a passing electron
            else if (c == LAMPLIT) { if (!nearE) r = 2; }   // dims when the pulse leaves
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = LAMPLIT;
            else if (scratch[i] == 2) grid[i] = LAMP;
        }
}

// Petrify: a creeping stone-curse -- medusa for the sandbox. A PETRIFY cell turns every
// LIVING thing it touches -- PLANT, WOOD, MOSS, CORAL -- to stone, and itself settles to
// OBSIDIAN a single frame later, so the curse sweeps through a connected mass of greenery
// one ring per frame and leaves a frozen OBSIDIAN statue behind (the wavefront is PETRIFY,
// the trail is stone -- the same one-frame-token trick FUSE uses). It is finite: living
// matter is consumed and every curse-cell settles, so a petrification always terminates.
// Drop it on a forest, a moss-clad wall or a coral reef and watch life turn to rock. Two-pass
// snapshot: pass 1 marks each living cell beside the curse (-> PETRIFY) and each PETRIFY
// (-> OBSIDIAN); pass 2 applies. Order-independent / GPU-identical.
inline bool petrifiable(uint8_t m) { return m == PLANT || m == WOOD || m == MOSS || m == CORAL; }
inline void petrify(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == PETRIFY) {
                r = 2;  // the curse-cell settles to stone after one frame
            } else if (petrifiable(c)) {
                if (grid[i-1]==PETRIFY || grid[i+1]==PETRIFY || grid[i-SW]==PETRIFY || grid[i+SW]==PETRIFY) r = 1;  // cursed
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = PETRIFY;
            else if (scratch[i] == 2) grid[i] = OBSIDIAN;
        }
}

static constexpr uint32_t SALT_MELT = 40;      // of 256/frame -> ice next to salt thaws (no heat)
inline bool saltMeltsIce(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 71u + (uint32_t)y * 187u + frame * 113u) & 0xFFu;
    return h < SALT_MELT;
}
static constexpr uint32_t SALT_DISSOLVE = 24;  // of 256/frame -> salt in water dissolves away
inline bool saltDissolves(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 211u + (uint32_t)y * 79u + frame * 167u) & 0xFFu;
    return h < SALT_DISSOLVE;
}

// Salt (de-icer): the ICE it touches melts to WATER without any heat, and the SALT
// itself dissolves away in WATER -- so a sprinkle on a frozen pond thaws a patch and
// then disappears into the meltwater. One combined two-pass snapshot (mark 1=salt
// dissolves to empty, 2=ice melts to water; then apply), order-independent and
// GPU-identical.
inline void saltCycle(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i], r = 0;
            if (c == SALT) {
                bool water = grid[i-1]==WATER || grid[i+1]==WATER || grid[i-SW]==WATER || grid[i+SW]==WATER;
                if (water && saltDissolves(x, y, frame)) r = 1;
            } else if (c == ICE) {
                bool salt = grid[i-1]==SALT || grid[i+1]==SALT || grid[i-SW]==SALT || grid[i+SW]==SALT;
                if (salt && saltMeltsIce(x, y, frame)) r = 2;
            }
            scratch[i] = r;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i] == 1) grid[i] = EMPTY;
            else if (scratch[i] == 2) grid[i] = WATER;
        }
}

static constexpr uint32_t MERC_POISON = 20;    // of 256/frame -> plant touching mercury withers
inline bool mercuryPoisons(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 101u + (uint32_t)y * 217u + frame * 131u) & 0xFFu;
    return h < MERC_POISON;
}
// Mercury poisoning: PLANT touching the toxic liquid metal MERCURY withers to EMPTY
// (frame-hashed, so a vine dies back gradually). Two-pass snapshot -> order-
// independent and GPU-identical.
inline void poisonMercury(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool merc = grid[i-1]==MERCURY || grid[i+1]==MERCURY || grid[i-SW]==MERCURY || grid[i+SW]==MERCURY;
            scratch[i] = (grid[i] == PLANT && merc && mercuryPoisons(x, y, frame)) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = EMPTY;
        }
}

// Which cells an explosion consumes: the soft / flammable stuff (and TNT itself,
// so blasts chain). WALL, GLASS, WATER, LAVA, ICE, STEAM, ACID, SPRING survive --
// a blast meeting water just stops at the waterline and gets quenched.
inline bool blastable(uint8_t m) {
    return m == EMPTY || m == SAND || m == OIL || m == GAS || m == WOOD || m == PLANT || m == SMOKE || m == TNT || m == GUNPOWDER;
}

// TNT: an explosive that detonates when FIRE/LAVA touches it, bursting into FIRE
// across its 8-neighbourhood and chain-detonating adjacent TNT. Two-pass snapshot
// through the scratch buffer: pass 1 marks every TNT cell touching something hot
// (the detonators); pass 2 turns each detonator -- and every blastable cell next to
// one -- into FIRE. Pass 2 reads the pass-1 marks (in scratch) plus its own grid
// cell, writing only itself, so it stays order-independent and GPU-identical. The
// blast wave then expands one ring per frame as the new fire detonates the next TNT.
inline void detonateTnt(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
            scratch[i] = ((grid[i] == TNT || grid[i] == GUNPOWDER) && hot) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool nearBlast = scratch[i-1] || scratch[i+1] || scratch[i-SW] || scratch[i+SW] ||
                             scratch[i-SW-1] || scratch[i-SW+1] || scratch[i+SW-1] || scratch[i+SW+1];
            if (scratch[i] || (nearBlast && blastable(grid[i]))) grid[i] = FIRE;
        }
}

// Which solids THERMITE melts through: the otherwise-toughest stuff (WALL/GLASS/
// OBSIDIAN -- nothing else touches them) plus SAND and WOOD. Liquids/powders/EMPTY
// are skipped (it just combusts amid them).
inline bool meltable(uint8_t m) {
    return m == WALL || m == GLASS || m == OBSIDIAN || m == SAND || m == WOOD;
}

// THERMITE: an incendiary powder that burns through stone. Same two-pass snapshot
// shape as detonateTnt -- pass 1 marks every THERMITE cell touching something hot
// (the cells that ignite this frame); pass 2 turns each marked cell into FIRE (it
// combusts) and melts every adjacent meltable solid into LAVA. Pass 2 reads the
// pass-1 marks (in scratch) of its 4 neighbours plus its own grid cell, writing only
// itself, so it stays order-independent and GPU-identical. The burn front advances
// one ring per frame (the new fire lights the next thermite), eating a molten cavity.
inline void burnThermite(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            bool hot = isHot(grid[i-1]) || isHot(grid[i+1]) || isHot(grid[i-SW]) || isHot(grid[i+SW]);
            scratch[i] = (grid[i] == THERMITE && hot) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) { grid[i] = FIRE; continue; }
            bool nearBurn = scratch[i-1] || scratch[i+1] || scratch[i-SW] || scratch[i+SW];
            if (nearBurn && meltable(grid[i])) grid[i] = LAVA;
        }
}
