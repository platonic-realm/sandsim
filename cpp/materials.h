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
enum Material : uint8_t {
    EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7,
    STEAM = 8, WOOD = 9, PLANT = 10, ACID = 11, SMOKE = 12, GLASS = 13, ICE = 14,
    SPRING = 15, TNT = 16, ASH = 17, VOLCANO = 18, VOID = 19, MUD = 20, VIRUS = 21,
    SPARK = 22, OBSIDIAN = 23, SALT = 24, SNOW = 25, MERCURY = 26, GUNPOWDER = 27,
    THERMITE = 28,
    MATERIAL_COUNT = 29
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
            bool ign = ((c == OIL || c == PLANT || c == GAS) && hot)   // oil, plant & gas: instant
                    || (c == WOOD && hot && woodCatches(x, y, frame)); // wood: slow
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
            else if (c == GAS || c == OIL) { if (nSpark) r = 4; }
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
