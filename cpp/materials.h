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
// SAND in LAVA, and it resists fire and acid.
enum Material : uint8_t {
    EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7,
    STEAM = 8, WOOD = 9, PLANT = 10, ACID = 11, SMOKE = 12, GLASS = 13, MATERIAL_COUNT = 14
};

// Fire burn-out: a per-cell, time-varying transform that is a PURE function of
// (x, y, frame) -- no neighbour reads -- so it stays order-independent and is
// bit-identical on CPU SIMD and the GPU compute backends (which compute the same
// hash). ~FIRE_DECAY/256 of the flame winks out each frame.
static constexpr uint32_t FIRE_DECAY = 12;     // of 256 -> avg flame life ~21 frames
static constexpr uint32_t SMOKE_FROM_FIRE = 4; // of those burn-outs, this many leave smoke
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
                if (h < FIRE_DECAY) grid[i] = (h < SMOKE_FROM_FIRE) ? SMOKE : EMPTY;  // some flame -> smoke
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
            bool ign = ((c == OIL || c == PLANT) && hot)      // oil & dry plant: instant
                    || (c == WOOD && hot && woodCatches(x, y, frame));   // wood: slow
            scratch[i] = ign ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = FIRE;
        }
}

// Water meets hot: when WATER touches FIRE or LAVA, the water flashes to STEAM,
// fire is quenched to EMPTY, and lava freezes to WALL (stone). One two-pass
// snapshot (the `moved` scratch): pass 1 marks every cell at such an interface,
// pass 2 transforms each by its own type -- order-independent, GPU-identical.
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
            else if (c == FIRE)  react = nb(i, WATER);
            else if (c == LAVA)  react = nb(i, WATER);
            scratch[i] = react ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (!scratch[i]) continue;
            uint8_t c = grid[i];
            grid[i] = (c == WATER) ? STEAM : (c == FIRE) ? EMPTY : WALL;   // lava -> stone
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
