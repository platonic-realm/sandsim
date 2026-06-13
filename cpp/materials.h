// Material ids shared by the baseline host code and the SIMD step TUs.
#pragma once
#include <cstdint>
#include <cstddef>

// Density order (heavy -> light): SAND > LAVA > WATER > OIL > air > GAS > FIRE.
// OIL floats on water; FIRE rises and burns out; LAVA is a heavy molten liquid
// that sets oil ablaze and freezes to WALL where it meets water.
enum Material : uint8_t {
    EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7, MATERIAL_COUNT = 8
};

// Fire burn-out: a per-cell, time-varying transform that is a PURE function of
// (x, y, frame) -- no neighbour reads -- so it stays order-independent and is
// bit-identical on CPU SIMD and the GPU compute backends (which compute the same
// hash). ~FIRE_DECAY/256 of the flame winks out each frame.
static constexpr uint32_t FIRE_DECAY = 12;   // of 256 -> avg flame life ~21 frames

inline bool fireBurnsOut(int x, int y, uint32_t frame) {
    uint32_t h = ((uint32_t)x * 167u + (uint32_t)y * 101u + frame * 131u) & 0xFFu;
    return h < FIRE_DECAY;
}

// Run the burn-out over the live interior (scalar; cheap and identical on every
// SIMD width). Gated by the world on "is there any fire?" so fire-free worlds
// (e.g. the benchmark) pay nothing.
inline void decayFire(uint8_t* grid, int SW, int X0, int X1, int Y0, int Y1, uint32_t frame) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (grid[i] == FIRE && fireBurnsOut(x, y, frame)) grid[i] = EMPTY;
        }
}

// Ignition: OIL touching FIRE (4-neighbour) catches and becomes FIRE, so flame
// spreads through fuel one layer per frame. Done in two per-cell passes through a
// scratch buffer (the `moved` flags, free after the movement step): pass 1 reads
// a consistent snapshot of the grid and records intent; pass 2 applies it. Each
// pass reads one buffer and writes another, so it is order-independent and the
// GPU reproduces it bit-for-bit.
inline bool isHot(uint8_t m) { return m == FIRE || m == LAVA; }   // ignites adjacent oil

inline void igniteFire(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            scratch[i] = (grid[i] == OIL &&
                          (isHot(grid[i - 1]) || isHot(grid[i + 1]) ||
                           isHot(grid[i - SW]) || isHot(grid[i + SW]))) ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = FIRE;
        }
}

// LAVA + WATER freeze to stone: any LAVA touching WATER (or vice-versa) becomes
// WALL. Same two-pass snapshot as ignition so it's order-independent and the GPU
// matches exactly.
inline void lavaReact(uint8_t* grid, uint8_t* scratch, int SW, int X0, int X1, int Y0, int Y1) {
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            uint8_t c = grid[i];
            bool react = false;
            if (c == LAVA)
                react = (grid[i-1]==WATER || grid[i+1]==WATER || grid[i-SW]==WATER || grid[i+SW]==WATER);
            else if (c == WATER)
                react = (grid[i-1]==LAVA || grid[i+1]==LAVA || grid[i-SW]==LAVA || grid[i+SW]==LAVA);
            scratch[i] = react ? 1 : 0;
        }
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x) {
            size_t i = (size_t)y * SW + x;
            if (scratch[i]) grid[i] = WALL;
        }
}
