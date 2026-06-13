// Material ids shared by the baseline host code and the SIMD step TUs.
#pragma once
#include <cstdint>
#include <cstddef>

// Density order (heavy -> light): SAND > WATER > OIL > air(EMPTY) > GAS > FIRE.
// OIL floats on water; FIRE rises like flame and burns out over time.
enum Material : uint8_t {
    EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, MATERIAL_COUNT = 7
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
