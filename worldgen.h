// Shared world generator -- one deterministic, hash-based seed used by every
// backend, so the C++ SIMD, OpenGL and Vulkan builds all conjure the SAME diverse
// world from nothing (and the streaming benchmark, which generates and pans this
// world, stays bit-identical across all three). Uses only integer hash math and the
// Material enum names; each backend defines the enum (with identical values) before
// including this header, so the bare names below resolve in every translation unit.
#pragma once
#include <cstdint>

static inline uint32_t hashCoord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}

// A diverse, randomised-but-deterministic world: an open sky over a rolling
// surface whose tall peaks are SNOW-capped, a sandy crust dotted with plant and
// wood, and an underground of rock carved by caves and packed with region-clustered
// pools and seams -- water, oil, acid, gas, ice, plus buried COAL seams, pale SALT
// deposits and silvery MERCURY pools -- with LAVA welling up from the deep and the
// rare water SPRING or cache of TNT. Caverns hold pockets of bubbling WISP (marsh
// gas) and the odd GEYSER vent that pulses steam on a cycle. Materials are clustered
// (the pool choice is keyed to a coarse region hash) rather than salt-and-pepper, so
// the world reads as pools, veins and caverns that then come alive as the reactions run
// (coal seams catch where lava meets them, snow melts at the warm edges, geysers gush
// steam that rains back, marsh gas drifts up and flares near a vein of lava, and so on).
static inline uint8_t seedMat(int gx, int gy) {
    uint32_t cell = hashCoord(gx, gy);

    // rolling surface line: a coarse hill plus a finer ripple
    int hillC = (int)(hashCoord(gx >> 6, 0x51) % 64u);     // per 64 cols, 0..63
    int hillF = (int)(hashCoord(gx >> 4, 0xA2) % 20u);     // per 16 cols, 0..19
    int surface = 64 + hillC + hillF;                      // ~64..146

    // --- sky: open air with the odd drifting gas cloud ---
    if (gy < surface) {
        if ((hashCoord(gx >> 4, gy >> 4) % 1000u) < 30u && (cell % 100u) < 55u) return GAS;
        return EMPTY;
    }

    int depth = gy - surface;

    // --- topsoil: a sandy crust with plant tufts and wood posts; tall peaks wear snow ---
    if (depth < 6) {
        uint32_t s = cell % 1000u;
        if (surface < 84) {                                // the high peaks are snow-capped
            return (s < 70u) ? WALL : SNOW;                // bare rock pokes through the drift
        }
        if (s < 110u) return PLANT;
        if (s < 140u) return WOOD;
        return SAND;
    }

    // --- pick a clustered pool material for this ~16-cell region ---
    uint32_t region = hashCoord((gx >> 4) + 31, (gy >> 4) + 67) % 1000u;
    uint8_t pool;
    if      (region < 200u) pool = WATER;
    else if (region < 330u) pool = OIL;
    else if (region < 420u) pool = ACID;
    else if (region < 500u) pool = GAS;
    else if (region < 580u) pool = ICE;
    else if (region < 650u) pool = PLANT;
    else if (region < 710u) pool = SAND;
    else if (region < 790u) pool = COAL;      // buried coal seams
    else if (region < 850u) pool = SALT;      // pale salt deposits
    else if (region < 895u) pool = MERCURY;   // silvery quicksilver pools
    else                    pool = WATER;

    // lava wells up from the deep, taking over the pool the lower we go
    if (depth > 150 || (depth > 90 && (hashCoord((gx >> 4) + 5, (gy >> 4) + 9) % 1000u) < 320u))
        pool = LAVA;

    // rare buried features: a water spring, a cache of tnt, or a deep lava vent
    uint32_t rare = cell % 100000u;
    if (rare < 6u)  return SPRING;
    if (rare < 16u) return TNT;
    if (rare < 19u && depth > 110) return VOLCANO;   // volcanoes only well down deep

    // caves: a coarse hash opens roomy caverns, textured by a fine hash
    bool cave = (hashCoord((gx >> 4) + 808, (gy >> 4) + 909) % 1000u) < 340u && (cell % 1000u) < 800u;
    if (cave) {
        if ((cell % 1000u) < 2u && depth > 40)   return GEYSER;  // rare geothermal vents pulse steam into caverns
        if ((cell % 1000u) < 9u && depth > 30)   return WISP;    // pockets of marsh gas bubble up and gather
        if ((cell % 100u) < 18u) return pool;              // scattered deposits / drips
        return EMPTY;                                      // open cavern
    }

    // solid ground: a rock matrix veined with ore pockets of the region pool
    if ((hashCoord((gx >> 3) + 17, (gy >> 3) + 23) % 1000u) < 500u) return WALL;
    if ((cell % 100u) < 80u) return pool;
    return WALL;
}
