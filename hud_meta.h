// Shared viewer metadata: material display names, the category-grouped palette layout,
// and per-material glow strength. Uses the bare Material enum names, so every backend
// includes this AFTER defining the enum (identical values), exactly like worldgen.h.
// Keeping it here means the CPU/SDL, Vulkan/SDL and OpenGL viewers share one source of
// truth for names/categories/glow instead of drifting per-file copies.
#pragma once
#include <cstdint>

// Display names for the HUD/tooltips -- one per material id, in enum order.
static const char* kNames[MATERIAL_COUNT] = {
    "ERASER", "WALL", "SAND", "WATER", "GAS", "OIL", "FIRE", "LAVA", "STEAM", "WOOD",
    "PLANT", "ACID", "SMOKE", "GLASS", "ICE", "SPRING", "TNT", "ASH", "VOLCANO", "VOID",
    "MUD", "VIRUS", "SPARK", "OBSIDIAN", "SALT", "SNOW", "MERCURY", "GUNPOWDER", "THERMITE", "FROST",
    "WISP", "COAL", "EMBER", "CLONER", "CRYSTAL", "ANTIMATTER", "MOSS", "FUMES", "WIRE", "E-HEAD",
    "E-TAIL", "IGNITER", "SENSOR", "LIFE", "GEYSER", "LYE", "SODIUM", "CORAL", "PHOSPHORUS", "CEMENT",
    "CHLORINE", "BATTERY", "FUSE", "BURN-FUSE", "CRYO", "LAMP", "LAMP-LIT", "PETRIFY", "FIREWORK", "LEVITON",
    "SPROUT", "BELT", "MAGNET", "IRON", "NITRO", "RUST", "SEED", "LASER", "BEAM", "ICICLE",
};

// The palette is laid out grouped by category (a navigable toolbox) rather than in raw
// id order. kPaletteOrder is the swatch order (material ids); kSlotCat[i] is the category
// of swatch i; each category has a name and an accent colour drawn under its swatches.
static const char* kCatNames[] = {
    "TOOLS", "SOLIDS", "POWDERS", "LIQUIDS", "GASES", "FIRE & HEAT",
    "EXPLOSIVES", "LIFE & GROWTH", "CIRCUITS", "MACHINES", "SOURCES & MAGIC",
};
static const uint32_t kCatAccent[] = {
    0xFF8890A0u, 0xFF6E6E78u, 0xFFC8A060u, 0xFF4488FFu, 0xFFA6C2D2u, 0xFFFF6633u,
    0xFFFF3344u, 0xFF44C060u, 0xFFFFCC22u, 0xFFB060FFu, 0xFF40E0C0u,
};
static const uint8_t kPaletteOrder[MATERIAL_COUNT] = {
    EMPTY,                                                                       // tools
    WALL, WOOD, GLASS, ICE, OBSIDIAN,                                            // solids
    SAND, ASH, SALT, SNOW, MUD, COAL, LYE, CEMENT, IRON, RUST, LEVITON, SEED,    // powders
    WATER, OIL, ACID, MERCURY, CRYO,                                            // liquids
    GAS, STEAM, SMOKE, WISP, FUMES, CHLORINE,                                   // gases
    FIRE, LAVA, EMBER, FROST,                                                   // fire & heat
    TNT, GUNPOWDER, THERMITE, SODIUM, PHOSPHORUS, NITRO, FUSE, BURNFUSE,        // explosives
    PLANT, CRYSTAL, MOSS, CORAL, SPROUT, ICICLE, VIRUS, LIFE,                   // life & growth
    SPARK, WIRE, EHEAD, ETAIL, IGNITER, SENSOR, BATTERY, LAMP, LAMPLIT,         // circuits
    CLONER, BELT, MAGNET, LASER, BEAM, ANTIMATTER,                             // machines
    SPRING, VOLCANO, GEYSER, VOID, PETRIFY, FIREWORK,                          // sources & magic
};
static const uint8_t kSlotCat[MATERIAL_COUNT] = {
    0,
    1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,
    4,4,4,4,4,4,
    5,5,5,5,
    6,6,6,6,6,6,6,6,
    7,7,7,7,7,7,7,7,
    8,8,8,8,8,8,8,8,8,
    9,9,9,9,9,9,
    10,10,10,10,10,10,
};

// Render-only: how brightly each material glows. Emissive cells cast a soft additive
// bloom into their surroundings so fire, lava, lasers and lamps light up the scene.
static inline float emissionStrength(uint8_t m) {
    switch (m) {
        case FIRE:       return 1.00f;
        case BEAM:       return 1.00f;
        case SPARK:      return 0.90f;
        case LAVA:       return 0.85f;
        case LAMPLIT:    return 0.80f;
        case EMBER:      return 0.70f;
        case FIREWORK:   return 0.70f;
        case BURNFUSE:   return 0.70f;
        case LASER:      return 0.60f;
        case ANTIMATTER: return 0.60f;
        case EHEAD:      return 0.50f;
        default:         return 0.00f;
    }
}
