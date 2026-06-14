// Freeplay scenes: ready-made sandboxes you can drop in and play with (press F1 to cycle
// them). Unlike challenges there is no goal -- they just stamp a fun, living starting
// scene that shows off the materials. Backend-agnostic: a scene only fills a flat W*H
// cell buffer, stamped into the viewport via world.loadView, so all three viewers share
// them. Include AFTER the Material enum (uses the bare enum names, like worldgen.h).
#pragma once
#include <cstdint>
#include <cstddef>

namespace scene {

inline void rect(uint8_t* c, int W, int H, int x0, int y0, int x1, int y1, uint8_t m) {
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            if (x >= 0 && x < W && y >= 0 && y < H) c[(size_t)y * W + x] = m;
}
inline void dot(uint8_t* c, int W, int H, int x, int y, uint8_t m) {
    if (x >= 0 && x < W && y >= 0 && y < H) c[(size_t)y * W + x] = m;
}

// A volcano: a stone cone with a vent shaft down to a lava chamber and a VOLCANO emitter,
// so lava wells up and erupts out of the top.
inline void buildVolcano(uint8_t* c, int W, int H) {
    int peak = H / 5, base = 4 * H / 5, cx = W / 2, halfMax = W / 3;
    for (int y = peak; y <= base; ++y) {
        int half = (y - peak) * halfMax / (base - peak);
        rect(c, W, H, cx - half, y, cx + half, y, WALL);     // filled cone
    }
    rect(c, W, H, cx - 1, peak + 1, cx + 1, base, EMPTY);    // carve the vent shaft
    rect(c, W, H, cx - 5, base - 8, cx + 5, base, LAVA);     // lava chamber
    rect(c, W, H, cx - 1, base, cx + 1, base, VOLCANO);      // keeps it erupting
}

// A fireworks volley: a row of rockets on the ground that launch and burst overhead.
inline void buildFireworks(uint8_t* c, int W, int H) {
    int fl = 4 * H / 5;
    rect(c, W, H, 0, fl, W - 1, fl + 3, WALL);               // ground
    int step = W / 16 > 0 ? W / 16 : 1;
    for (int x = W / 8; x < 7 * W / 8; x += step)
        rect(c, W, H, x, fl - 2, x, fl - 1, FIREWORK);       // rockets
}

// An aquarium: a glass tank of water with a sandy bed, coral and a few plants.
inline void buildAquarium(uint8_t* c, int W, int H) {
    int x0 = W / 6, x1 = 5 * W / 6, top = H / 5, bot = 4 * H / 5;
    rect(c, W, H, x0, top, x0 + 1, bot, GLASS);              // left pane
    rect(c, W, H, x1 - 1, top, x1, bot, GLASS);              // right pane
    rect(c, W, H, x0, bot - 1, x1, bot, GLASS);              // floor pane
    rect(c, W, H, x0 + 2, top + 1, x1 - 2, bot - 2, WATER);  // water
    rect(c, W, H, x0 + 2, bot - 3, x1 - 2, bot - 2, SAND);   // sandy bed
    int cs = W / 9 > 0 ? W / 9 : 1, ps = W / 7 > 0 ? W / 7 : 1;
    for (int x = x0 + 5; x < x1 - 3; x += cs) dot(c, W, H, x, bot - 4, CORAL);
    for (int x = x0 + 8; x < x1 - 3; x += ps) { dot(c, W, H, x, bot - 4, PLANT); dot(c, W, H, x, bot - 5, PLANT); }
}

struct Scene { const char* name; void (*build)(uint8_t*, int, int); };

static const Scene kScenes[] = {
    { "VOLCANO",   buildVolcano   },
    { "FIREWORKS", buildFireworks },
    { "AQUARIUM",  buildAquarium  },
};
static const int kNumScenes = (int)(sizeof(kScenes) / sizeof(kScenes[0]));

} // namespace scene
