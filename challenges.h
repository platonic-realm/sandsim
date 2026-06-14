// Challenge mode: a handful of self-contained mini-puzzles that turn the sandbox into a
// little game. Each challenge stamps a starting scene into the viewport and reports a
// 0..1 progress toward its goal, evaluated on the live cells every frame (solved at >= 1).
// Everything here is backend-agnostic -- a challenge only fills/reads a flat W*H cell
// buffer -- so all three viewers share it (the host stamps the scene via world.loadView
// and feeds back the live viewport).
//
// Include AFTER the Material enum (uses the bare enum names, like worldgen.h).
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace chal {

inline void rect(uint8_t* c, int W, int H, int x0, int y0, int x1, int y1, uint8_t m) {
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            if (x >= 0 && x < W && y >= 0 && y < H) c[(size_t)y * W + x] = m;
}
inline int count(const uint8_t* c, int W, int H, uint8_t m) {
    int n = 0; for (int i = 0; i < W * H; ++i) if (c[i] == m) ++n; return n;
}
inline float frac01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// 1. FILL THE TANK -- pour WATER into the open-topped tank until it is mostly full.
inline void buildTank(uint8_t* c, int W, int H) {
    int x0 = W / 4, x1 = 3 * W / 4, top = H / 4, bot = 3 * H / 4;
    rect(c, W, H, x0, top, x0 + 2, bot, WALL);
    rect(c, W, H, x1 - 2, top, x1, bot, WALL);
    rect(c, W, H, x0, bot - 2, x1, bot, WALL);
}
inline float progTank(const uint8_t* c, int W, int H) {
    int x0 = W / 4 + 3, x1 = 3 * W / 4 - 3, top = H / 4 + 2, bot = 3 * H / 4 - 2;
    int w = 0, cells = 0;
    for (int y = top; y < bot; ++y)
        for (int x = x0; x < x1; ++x) { ++cells; if (c[(size_t)y * W + x] == WATER) ++w; }
    return cells > 0 ? frac01(((float)w / cells) / 0.6f) : 0.0f;   // 60% full = done
}

// 2. MELT THE ICE -- a block of ICE, inert until you bring heat (FIRE / LAVA).
inline void buildIce(uint8_t* c, int W, int H) {
    rect(c, W, H, W / 8, 3 * H / 4, 7 * W / 8, 3 * H / 4 + 2, WALL);
    rect(c, W, H, 3 * W / 8, H / 3, 5 * W / 8, 3 * H / 4 - 1, ICE);
}
inline float progIce(const uint8_t* c, int W, int H) {
    int init = (5 * W / 8 - 3 * W / 8 + 1) * ((3 * H / 4 - 1) - H / 3 + 1);
    int cur = count(c, W, H, ICE);
    if (init <= 4) return 1.0f;
    return frac01((float)(init - cur) / (init - 4));               // melted away = done
}

// 3. BREACH THE WALL -- a thick stone barrier; only destroyers can punch through.
inline void buildVault(uint8_t* c, int W, int H) {
    int by = H / 2;
    rect(c, W, H, W / 6, by - 1, 5 * W / 6, by + 1, WALL);
    rect(c, W, H, W / 6, 3 * H / 4, 5 * W / 6, 3 * H / 4 + 1, WALL);
}
inline float progVault(const uint8_t* c, int W, int H) {
    int by = H / 2, x0 = W / 6, x1 = 5 * W / 6, run = 0, best = 0;
    for (int x = x0; x <= x1; ++x) {
        bool wall = false;
        for (int dy = -1; dy <= 1; ++dy) if (c[(size_t)(by + dy) * W + x] == WALL) wall = true;
        if (!wall) { ++run; if (run > best) best = run; } else run = 0;
    }
    return frac01((float)best / 4.0f);                             // a 4-wide breach = done
}

// 4. GROW A FOREST -- plant SEED (or PLANT) on the damp soil and raise a canopy.
inline void buildForest(uint8_t* c, int W, int H) {
    int fl = 3 * H / 4;
    rect(c, W, H, W / 8, fl, 7 * W / 8, fl + 2, WALL);             // bedrock
    rect(c, W, H, W / 8, fl - 3, 7 * W / 8, fl - 1, SAND);         // sandy soil
    rect(c, W, H, 2 * W / 5, fl - 1, 3 * W / 5, fl - 1, WATER);    // a damp patch
}
inline float progForest(const uint8_t* c, int W, int H) {
    int g = count(c, W, H, PLANT) + count(c, W, H, WOOD) + count(c, W, H, SPROUT);
    return frac01((float)g / 50.0f);                              // a little forest = done
}

// 5. FREEZE THE LAKE -- chill a pool of water solid (FROST/CRYO/ICE), the mirror of #2.
inline void buildLake(uint8_t* c, int W, int H) {
    int x0 = W / 5, x1 = 4 * W / 5, top = H / 3, bot = 3 * H / 4;
    rect(c, W, H, x0, top, x0 + 2, bot, WALL);
    rect(c, W, H, x1 - 2, top, x1, bot, WALL);
    rect(c, W, H, x0, bot - 2, x1, bot, WALL);
    rect(c, W, H, x0 + 3, top + 2, x1 - 3, bot - 3, WATER);
}
inline float progLake(const uint8_t* c, int W, int H) {
    int x0 = W / 5 + 3, x1 = 4 * W / 5 - 3, top = H / 3 + 2, bot = 3 * H / 4 - 3;
    int ice = 0, total = 0;
    for (int y = top; y < bot; ++y)
        for (int x = x0; x < x1; ++x) { uint8_t m = c[(size_t)y * W + x]; if (m == ICE) ++ice; if (m == ICE || m == WATER) ++total; }
    return total > 0 ? frac01(((float)ice / total) / 0.7f) : 0.0f; // 70% frozen = done
}

struct Challenge {
    const char* name;
    const char* goal;
    void  (*build)(uint8_t*, int, int);
    float (*progress)(const uint8_t*, int, int);   // 0..1, solved at >= 1
};

static const Challenge kChallenges[] = {
    { "FILL THE TANK",   "pour WATER until the tank is full",            buildTank,   progTank   },
    { "MELT THE ICE",    "bring heat (FIRE/LAVA) to melt the ice",       buildIce,    progIce    },
    { "BREACH THE WALL", "blast through the stone (TNT/ACID/THERMITE)",  buildVault,  progVault  },
    { "GROW A FOREST",   "plant SEED on the damp soil and grow trees",   buildForest, progForest },
    { "FREEZE THE LAKE", "chill the water solid (FROST/CRYO)",           buildLake,   progLake   },
};
static const int kNumChallenges = (int)(sizeof(kChallenges) / sizeof(kChallenges[0]));

} // namespace chal
