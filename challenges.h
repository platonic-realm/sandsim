// Challenge mode: a handful of self-contained mini-puzzles that turn the sandbox into a
// little game. Each challenge stamps a starting scene into the viewport and has a win
// predicate evaluated on the live cells every frame. Everything here is backend-agnostic
// -- a challenge only fills/reads a flat W*H cell buffer -- so all three viewers share it
// (the host stamps the scene via world.loadView and feeds back the live viewport).
//
// Include AFTER the Material enum (uses the bare enum names, like worldgen.h).
#pragma once
#include <cstdint>
#include <cstddef>

namespace chal {

inline void rect(uint8_t* c, int W, int H, int x0, int y0, int x1, int y1, uint8_t m) {
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            if (x >= 0 && x < W && y >= 0 && y < H) c[(size_t)y * W + x] = m;
}
inline int count(const uint8_t* c, int W, int H, uint8_t m) {
    int n = 0; for (int i = 0; i < W * H; ++i) if (c[i] == m) ++n; return n;
}

// 1. FILL THE TANK -- pour WATER into the open-topped tank until it is mostly full.
inline void buildTank(uint8_t* c, int W, int H) {
    int x0 = W / 4, x1 = 3 * W / 4, top = H / 4, bot = 3 * H / 4;
    rect(c, W, H, x0, top, x0 + 2, bot, WALL);       // left wall
    rect(c, W, H, x1 - 2, top, x1, bot, WALL);       // right wall
    rect(c, W, H, x0, bot - 2, x1, bot, WALL);       // floor
}
inline bool wonTank(const uint8_t* c, int W, int H) {
    int x0 = W / 4 + 3, x1 = 3 * W / 4 - 3, top = H / 4 + 2, bot = 3 * H / 4 - 2;
    int w = 0, cells = 0;
    for (int y = top; y < bot; ++y)
        for (int x = x0; x < x1; ++x) { ++cells; if (c[(size_t)y * W + x] == WATER) ++w; }
    return cells > 0 && w >= cells * 3 / 5;          // tank ~60% full
}

// 2. MELT THE ICE -- a block of ICE that is inert until you bring heat (FIRE / LAVA).
inline void buildIce(uint8_t* c, int W, int H) {
    rect(c, W, H, W / 8, 3 * H / 4, 7 * W / 8, 3 * H / 4 + 2, WALL);   // floor
    rect(c, W, H, 3 * W / 8, H / 3, 5 * W / 8, 3 * H / 4 - 1, ICE);    // ice block
}
inline bool wonIce(const uint8_t* c, int W, int H) {
    return count(c, W, H, ICE) <= 4;                 // melted away
}

// 3. BREACH THE WALL -- a thick stone barrier; only destroyers (TNT, ACID, THERMITE,
//    NITRO, ANTIMATTER, LASER) can punch through. Win on a 4-wide breach.
inline void buildVault(uint8_t* c, int W, int H) {
    int by = H / 2;
    rect(c, W, H, W / 6, by - 1, 5 * W / 6, by + 1, WALL);            // the barrier
    rect(c, W, H, W / 6, 3 * H / 4, 5 * W / 6, 3 * H / 4 + 1, WALL);  // floor below
}
inline bool wonVault(const uint8_t* c, int W, int H) {
    int by = H / 2, x0 = W / 6, x1 = 5 * W / 6, run = 0, best = 0;
    for (int x = x0; x <= x1; ++x) {
        bool wall = false;
        for (int dy = -1; dy <= 1; ++dy) if (c[(size_t)(by + dy) * W + x] == WALL) wall = true;
        if (!wall) { ++run; if (run > best) best = run; } else run = 0;
    }
    return best >= 4;
}

struct Challenge {
    const char* name;
    const char* goal;
    void (*build)(uint8_t*, int, int);
    bool (*won)(const uint8_t*, int, int);
};

static const Challenge kChallenges[] = {
    { "FILL THE TANK",  "pour WATER until the tank is full",          buildTank,  wonTank  },
    { "MELT THE ICE",   "bring heat (FIRE/LAVA) to melt the ice",     buildIce,   wonIce   },
    { "BREACH THE WALL","blast through the stone (TNT/ACID/THERMITE)", buildVault, wonVault },
};
static const int kNumChallenges = (int)(sizeof(kChallenges) / sizeof(kChallenges[0]));

} // namespace chal
