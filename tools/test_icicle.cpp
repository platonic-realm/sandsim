// Unit test for ICICLE: a tip that grows downward, laying ICE, then freezes off.
#include "../cpp/materials.h"
#include <cstdio>
#include <vector>

static const int SWi = 12, SHi = 40;
static std::vector<uint8_t> g, s;
static uint8_t& C(int x, int y) { return g[(size_t)y * SWi + x]; }
static void clear() { g.assign(SWi * SHi, EMPTY); s.assign(SWi * SHi, 0); }
static void run(uint32_t f) { growIcicle(g.data(), s.data(), SWi, 1, SWi - 1, 1, SHi - 1, f); }
static int count(uint8_t m) { int n = 0; for (auto v : g) if (v == m) ++n; return n; }

int main() {
    int fails = 0;

    // 1. A tip descends one cell per frame, laying ICE behind it.
    clear();
    C(6, 2) = ICICLE; C(6, 1) = WALL;   // hangs from a ceiling
    run(0);
    if (C(6, 2) != ICE || C(6, 3) != ICICLE) {
        printf("FAIL: tip didn't descend leaving ice (at(6,2)=%d at(6,3)=%d)\n", C(6,2), C(6,3)); ++fails;
    } else printf("ok: ICICLE descends one cell/frame, laying ICE behind it\n");

    // 2. After several frames it has built a vertical column of ICE with one tip.
    clear();
    C(6, 2) = ICICLE; C(6, 1) = WALL;
    for (uint32_t f = 0; f < 6; ++f) run(f);
    int ice = count(ICE), tip = count(ICICLE);
    if (ice < 5 || tip != 1) { printf("FAIL: expected a growing ice column + 1 tip (ice=%d tip=%d)\n", ice, tip); ++fails; }
    else printf("ok: builds a hanging ICE column with a single live tip (ice=%d)\n", ice);

    // 3. Hitting a floor freezes the tip (no ICICLE left, all ICE), growth stops.
    clear();
    C(6, 2) = ICICLE; C(6, 1) = WALL; C(6, 6) = WALL;   // floor 4 cells below
    for (uint32_t f = 0; f < 40; ++f) run(f);
    if (count(ICICLE) != 0) { printf("FAIL: tip did not freeze at the floor (tip=%d)\n", count(ICICLE)); ++fails; }
    else printf("ok: the tip freezes solid when it reaches a floor\n");

    // 4. It eventually tapers off on its own (finite length) even with open space below.
    clear();
    C(6, 2) = ICICLE; C(6, 1) = WALL;
    bool stopped = false;
    for (uint32_t f = 0; f < 2000 && !stopped; ++f) { run(f); if (count(ICICLE) == 0) stopped = true; }
    if (!stopped) { printf("FAIL: icicle never tapered off\n"); ++fails; }
    else printf("ok: the icicle tapers off on its own (finite length)\n");

    // 5. Determinism.
    clear(); C(6,2)=ICICLE; C(6,1)=WALL; for(uint32_t f=0;f<30;++f) run(f); auto a=g;
    clear(); C(6,2)=ICICLE; C(6,1)=WALL; for(uint32_t f=0;f<30;++f) run(f);
    if (g != a) { printf("FAIL: non-deterministic\n"); ++fails; }
    else printf("ok: deterministic\n");

    printf(fails ? "\n%d FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
