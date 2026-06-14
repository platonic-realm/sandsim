// Unit test for SEED: a grounded seed beside water germinates into a SPROUT.
#include "../cpp/world_step.h"
#include "../cpp/materials.h"
#include <cstdio>
#include <vector>

static const int SWs = 16, SHs = 12;
static std::vector<uint8_t> g, s;
static uint8_t& C(int x, int y) { return g[(size_t)y * SWs + x]; }
static void run(uint32_t f) { germinateSeed(g.data(), s.data(), SWs, 1, SWs - 1, 1, SHs - 1, f); }
static void clear() { g.assign(SWs * SHs, EMPTY); s.assign(SWs * SHs, 0); }

int main() {
    int fails = 0;

    // 1. SEED resting on solid ground beside WATER germinates into SPROUT.
    clear();
    C(8, 6) = SEED; C(8, 7) = WALL; C(9, 6) = WATER;
    bool grew = false;
    for (uint32_t f = 0; f < 400 && !grew; ++f) { run(f); if (C(8, 6) == SPROUT) grew = true; }
    if (!grew) { printf("FAIL: grounded watered seed never germinated\n"); ++fails; }
    else printf("ok: grounded SEED beside WATER germinates into SPROUT\n");

    // 2. SEED in mid-air (nothing below) beside water does NOT germinate.
    clear();
    C(8, 6) = SEED; C(9, 6) = WATER;   // (8,7) is EMPTY -> unsupported
    bool sprouted = false;
    for (uint32_t f = 0; f < 400; ++f) { run(f); if (C(8, 6) == SPROUT) sprouted = true; }
    if (sprouted) { printf("FAIL: mid-air seed germinated (should need ground)\n"); ++fails; }
    else printf("ok: mid-air SEED beside WATER stays dormant\n");

    // 3. SEED on the ground with NO water stays inert.
    clear();
    C(8, 6) = SEED; C(8, 7) = WALL; C(9, 6) = SAND;
    for (uint32_t f = 0; f < 400; ++f) run(f);
    if (C(8, 6) != SEED) { printf("FAIL: dry grounded seed changed\n"); ++fails; }
    else printf("ok: dry grounded SEED stays inert\n");

    // 4. Determinism.
    clear(); C(8,6)=SEED; C(8,7)=WALL; C(9,6)=WATER; for(uint32_t f=0;f<50;++f) run(f); auto a=g;
    clear(); C(8,6)=SEED; C(8,7)=WALL; C(9,6)=WATER; for(uint32_t f=0;f<50;++f) run(f);
    if (g != a) { printf("FAIL: non-deterministic\n"); ++fails; }
    else printf("ok: deterministic\n");

    // 5. Movement: SEED falls and rests exactly like SAND (a heavy powder).
    const int W = 80, H = 80;
    std::vector<uint8_t> gg, mv;
    auto restRow = [&](uint8_t mat) {
        gg.assign(W*H, EMPTY); gg[(size_t)5*W+40] = mat;
        for (int it=0; it<H+8; ++it) { mv.assign(W*H,0); worldStepSSE(gg.data(), mv.data(), W, 1, W-1, 1, H-1, 0); }
        int yy=-1; for(int y=0;y<H;++y) for(int x=0;x<W;++x) if(gg[(size_t)y*W+x]==mat) yy=y; return yy;
    };
    int se = restRow(SEED), sa = restRow(SAND);
    if (se != sa) { printf("FAIL: SEED rests at y=%d but SAND at y=%d\n", se, sa); ++fails; }
    else printf("ok: SEED falls and rests exactly like SAND (y=%d)\n", se);

    printf(fails ? "\n%d FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
