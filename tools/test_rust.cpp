// Unit test for RUST: iron corrodes in water/acid; rust smelts back near fire/lava.
#include "../cpp/world_step.h"
#include "../cpp/materials.h"
#include <cstdio>
#include <vector>

static const int SWr = 16, SHr = 12;
static std::vector<uint8_t> g, s;
static uint8_t& C(int x, int y) { return g[(size_t)y * SWr + x]; }
static void run(uint32_t f) { rustCycle(g.data(), s.data(), SWr, 1, SWr - 1, 1, SHr - 1, f); }
static void clear() { g.assign(SWr * SHr, EMPTY); s.assign(SWr * SHr, 0); }
static int count(uint8_t m) { int n = 0; for (auto v : g) if (v == m) ++n; return n; }

int main() {
    int fails = 0;

    // 1. IRON beside WATER eventually corrodes to RUST.
    clear();
    C(8, 6) = IRON; C(9, 6) = WATER;
    bool rusted = false;
    for (uint32_t f = 0; f < 400 && !rusted; ++f) { run(f); if (C(8, 6) == RUST) rusted = true; }
    if (!rusted) { printf("FAIL: iron in water never rusted\n"); ++fails; }
    else printf("ok: IRON beside WATER corrodes to RUST\n");

    // 1b. ...ACID corrodes it too.
    clear();
    C(8, 6) = IRON; C(9, 6) = ACID;
    rusted = false;
    for (uint32_t f = 0; f < 400 && !rusted; ++f) { run(f); if (C(8, 6) == RUST) rusted = true; }
    if (!rusted) { printf("FAIL: iron in acid never rusted\n"); ++fails; }
    else printf("ok: IRON beside ACID corrodes to RUST\n");

    // 2. RUST beside FIRE/LAVA smelts back to IRON.
    clear();
    C(8, 6) = RUST; C(9, 6) = LAVA;
    bool smelted = false;
    for (uint32_t f = 0; f < 400 && !smelted; ++f) { run(f); if (C(8, 6) == IRON) smelted = true; }
    if (!smelted) { printf("FAIL: rust beside lava never smelted back to iron\n"); ++fails; }
    else printf("ok: RUST beside LAVA smelts back to IRON\n");

    // 3. Dry iron (no water/acid) and cold rust (no fire/lava) are inert.
    clear();
    C(8, 6) = IRON; C(9, 6) = SAND; C(4, 6) = RUST; C(5, 6) = SAND;
    for (uint32_t f = 0; f < 100; ++f) run(f);
    if (C(8, 6) != IRON || C(4, 6) != RUST) { printf("FAIL: dry iron / cold rust changed\n"); ++fails; }
    else printf("ok: dry IRON and cold RUST stay inert\n");

    // 4. Determinism.
    clear(); C(8,6)=IRON; C(9,6)=WATER; for(uint32_t f=0;f<50;++f) run(f); auto a=g;
    clear(); C(8,6)=IRON; C(9,6)=WATER; for(uint32_t f=0;f<50;++f) run(f);
    if (g != a) { printf("FAIL: non-deterministic\n"); ++fails; }
    else printf("ok: deterministic\n");

    // 5. Movement: RUST falls and rests exactly like SAND (a heavy powder).
    const int W = 80, H = 80;
    std::vector<uint8_t> gg, mv;
    auto restRow = [&](uint8_t mat) {
        gg.assign(W*H, EMPTY); gg[(size_t)5*W+40] = mat;
        for (int it=0; it<H+8; ++it) { mv.assign(W*H,0); worldStepSSE(gg.data(), mv.data(), W, 1, W-1, 1, H-1, 0); }
        int yy=-1; for(int y=0;y<H;++y) for(int x=0;x<W;++x) if(gg[(size_t)y*W+x]==mat) yy=y; return yy;
    };
    int ru = restRow(RUST), sa = restRow(SAND);
    if (ru != sa) { printf("FAIL: RUST rests at y=%d but SAND at y=%d\n", ru, sa); ++fails; }
    else printf("ok: RUST falls and rests exactly like SAND (y=%d)\n", ru);

    printf(fails ? "\n%d FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
