// Unit test for LASER + BEAM: a fixed emitter fires a rightward beam that extends
// through empty space, burns flammables to FIRE, and is stopped by solid walls.
#include "../cpp/materials.h"
#include <cstdio>
#include <vector>

static const int SWl = 24, SHl = 10;
static std::vector<uint8_t> g, s;
static uint8_t& C(int x, int y) { return g[(size_t)y * SWl + x]; }
static void clear() { g.assign(SWl * SHl, EMPTY); s.assign(SWl * SHl, 0); }
static void run(int n) { for (int k = 0; k < n; ++k) { s.assign(SWl * SHl, 0); laserBeam(g.data(), s.data(), SWl, 1, SWl - 1, 1, SHl - 1); } }

int main() {
    int fails = 0;

    // 1. LASER emits a BEAM that extends rightward through empty space.
    clear();
    C(2, 5) = LASER;
    run(8);
    bool ray = C(3,5)==BEAM && C(4,5)==BEAM && C(8,5)==BEAM;   // a continuous ray to the right
    if (!ray) { printf("FAIL: laser did not project a rightward beam (3=%d 4=%d 8=%d)\n", C(3,5),C(4,5),C(8,5)); ++fails; }
    else printf("ok: LASER projects a continuous rightward BEAM\n");

    // 2. The beam BURNS a flammable (WOOD) it reaches into FIRE.
    clear();
    C(2, 5) = LASER; C(8, 5) = WOOD;
    run(12);
    if (C(8, 5) != FIRE) { printf("FAIL: beam did not ignite the wood (got %d)\n", C(8,5)); ++fails; }
    else printf("ok: BEAM burns the WOOD it strikes into FIRE\n");

    // 3. The beam is STOPPED by a solid WALL (does not pass through).
    clear();
    C(2, 5) = LASER; C(8, 5) = WALL;
    run(12);
    if (C(7,5) != BEAM || C(8,5) != WALL || C(9,5) != EMPTY) {
        printf("FAIL: beam not stopped by wall (7=%d 8=%d 9=%d)\n", C(7,5),C(8,5),C(9,5)); ++fails;
    } else printf("ok: BEAM is stopped cold by a WALL\n");

    // 4. A BEAM with no emitter feeding it retracts to EMPTY.
    clear();
    C(5, 5) = BEAM;   // a lone beam, nothing to its left
    run(1);
    if (C(5, 5) != EMPTY) { printf("FAIL: unfed beam did not vanish (got %d)\n", C(5,5)); ++fails; }
    else printf("ok: an unfed BEAM retracts to EMPTY\n");

    // 5. Determinism.
    clear(); C(2,5)=LASER; C(14,5)=WOOD; run(20); auto a=g;
    clear(); C(2,5)=LASER; C(14,5)=WOOD; run(20);
    if (g != a) { printf("FAIL: non-deterministic\n"); ++fails; }
    else printf("ok: deterministic\n");

    printf(fails ? "\n%d FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
