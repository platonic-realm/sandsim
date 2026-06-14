// Shared multi-material falling-sand update, parameterised over the SIMD width
// (SSE = 16 lanes, AVX2 = 32 lanes). The same rule is implemented by the GPU
// compute shaders so all backends produce a bit-identical world.
//
// The rule is ORDER-INDEPENDENT: each frame is a fixed sequence of sub-passes,
// and within a sub-pass every move is between a DISJOINT pair of cells, so the
// result is a pure function of the previous frame (no scan-order dependence).
// That is what lets a massively parallel GPU reproduce the CPU result exactly.
//
// Disjointness is achieved by partitioning each direction:
//   * vertical moves (dx=0) split by ROW parity  -> dominoes (y, y+dy) don't overlap
//   * diagonal moves split by COLUMN parity       -> source/target columns disjoint
//   * horizontal moves split by COLUMN parity      (same-row swap, see horizBlock)
// A per-cell `moved` mask (reset each frame) gives one-move-per-frame priority
// across the passes: a cell that fell can't also slide the same frame.
//
// Materials: EMPTY/WALL/SAND/WATER/GAS/OIL/FIRE, density swaps
//   (OIL floats on water; FIRE rises like flame. FIRE's burn-out is a separate
//   per-cell time-varying pass in materials.h, kept out of this movement step.)
#pragma once
#include "materials.h"
#include <emmintrin.h>
#include <smmintrin.h>   // SSE4.1
#include <immintrin.h>   // AVX2
#include <cstdint>
#include <cstring>

enum SimdGroup { SG_DOWN, SG_GAS, SG_HORIZ };  // which materials move in a pass

struct SseOps {
    using V = __m128i;
    static constexpr int W = 16;
    static V loadu(const uint8_t* p) { return _mm_loadu_si128((const V*)p); }
    static void storeu(uint8_t* p, V v) { _mm_storeu_si128((V*)p, v); }
    static V set1(int b) { return _mm_set1_epi8((char)b); }
    static V zero() { return _mm_setzero_si128(); }
    static V eq(V a, V b) { return _mm_cmpeq_epi8(a, b); }
    static V And(V a, V b) { return _mm_and_si128(a, b); }
    static V Or(V a, V b) { return _mm_or_si128(a, b); }
    static V blend(V a, V b, V m) { return _mm_blendv_epi8(a, b, m); }
    static V shr1(V a) { return _mm_srli_si128(a, 1); }
    static V shl1(V a) { return _mm_slli_si128(a, 1); }
    static V ones() { return _mm_set1_epi8((char)0xFF); }
    static V even() { return _mm_set_epi8(0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1); }
    static V odd()  { return _mm_set_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0); }
    static V notStart() { return _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0); }
    static V notEnd()   { return _mm_set_epi8(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1); }
};

#ifdef __AVX2__
struct AvxOps {
    using V = __m256i;
    static constexpr int W = 32;
    static V loadu(const uint8_t* p) { return _mm256_loadu_si256((const V*)p); }
    static void storeu(uint8_t* p, V v) { _mm256_storeu_si256((V*)p, v); }
    static V set1(int b) { return _mm256_set1_epi8((char)b); }
    static V zero() { return _mm256_setzero_si256(); }
    static V eq(V a, V b) { return _mm256_cmpeq_epi8(a, b); }
    static V And(V a, V b) { return _mm256_and_si256(a, b); }
    static V Or(V a, V b) { return _mm256_or_si256(a, b); }
    static V blend(V a, V b, V m) { return _mm256_blendv_epi8(a, b, m); }
    static V shr1(V a) { return _mm256_srli_si256(a, 1); }   // per-128 lane
    static V shl1(V a) { return _mm256_slli_si256(a, 1); }
    static V ones() { return _mm256_set1_epi8((char)0xFF); }
    static V even() { return _mm256_set_epi8(0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,
                                             0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1); }
    static V odd()  { return _mm256_set_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,
                                             -1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0); }
    static V notStart() { return _mm256_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,
                                                  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0); }
    static V notEnd()   { return _mm256_set_epi8(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                                                  0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1); }
};
#endif

template <class Ops>
inline void simdStep(uint8_t* grid, uint8_t* moved, int SW,
                     int X0, int X1, int Y0, int Y1, uint32_t /*frame*/) {
    using V = typename Ops::V;
    const int W = Ops::W;
    const V vE = Ops::zero(), vS = Ops::set1(SAND), vW = Ops::set1(WATER),
            vG = Ops::set1(GAS), vO = Ops::set1(OIL), vF = Ops::set1(FIRE), vL = Ops::set1(LAVA),
            vSt = Ops::set1(STEAM), vA = Ops::set1(ACID), vSm = Ops::set1(SMOKE), vAsh = Ops::set1(ASH),
            vSnow = Ops::set1(SNOW), vMerc = Ops::set1(MERCURY), vGun = Ops::set1(GUNPOWDER),
            vTh = Ops::set1(THERMITE);

    // move mask: which lanes hold an eligible mover whose target is enterable and
    // where neither cell has already moved this frame. Density SAND>LAVA>ACID>
    // WATER>OIL>air>GAS>FIRE/STEAM: a mover can swap into anything strictly lighter.
    auto mask = [&](V cur, V tgt, V mc, V mt, int grp) -> V {
        V isS = Ops::eq(cur, vS), isW = Ops::eq(cur, vW), isG = Ops::eq(cur, vG),
          isO = Ops::eq(cur, vO), isF = Ops::eq(cur, vF), isL = Ops::eq(cur, vL),
          isSt = Ops::eq(cur, vSt), isA = Ops::eq(cur, vA), isSm = Ops::eq(cur, vSm),
          isAsh = Ops::eq(cur, vAsh), isSnow = Ops::eq(cur, vSnow), isMerc = Ops::eq(cur, vMerc),
          isGun = Ops::eq(cur, vGun), isTh = Ops::eq(cur, vTh);
        V tE = Ops::eq(tgt, vE), tW = Ops::eq(tgt, vW), tG = Ops::eq(tgt, vG),
          tO = Ops::eq(tgt, vO), tF = Ops::eq(tgt, vF), tL = Ops::eq(tgt, vL),
          tSt = Ops::eq(tgt, vSt), tA = Ops::eq(tgt, vA), tSm = Ops::eq(tgt, vSm),
          tSnow = Ops::eq(tgt, vSnow), tS = Ops::eq(tgt, vS);
        V lightGas   = Ops::Or(Ops::Or(tG, tF), Ops::Or(tSt, Ops::Or(tSm, tE)));            // G,F,STEAM,SMOKE,E
        V lightSnow  = Ops::Or(lightGas, tSnow);                                            // SNOW + light (SNOW is lightest liquid-tier)
        V belowAcid  = Ops::Or(Ops::Or(tW, tO), lightSnow);                                 // W,O,SNOW,+light
        V belowSand  = Ops::Or(Ops::Or(tL, tA), belowAcid);                                 // L,A,W,O,SNOW,+light
        V belowLava  = Ops::Or(tA, belowAcid);                                              // A,W,O,SNOW,+light
        V belowWater = Ops::Or(tO, lightSnow);                                              // O,SNOW,+light
        V belowOil   = lightSnow;                                                           // SNOW,G,F,STEAM,SMOKE,E
        V belowSnow  = lightGas;                                                            // SNOW floats on liquids: enters only G,F,St,Sm,E
        V belowMerc  = Ops::Or(tS, belowSand);                                              // MERCURY is heaviest: sinks through SAND + everything
        V rise = Ops::Or(isG, Ops::Or(isF, Ops::Or(isSt, isSm)));                           // gas/fire/steam/smoke
        V can = Ops::Or(Ops::Or(Ops::Or(Ops::Or(
            Ops::Or(Ops::And(Ops::Or(Ops::Or(Ops::Or(isS, isAsh), isGun), isTh), belowSand), Ops::And(isL, belowLava)),  // SAND/ASH/GUNPOWDER/THERMITE: heavy powders
            Ops::Or(Ops::And(isA, belowAcid), Ops::And(isW, belowWater))),
            Ops::Or(Ops::And(isO, belowOil), Ops::And(rise, tE))),
            Ops::And(isSnow, belowSnow)),                                                  // SNOW: light powder
            Ops::And(isMerc, belowMerc));                                                  // MERCURY: heaviest liquid
        V elig = (grp == SG_DOWN) ? Ops::Or(Ops::Or(Ops::Or(Ops::Or(Ops::Or(Ops::Or(Ops::Or(isS, isAsh), isGun), isTh), isSnow), isMerc), isL), Ops::Or(isA, Ops::Or(isW, isO)))
               : (grp == SG_GAS)  ? rise
                                  : Ops::Or(Ops::Or(Ops::Or(isL, isMerc), isA), Ops::Or(Ops::Or(isW, isO), rise));
        return Ops::And(Ops::And(elig, can),
                        Ops::And(Ops::eq(mc, vE), Ops::eq(mt, vE)));
    };

    // Swap source (y,x..) with target (y+dy, x+dx..) for dy != 0 (different rows,
    // so no same-row overlap). laneMask selects which columns may move.
    auto swapBlock = [&](int y, int x, int dx, int dy, int grp, V laneMask) {
        uint8_t* gs = &grid[(size_t)y * SW + x];
        uint8_t* gt = &grid[(size_t)(y + dy) * SW + (x + dx)];
        uint8_t* ms = &moved[(size_t)y * SW + x];
        uint8_t* mt = &moved[(size_t)(y + dy) * SW + (x + dx)];
        V cur = Ops::loadu(gs), tgt = Ops::loadu(gt), mc = Ops::loadu(ms), mtt = Ops::loadu(mt);
        V m = Ops::And(mask(cur, tgt, mc, mtt, grp), laneMask);
        Ops::storeu(gt, Ops::blend(tgt, cur, m));
        Ops::storeu(gs, Ops::blend(cur, tgt, m));
        Ops::storeu(mt, Ops::Or(mtt, m));
        Ops::storeu(ms, Ops::Or(mc, m));
    };

    // Horizontal swap (dy=0, same row): the target lands inside the source
    // register, so use a 1-lane shift; skip the group-boundary lane so the
    // per-128 AVX shift stays conserving.
    auto horizBlock = [&](int y, int x, int dx, int grp, V laneMask) {
        uint8_t* gs = &grid[(size_t)y * SW + x];
        uint8_t* gn = &grid[(size_t)y * SW + (x + dx)];
        uint8_t* ms = &moved[(size_t)y * SW + x];
        uint8_t* mn = &moved[(size_t)y * SW + (x + dx)];
        V cur = Ops::loadu(gs), nbr = Ops::loadu(gn), mc = Ops::loadu(ms), mn_ = Ops::loadu(mn);
        V m = Ops::And(mask(cur, nbr, mc, mn_, grp), laneMask);
        V shCur = (dx < 0) ? Ops::shr1(cur) : Ops::shl1(cur);
        V shM   = (dx < 0) ? Ops::shr1(m)   : Ops::shl1(m);
        V out = Ops::blend(cur, nbr, m);
        out = Ops::blend(out, shCur, shM);
        Ops::storeu(gs, out);
        Ops::storeu(ms, Ops::Or(Ops::Or(mc, m), shM));
    };

    const V ALL = Ops::ones(), EVEN = Ops::even(), ODD = Ops::odd();
    // vertical pass: rows of one parity, full width
    auto vert = [&](int dy, int parity, int grp) {
        for (int y = Y0 + parity; y < Y1; y += 2)
            for (int x = X0; x < X1; x += W) swapBlock(y, x, 0, dy, grp, ALL);
    };
    // diagonal pass: all rows, one column parity
    auto diag = [&](int dx, int dy, bool evenCols, int grp) {
        V lm = evenCols ? EVEN : ODD;
        for (int y = Y0; y < Y1; ++y)
            for (int x = X0; x < X1; x += W) swapBlock(y, x, dx, dy, grp, lm);
    };
    // horizontal pass: all rows, one column parity
    auto horiz = [&](int dx, bool evenCols, int grp) {
        V lm = evenCols ? EVEN : ODD;
        if (dx < 0 && evenCols)  lm = Ops::And(lm, Ops::notStart());
        if (dx > 0 && !evenCols) lm = Ops::And(lm, Ops::notEnd());
        for (int y = Y0; y < Y1; ++y)
            for (int x = X0; x < X1; x += W) horizBlock(y, x, dx, grp, lm);
    };

    std::memset(moved + (size_t)Y0 * SW, 0, (size_t)(Y1 - Y0) * SW);

    vert(1, 0, SG_DOWN); vert(1, 1, SG_DOWN);                       // sand/water fall
    diag(-1, 1, true, SG_DOWN); diag(-1, 1, false, SG_DOWN);        // ... down-left
    diag( 1, 1, true, SG_DOWN); diag( 1, 1, false, SG_DOWN);        // ... down-right
    vert(-1, 0, SG_GAS); vert(-1, 1, SG_GAS);                       // gas rises
    diag(-1, -1, true, SG_GAS); diag(-1, -1, false, SG_GAS);        // ... up-left
    diag( 1, -1, true, SG_GAS); diag( 1, -1, false, SG_GAS);        // ... up-right
    horiz(-1, true, SG_HORIZ); horiz(-1, false, SG_HORIZ);          // water/gas spread left
    horiz( 1, true, SG_HORIZ); horiz( 1, false, SG_HORIZ);          // ... and right
}
