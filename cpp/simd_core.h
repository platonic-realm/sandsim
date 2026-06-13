// Shared multi-material single-grid SIMD update, parameterised over the vector
// width (SSE = 16 lanes, AVX2 = 32 lanes). Used by the SSE/AVX materials and
// world variants so the one tricky piece of code lives in one place.
//
// The grid is one contiguous, padded (WALL-bordered) array of material ids. The
// lanes are adjacent cells of that grid, so material flows freely (connected).
// Each directional move maps a source column x to a distinct target column
// x+dx, which makes a whole directional pass conflict-free; horizontal (same
// row) swaps chain lane-to-lane, so they are split into even/odd column phases.
// A per-cell `moved` mask gives one-move-per-frame priority.
//
// AVX note: _mm256_s{l,r}li_si256 shift within each 128-bit half, so the
// horizontal cross-lane shift would lose material across the 128-lane boundary.
// We avoid that by skipping the group-boundary lanes (0 and 16 for left, 15 and
// 31 for right) — those cells just don't move that frame (conserving). SSE and
// AVX therefore skip the same columns and produce identical results.
#pragma once
#include <emmintrin.h>
#include <smmintrin.h>   // SSE4.1
#include <immintrin.h>   // AVX2
#include <cstdint>
#include <cstring>

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };

enum SimdGroup { SG_DOWN, SG_GAS, SG_HORIZ };

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
    static V shr1(V a) { return _mm256_srli_si256(a, 1); }  // per-128 lane
    static V shl1(V a) { return _mm256_slli_si256(a, 1); }
    static V even() { return _mm256_set_epi8(0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,
                                             0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1); }
    static V odd()  { return _mm256_set_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,
                                             -1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0); }
    // zero at bytes 0 and 16 (the two 128-lane group starts)
    static V notStart() { return _mm256_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,
                                                  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0); }
    // zero at bytes 15 and 31 (the two 128-lane group ends)
    static V notEnd()   { return _mm256_set_epi8(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                                                  0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1); }
};
#endif  // __AVX2__

// One materials step over the padded grid interior [X0,X1) x [Y0,Y1).
template <class Ops>
inline void simdStep(uint8_t* grid, uint8_t* moved, int SW,
                     int X0, int X1, int Y0, int Y1, uint32_t frame) {
    using V = typename Ops::V;
    const int W = Ops::W;
    const V vE = Ops::zero(), vS = Ops::set1(SAND), vW = Ops::set1(WATER), vG = Ops::set1(GAS);

    auto mask = [&](V cur, V tgt, V mvCur, V mvTgt, int grp) -> V {
        V isSand = Ops::eq(cur, vS), isWater = Ops::eq(cur, vW), isGas = Ops::eq(cur, vG);
        V tEm = Ops::eq(tgt, vE), tWa = Ops::eq(tgt, vW), tGa = Ops::eq(tgt, vG);
        V canEnter = Ops::Or(
            Ops::Or(Ops::And(isSand, Ops::Or(Ops::Or(tEm, tWa), tGa)),
                    Ops::And(isWater, Ops::Or(tEm, tGa))),
            Ops::And(isGas, tEm));
        V eligible = (grp == SG_DOWN) ? Ops::Or(isSand, isWater)
                   : (grp == SG_GAS)  ? isGas
                                      : Ops::Or(isWater, isGas);
        V notC = Ops::eq(mvCur, vE), notT = Ops::eq(mvTgt, vE);
        return Ops::And(Ops::And(eligible, canEnter), Ops::And(notC, notT));
    };

    auto vertPass = [&](int dx, int dy, int grp, bool topDown) {
        int yS = topDown ? Y0 : (Y1 - 1), yE = topDown ? Y1 : (Y0 - 1), yStep = topDown ? 1 : -1;
        for (int y = yS; y != yE; y += yStep)
            for (int x = X0; x < X1; x += W) {
                uint8_t* gs = &grid[(size_t)y * SW + x];
                uint8_t* gt = &grid[(size_t)(y + dy) * SW + (x + dx)];
                uint8_t* ms = &moved[(size_t)y * SW + x];
                uint8_t* mt = &moved[(size_t)(y + dy) * SW + (x + dx)];
                V cur = Ops::loadu(gs), tgt = Ops::loadu(gt), mc = Ops::loadu(ms), mtt = Ops::loadu(mt);
                V m = mask(cur, tgt, mc, mtt, grp);
                Ops::storeu(gt, Ops::blend(tgt, cur, m));
                Ops::storeu(gs, Ops::blend(cur, tgt, m));
                Ops::storeu(mt, Ops::Or(mtt, m));
                Ops::storeu(ms, Ops::Or(mc, m));
            }
    };

    auto horizPhase = [&](int dx, bool evenPhase, int grp) {
        V laneMask = evenPhase ? Ops::even() : Ops::odd();
        if (dx < 0 && evenPhase)  laneMask = Ops::And(laneMask, Ops::notStart());
        if (dx > 0 && !evenPhase) laneMask = Ops::And(laneMask, Ops::notEnd());
        for (int y = Y0; y < Y1; ++y)
            for (int x = X0; x < X1; x += W) {
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
            }
    };

    std::memset(moved + (size_t)Y0 * SW, 0, (size_t)(Y1 - Y0) * SW);
    bool flip = frame & 1;
    int dA = flip ? -1 : 1, dB = -dA;
    vertPass(0,  1, SG_DOWN, false);
    vertPass(dA, 1, SG_DOWN, false);
    vertPass(dB, 1, SG_DOWN, false);
    vertPass(0, -1, SG_GAS, true);
    vertPass(dA, -1, SG_GAS, true);
    vertPass(dB, -1, SG_GAS, true);
    horizPhase(dA, true,  SG_HORIZ); horizPhase(dA, false, SG_HORIZ);
    horizPhase(dB, true,  SG_HORIZ); horizPhase(dB, false, SG_HORIZ);
}
