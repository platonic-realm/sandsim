// One materials step over the padded grid interior. Two implementations live in
// separate TUs compiled with -msse4.1 and -mavx2; the host picks the widest the
// running CPU supports at startup. Both compute the same result (the rule is
// width-independent), so the choice is purely performance.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

using StepFn = void (*)(uint8_t* grid, uint8_t* moved, int SW,
                        int X0, int X1, int Y0, int Y1, uint32_t frame);

extern "C" void worldStepSSE(uint8_t* grid, uint8_t* moved, int SW,
                             int X0, int X1, int Y0, int Y1, uint32_t frame);
extern "C" void worldStepAVX(uint8_t* grid, uint8_t* moved, int SW,
                             int X0, int X1, int Y0, int Y1, uint32_t frame);

// SANDSIM_SIMD=sse|avx forces a path (for testing); otherwise pick the widest
// the running CPU supports (AVX2 -> 32 lanes, else SSE -> 16). Both compute the
// same result, so this is purely a performance / verification knob.
inline bool wantAVX() {
    const char* e = std::getenv("SANDSIM_SIMD");
    if (e && std::strcmp(e, "sse") == 0) return false;
    if (e && std::strcmp(e, "avx") == 0) return true;
    return __builtin_cpu_supports("avx2");
}
inline StepFn selectStep() { return wantAVX() ? worldStepAVX : worldStepSSE; }
inline const char* simdName() { return wantAVX() ? "avx2" : "sse4.1"; }
