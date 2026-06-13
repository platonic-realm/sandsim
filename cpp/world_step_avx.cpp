// AVX2 step (32 lanes). Compiled with -mavx2. Only called when the CPU supports
// AVX2 (see selectStep()), so its AVX2 instructions never run on an SSE-only CPU.
#include "simd_core.h"
#include "world_step.h"

extern "C" void worldStepAVX(uint8_t* grid, uint8_t* moved, int SW,
                             int X0, int X1, int Y0, int Y1, uint32_t frame) {
    simdStep<AvxOps>(grid, moved, SW, X0, X1, Y0, Y1, frame);
}
