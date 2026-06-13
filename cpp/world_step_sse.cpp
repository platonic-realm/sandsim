// SSE4.1 step (16 lanes). Compiled with -msse4.1.
#include "simd_core.h"
#include "world_step.h"

extern "C" void worldStepSSE(uint8_t* grid, uint8_t* moved, int SW,
                             int X0, int X1, int Y0, int Y1, uint32_t frame) {
    simdStep<SseOps>(grid, moved, SW, X0, X1, Y0, Y1, frame);
}
