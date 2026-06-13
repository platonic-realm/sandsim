// Material ids shared by the baseline host code and the SIMD step TUs.
#pragma once
#include <cstdint>
enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };
