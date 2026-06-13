// Material ids shared by the baseline host code and the SIMD step TUs.
#pragma once
#include <cstdint>
// Density order (heavy -> light): SAND > WATER > OIL > air(EMPTY) > GAS.
// OIL falls through air/gas but floats on water.
enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, MATERIAL_COUNT = 6 };
