CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall
CXXFLAGS += $(shell pkg-config --cflags sdl2 2>/dev/null)
LDLIBS   := $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)

SCALAR := sandsim_scalar_sb sandsim_scalar_mb
SSE    := sandsim_sse_sb sandsim_sse_mb
AVX    := sandsim_avx_sb sandsim_avx_mb

# Default builds the x86-friendly variants (NEON is ARM-only; see `make neon`).
all: $(SCALAR) $(SSE) $(AVX) sandsim_materials sandsim_world sandsim_world_simd

sandsim_materials: sandsim_materials.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

sandsim_world: sandsim_world.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

sandsim_world_simd: sandsim_world_simd.cpp
	$(CXX) $(CXXFLAGS) -msse4.1 $< -o $@ $(LDLIBS)

sandsim_scalar_%: sandsim_scalar_%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
sandsim_sse_%: sandsim_sse_%.cpp
	$(CXX) $(CXXFLAGS) -msse4.1 $< -o $@ $(LDLIBS)
sandsim_avx_%: sandsim_avx_%.cpp
	$(CXX) $(CXXFLAGS) -mavx2 $< -o $@ $(LDLIBS)

# NEON multi-buffer: only on ARM (armv8-a+simd).
neon: sandsim_neon_mb.cpp
	$(CXX) $(CXXFLAGS) -march=armv8-a+simd sandsim_neon_mb.cpp -o sandsim_neon_mb $(LDLIBS)

.PHONY: all neon bench clean
# scalar_sb is the golden reference for the cross-language --bench checksum.
bench: sandsim_scalar_sb
	./sandsim_scalar_sb --bench 1000 400 300

clean:
	rm -f $(SCALAR) $(SSE) $(AVX) sandsim_neon_mb sandsim_materials sandsim_world sandsim_world_simd
