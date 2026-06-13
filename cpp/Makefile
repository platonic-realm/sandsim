CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall
CXXFLAGS += $(shell pkg-config --cflags sdl2 2>/dev/null)
LDLIBS   := $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)

# Three physics engines (sand / materials / world), each in three SIMD tiers
# (scalar / SSE / AVX2). The SIMD materials and world share simd_core.h.
SAND      := sandsim_scalar_sb sandsim_sse_sb sandsim_avx_sb
MATERIALS := sandsim_materials sandsim_materials_sse sandsim_materials_avx
WORLD     := sandsim_world sandsim_world_sse sandsim_world_avx

all: $(SAND) $(MATERIALS) $(WORLD)

# --- sand (single material) -------------------------------------------------
sandsim_scalar_sb: sandsim_scalar_sb.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
sandsim_sse_sb: sandsim_sse_sb.cpp
	$(CXX) $(CXXFLAGS) -msse4.1 $< -o $@ $(LDLIBS)
sandsim_avx_sb: sandsim_avx_sb.cpp
	$(CXX) $(CXXFLAGS) -mavx2 $< -o $@ $(LDLIBS)

# --- materials (multi-material, fixed grid) ---------------------------------
sandsim_materials: sandsim_materials.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
sandsim_materials_sse: sandsim_materials_sse.cpp simd_core.h
	$(CXX) $(CXXFLAGS) -msse4.1 $< -o $@ $(LDLIBS)
sandsim_materials_avx: sandsim_materials_avx.cpp simd_core.h
	$(CXX) $(CXXFLAGS) -mavx2 $< -o $@ $(LDLIBS)

# --- world (chunked, disk-streamed) -----------------------------------------
sandsim_world: sandsim_world.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
sandsim_world_sse: sandsim_world_sse.cpp simd_core.h
	$(CXX) $(CXXFLAGS) -msse4.1 $< -o $@ $(LDLIBS)
sandsim_world_avx: sandsim_world_avx.cpp simd_core.h
	$(CXX) $(CXXFLAGS) -mavx2 $< -o $@ $(LDLIBS)

.PHONY: all bench clean
# scalar_sb is the golden reference for the cross-language --bench checksum.
bench: sandsim_scalar_sb
	./sandsim_scalar_sb --bench 1000 400 300

clean:
	rm -f $(SAND) $(MATERIALS) $(WORLD)
