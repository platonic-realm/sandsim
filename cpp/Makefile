CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall
CXXFLAGS += $(shell pkg-config --cflags sdl2 2>/dev/null)
LDLIBS   := $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)

# One binary that picks the widest SIMD the CPU supports at runtime. The two SIMD
# step variants are compiled in their own objects (-msse4.1 / -mavx2) and the
# baseline host object dispatches between them.
all: sandsim_world

sandsim_world: sandsim_world.o world_step_sse.o world_step_avx.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

sandsim_world.o: sandsim_world.cpp materials.h world_step.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
world_step_sse.o: world_step_sse.cpp simd_core.h world_step.h materials.h
	$(CXX) $(CXXFLAGS) -msse4.1 -c $< -o $@
world_step_avx.o: world_step_avx.cpp simd_core.h world_step.h materials.h
	$(CXX) $(CXXFLAGS) -mavx2 -c $< -o $@

.PHONY: all clean
clean:
	rm -f sandsim_world *.o
