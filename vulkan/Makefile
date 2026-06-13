GLSLC    ?= glslc
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
CXXFLAGS += $(shell pkg-config --cflags sdl2 2>/dev/null)
LDLIBS   := -lvulkan $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)

all: sandsim_world_vk

# Compile the GLSL compute shader to SPIR-V (world.comp -> world.comp.spv).
shaders/world.comp.spv: shaders/world.comp
	$(GLSLC) $< -o $@

sandsim_world_vk: sandsim_world_vk.cpp shaders/world.comp.spv ../worldgen.h ../ui.h
	$(CXX) $(CXXFLAGS) sandsim_world_vk.cpp -o $@ $(LDLIBS)

.PHONY: all shaders run bench clean
shaders: shaders/world.comp.spv

run: sandsim_world_vk
	./sandsim_world_vk

bench: sandsim_world_vk
	./sandsim_world_vk --bench 600 6 6

clean:
	rm -f sandsim_world_vk shaders/world.comp.spv
