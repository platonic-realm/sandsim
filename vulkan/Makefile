GLSLC    ?= glslc
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
CXXFLAGS += $(shell pkg-config --cflags sdl2 2>/dev/null)
LDLIBS   := -lvulkan $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)

SHADERS := shaders/sand.comp.spv shaders/sand_mb.comp.spv
BINS    := sandsim_vulkan_compute sandsim_vulkan_compute_mb

all: $(SHADERS) $(BINS)

# Compile each GLSL compute shader to SPIR-V (sand.comp -> sand.comp.spv).
shaders/%.spv: shaders/%
	$(GLSLC) $< -o $@

sandsim_vulkan_compute: sandsim_vulkan_compute.cpp shaders/sand.comp.spv
	$(CXX) $(CXXFLAGS) sandsim_vulkan_compute.cpp -o $@ $(LDLIBS)

sandsim_vulkan_compute_mb: sandsim_vulkan_compute_mb.cpp shaders/sand_mb.comp.spv
	$(CXX) $(CXXFLAGS) sandsim_vulkan_compute_mb.cpp -o $@ $(LDLIBS)

.PHONY: all shaders run run-mb clean
shaders: $(SHADERS)

run: sandsim_vulkan_compute
	./sandsim_vulkan_compute

run-mb: sandsim_vulkan_compute_mb
	./sandsim_vulkan_compute_mb

clean:
	rm -f $(BINS) $(SHADERS)
