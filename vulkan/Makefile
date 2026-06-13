GLSLC    ?= glslc
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
CXXFLAGS += $(shell pkg-config --cflags sdl2 2>/dev/null)
LDLIBS   := -lvulkan $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)

all: shaders/sand.comp.spv sandsim_vulkan_compute

# Compile the GLSL compute shader to SPIR-V (sand.comp -> sand.comp.spv).
shaders/%.spv: shaders/%
	$(GLSLC) $< -o $@

sandsim_vulkan_compute: sandsim_vulkan_compute.cpp shaders/sand.comp.spv
	$(CXX) $(CXXFLAGS) sandsim_vulkan_compute.cpp -o $@ $(LDLIBS)

.PHONY: all shaders run clean
shaders: shaders/sand.comp.spv

run: sandsim_vulkan_compute
	./sandsim_vulkan_compute

clean:
	rm -f sandsim_vulkan_compute shaders/sand.comp.spv
