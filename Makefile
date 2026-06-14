# sandsim - one engine (the multi-material streaming world), one implementation
# per platform: C++ (SIMD, runtime SSE/AVX2 dispatch), OpenGL, and Vulkan.

.PHONY: all cpp opengl vulkan benchmark clean help

help:
	@echo "sandsim targets:"
	@echo "  make all        build cpp + opengl + vulkan"
	@echo "  make cpp        build the C++ SIMD world (auto SSE/AVX2)"
	@echo "  make opengl     build the OpenGL world"
	@echo "  make vulkan     build the Vulkan world"
	@echo "  make benchmark  build all three, verify identical output, print a table"
	@echo "  make clean      remove build artifacts"

all: cpp opengl vulkan

cpp:
	$(MAKE) -C cpp
opengl:
	$(MAKE) -C opengl
vulkan:
	$(MAKE) -C vulkan

benchmark:
	@bash tools/benchmark.sh

clean:
	-$(MAKE) -C cpp clean
	-$(MAKE) -C opengl clean
	-$(MAKE) -C vulkan clean
