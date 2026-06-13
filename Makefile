# sandsim - root build orchestrator.
#
# Per-implementation targets plus an `all` that builds every implementation
# whose toolchain is present (and cleanly skips the rest), and a `bench` target
# that runs the cross-implementation benchmark harness.

ZIG_BUILD = zig build-exe sandsim.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim

.PHONY: all help c rust zig cpp opengl hip cuda vulkan bench clean

help:
	@echo "sandsim build targets:"
	@echo "  make all      build every implementation whose toolchain is installed"
	@echo "  make bench    build the benchmarkable implementations and print a table"
	@echo "  make <impl>   build one of: c rust zig cpp opengl hip cuda vulkan"
	@echo "  make clean    remove all build artifacts"

# --- individual implementations ---------------------------------------------
c:
	$(MAKE) -C c
rust:
	cd rust && cargo build --release --offline
zig:
	cd zig && $(ZIG_BUILD)
cpp:
	$(MAKE) -C cpp
opengl:
	$(MAKE) -C opengl
hip:
	$(MAKE) -C hip
cuda:
	$(MAKE) -C cuda
vulkan:
	$(MAKE) -C vulkan

# --- build everything available ---------------------------------------------
all:
	@echo "==> building every implementation whose toolchain is present"
	@if command -v cc    >/dev/null 2>&1; then $(MAKE) -C c;   else echo "skip c      (no cc)";            fi
	@if command -v cargo >/dev/null 2>&1; then cd rust && cargo build --release --offline; else echo "skip rust   (no cargo)"; fi
	@if command -v zig   >/dev/null 2>&1; then cd zig && $(ZIG_BUILD); else echo "skip zig    (no zig)";   fi
	@if command -v g++   >/dev/null 2>&1; then $(MAKE) -C cpp; else echo "skip cpp    (no g++)";           fi
	@if pkg-config --exists glfw3 glew gl 2>/dev/null; then $(MAKE) -C opengl; else echo "skip opengl (no glfw3/glew/gl)"; fi
	@if command -v glslc >/dev/null 2>&1; then $(MAKE) -C vulkan; else echo "skip vulkan (no glslc)";      fi
	@if command -v hipcc >/dev/null 2>&1; then $(MAKE) -C hip;  else echo "skip hip    (no hipcc)";        fi
	@if command -v nvcc  >/dev/null 2>&1; then $(MAKE) -C cuda; else echo "skip cuda   (no nvcc)";         fi
	@command -v mojo >/dev/null 2>&1 || echo "skip mojo   (no mojo; source-only)"

# --- benchmark --------------------------------------------------------------
bench:
	@bash tools/bench.sh

# --- cleanup ----------------------------------------------------------------
clean:
	-$(MAKE) -C c clean
	-cd rust && cargo clean
	-rm -f zig/sandsim
	-$(MAKE) -C cpp clean
	-$(MAKE) -C opengl clean
	-$(MAKE) -C hip clean
	-$(MAKE) -C cuda clean
	-$(MAKE) -C vulkan clean
