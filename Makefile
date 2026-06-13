# sandsim - root build orchestrator.
#
# Per-implementation targets plus an `all` that builds every implementation
# whose toolchain is present (and cleanly skips the rest), and a `bench` target
# that runs the cross-implementation benchmark harness.

ZIG_BUILD = zig build-exe sandsim.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim
ZIG_BUILD_MAT = zig build-exe sandsim_materials.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim_materials
ZIG_BUILD_WORLD = zig build-exe sandsim_world.zig -lSDL2 -lc -I/usr/include -O ReleaseFast -femit-bin=sandsim_world

.PHONY: all help c rust zig cpp opengl hip cuda vulkan bench materials bench-materials world bench-world clean

help:
	@echo "sandsim build targets:"
	@echo "  make all              build every implementation whose toolchain is installed"
	@echo "  make bench            build the benchmarkable implementations and print a table"
	@echo "  make materials        build the multi-material (Noita-style) implementations"
	@echo "  make bench-materials  cross-check the materials engine across languages"
	@echo "  make world            build the chunked streaming-world implementations"
	@echo "  make bench-world      cross-check the streaming world across languages"
	@echo "  make <impl>           build one of: c rust zig cpp opengl hip cuda vulkan"
	@echo "  make clean            remove all build artifacts"

# --- individual implementations ---------------------------------------------
c:
	$(MAKE) -C c
rust:
	cd rust && cargo build --release --offline
zig:
	cd zig && $(ZIG_BUILD) && $(ZIG_BUILD_MAT) && $(ZIG_BUILD_WORLD)
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
	@if command -v zig   >/dev/null 2>&1; then cd zig && $(ZIG_BUILD) && $(ZIG_BUILD_MAT) && $(ZIG_BUILD_WORLD); else echo "skip zig    (no zig)";   fi
	@if command -v g++   >/dev/null 2>&1; then $(MAKE) -C cpp; else echo "skip cpp    (no g++)";           fi
	@if pkg-config --exists glfw3 glew gl 2>/dev/null; then $(MAKE) -C opengl; else echo "skip opengl (no glfw3/glew/gl)"; fi
	@if command -v glslc >/dev/null 2>&1; then $(MAKE) -C vulkan; else echo "skip vulkan (no glslc)";      fi
	@if command -v hipcc >/dev/null 2>&1; then $(MAKE) -C hip;  else echo "skip hip    (no hipcc)";        fi
	@if command -v nvcc  >/dev/null 2>&1; then $(MAKE) -C cuda; else echo "skip cuda   (no nvcc)";         fi
	@command -v mojo >/dev/null 2>&1 || echo "skip mojo   (no mojo; source-only)"

# --- benchmark --------------------------------------------------------------
bench:
	@bash tools/bench.sh

# --- multi-material (Noita-style) -------------------------------------------
# Builds the material-capable implementations (C, C++, Rust, Zig; Python needs
# no build). The single-material benchmark above is unaffected.
materials:
	@if command -v cc    >/dev/null 2>&1; then $(MAKE) -C c sandsim_materials;     else echo "skip c    (no cc)";    fi
	@if command -v g++   >/dev/null 2>&1; then $(MAKE) -C cpp sandsim_materials sandsim_materials_sse sandsim_materials_avx; else echo "skip cpp  (no g++)"; fi
	@if command -v cargo >/dev/null 2>&1; then cd rust && cargo build --release --offline; else echo "skip rust (no cargo)"; fi
	@if command -v zig   >/dev/null 2>&1; then cd zig && $(ZIG_BUILD_MAT);         else echo "skip zig  (no zig)";   fi

bench-materials:
	@bash tools/bench_materials.sh

# --- chunked streaming world (Noita-style) ----------------------------------
world:
	@if command -v cc    >/dev/null 2>&1; then $(MAKE) -C c sandsim_world;   else echo "skip c    (no cc)";    fi
	@if command -v g++   >/dev/null 2>&1; then $(MAKE) -C cpp sandsim_world sandsim_world_sse sandsim_world_avx; else echo "skip cpp  (no g++)"; fi
	@if command -v cargo >/dev/null 2>&1; then cd rust && cargo build --release --offline; else echo "skip rust (no cargo)"; fi
	@if command -v zig   >/dev/null 2>&1; then cd zig && $(ZIG_BUILD_WORLD);  else echo "skip zig  (no zig)";   fi

bench-world:
	@bash tools/bench_world.sh

# --- cleanup ----------------------------------------------------------------
clean:
	-$(MAKE) -C c clean
	-cd rust && cargo clean
	-rm -f zig/sandsim zig/sandsim_materials zig/sandsim_world
	-$(MAKE) -C cpp clean
	-$(MAKE) -C opengl clean
	-$(MAKE) -C hip clean
	-$(MAKE) -C cuda clean
	-$(MAKE) -C vulkan clean
