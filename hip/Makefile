HIPCC    ?= hipcc
HIPFLAGS ?= -std=c++17 -O2
SDL_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null)
SDL_LIBS   := $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)

# Optional: target a specific GPU arch, e.g. `make GPU_ARCH=gfx1032`.
ifdef GPU_ARCH
HIPFLAGS += --offload-arch=$(GPU_ARCH)
endif

sandsim_hip: sandsim_hip.cpp
	$(HIPCC) $(HIPFLAGS) $(SDL_CFLAGS) sandsim_hip.cpp -o sandsim_hip $(SDL_LIBS)

.PHONY: run bench clean
run: sandsim_hip
	./sandsim_hip

bench: sandsim_hip
	./sandsim_hip --bench 1000 400 300

clean:
	rm -f sandsim_hip
