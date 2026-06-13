NVCC      ?= nvcc
NVFLAGS   ?= -O2 -std=c++17
SDL_LIBS  := $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)
HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

# Optional: target a specific compute capability, e.g. `make SM=86`.
ifdef SM
NVFLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)
endif

sandsim_cuda: sandsim_cuda.cu
ifeq ($(HAVE_NVCC),)
	@echo "nvcc not found - CUDA toolkit required to build this target. Skipping."
else
	$(NVCC) $(NVFLAGS) sandsim_cuda.cu -o sandsim_cuda $(SDL_LIBS)
endif

.PHONY: run bench clean
run: sandsim_cuda
	./sandsim_cuda

bench: sandsim_cuda
	./sandsim_cuda --bench 1000 400 300

clean:
	rm -f sandsim_cuda
