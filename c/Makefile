CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -Wall -Wextra
LDLIBS  := $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)
CFLAGS  += $(shell pkg-config --cflags sdl2 2>/dev/null)

all: sandsim sandsim_materials sandsim_world

sandsim: sandsim.c
	$(CC) $(CFLAGS) sandsim.c -o sandsim $(LDLIBS)

sandsim_materials: sandsim_materials.c
	$(CC) $(CFLAGS) sandsim_materials.c -o sandsim_materials $(LDLIBS)

sandsim_world: sandsim_world.c
	$(CC) $(CFLAGS) sandsim_world.c -o sandsim_world $(LDLIBS)

.PHONY: all run run-materials run-world bench clean
run: sandsim
	./sandsim

run-materials: sandsim_materials
	./sandsim_materials

run-world: sandsim_world
	./sandsim_world

bench: sandsim
	./sandsim --bench 1000 400 300

clean:
	rm -f sandsim sandsim_materials sandsim_world
