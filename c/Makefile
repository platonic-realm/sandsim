CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -Wall -Wextra
LDLIBS  := $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)
CFLAGS  += $(shell pkg-config --cflags sdl2 2>/dev/null)

sandsim: sandsim.c
	$(CC) $(CFLAGS) sandsim.c -o sandsim $(LDLIBS)

.PHONY: run bench clean
run: sandsim
	./sandsim

bench: sandsim
	./sandsim --bench 1000 400 300

clean:
	rm -f sandsim
