CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall
PKGS     := glew glfw3 gl
CXXFLAGS += $(shell pkg-config --cflags $(PKGS) 2>/dev/null)
LDLIBS   := $(shell pkg-config --libs $(PKGS) 2>/dev/null || echo -lGLEW -lglfw -lGL)

sandsim_gl: sandsim_gl.cpp
	$(CXX) $(CXXFLAGS) sandsim_gl.cpp -o sandsim_gl $(LDLIBS)

.PHONY: run bench clean
run: sandsim_gl
	./sandsim_gl

bench: sandsim_gl
	./sandsim_gl --bench 1000 400 300

clean:
	rm -f sandsim_gl
