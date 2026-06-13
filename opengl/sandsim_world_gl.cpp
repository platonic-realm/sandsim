/*
 * sandsim - multi-material streaming world, OpenGL compute (bit-identical to CPU)
 *
 * The same order-independent rule as cpp/simd_core.h, on the GPU. Each frame is
 * 16 compute dispatches (one per disjoint sub-pass) with a memory barrier
 * between. A thread decides whether its cell is the SOURCE of a swap purely from
 * its (x,y) coordinates (parity / boundary pattern) -- never from cell content --
 * so non-source threads touch nothing and the in-place update is race-free and
 * reproduces the CPU result exactly.
 *
 * Huge worlds stream like the CPU build: the live window lives in a GPU buffer;
 * a CPU shadow drives the identical chunk<->disk logic, synced only when the
 * camera moves. The interactive view renders each cell as a SCALE x SCALE virtual
 * pixel; window resolution and scale are configurable (--res WxH / --scale N, or
 * SANDSIM_RES / SANDSIM_SCALE; default 1024x768, 2x2), and the resident window is
 * sized to (winW/SCALE) x (winH/SCALE) cells.
 *
 * Modes:
 *   --bench [steps] [wch] [hch]   headless streaming benchmark (fixed 4x4 window)
 *   (default)                     interactive viewer (arrows pan, number keys paint)
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include "../ui.h"       // on-screen material palette (shared layout/hit-test)

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7, STEAM = 8, WOOD = 9, PLANT = 10, ACID = 11, SMOKE = 12, GLASS = 13, MATERIAL_COUNT = 14 };
enum { SG_DOWN, SG_GAS, SG_HORIZ };

static constexpr int CHUNK = 64;
static constexpr int PAD = 16;

// Window resolution + virtual-pixel scale + simulation rate. simHz is steps/second,
// decoupled from the render rate so the physics runs at the same wall-clock speed
// on every backend.
struct ViewCfg { int winW = 1024, winH = 768, scale = 2, simHz = 60; };
static ViewCfg parseView(int argc, char* argv[]) {
    ViewCfg c;
    if (const char* e = getenv("SANDSIM_RES"))   std::sscanf(e, "%dx%d", &c.winW, &c.winH);
    if (const char* e = getenv("SANDSIM_SCALE")) c.scale = std::atoi(e);
    if (const char* e = getenv("SANDSIM_SPS"))   c.simHz = std::atoi(e);
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--res") && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &c.winW, &c.winH);
        else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) c.scale = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--sps") && i + 1 < argc) c.simHz = std::atoi(argv[++i]);
    }
    if (c.scale < 1) c.scale = 1;
    if (c.simHz < 1) c.simHz = 1;
    if (c.winW < CHUNK * c.scale) c.winW = CHUNK * c.scale;
    if (c.winH < CHUNK * c.scale) c.winH = CHUNK * c.scale;
    return c;
}

static inline uint32_t hashCoord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}
static inline uint8_t seedMat(int gx, int gy) {
    if (gy % 40 == 39 && (gx % 11 != 0)) return WALL;
    uint32_t r = hashCoord(gx, gy) % 100u;
    switch ((gy / 40) % 3) {
        case 0:  return (r < 35u) ? SAND  : EMPTY;
        case 1:  return (r < 30u) ? WATER : EMPTY;
        default: return (r < 18u) ? GAS   : EMPTY;
    }
}

// The 16 sub-passes, in the exact order of cpp/simd_core.h.
struct Pass { int type, dx, dy, parity, grp; };   // type: 0 vert, 1 diag, 2 horiz
static const Pass kPasses[16] = {
    {0,  0,  1, 0, SG_DOWN}, {0,  0,  1, 1, SG_DOWN},
    {1, -1,  1, 0, SG_DOWN}, {1, -1,  1, 1, SG_DOWN},
    {1,  1,  1, 0, SG_DOWN}, {1,  1,  1, 1, SG_DOWN},
    {0,  0, -1, 0, SG_GAS},  {0,  0, -1, 1, SG_GAS},
    {1, -1, -1, 0, SG_GAS},  {1, -1, -1, 1, SG_GAS},
    {1,  1, -1, 0, SG_GAS},  {1,  1, -1, 1, SG_GAS},
    {2, -1,  0, 0, SG_HORIZ},{2, -1,  0, 1, SG_HORIZ},
    {2,  1,  0, 0, SG_HORIZ},{2,  1,  0, 1, SG_HORIZ},
};

// --------------------------------------------------------------------------- shaders
static const char* kComputeSrc = R"GLSL(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Cells { uint cells[]; };
layout(std430, binding = 1) buffer Moved { uint moved[]; };
uniform int uSW, uX0, uX1, uY0, uY1;
uniform int uType, uDx, uDy, uParity, uGrp, uFrame;
bool canEnter(uint s, uint t) {
    if (t == 1u) return false;                                       // WALL
    if (s == 2u) return t==7u||t==11u||t==3u||t==5u||t==4u||t==6u||t==8u||t==12u||t==0u;  // SAND -> L,A,W,O,G,F,St,Sm,E
    if (s == 7u) return t==11u||t==3u||t==5u||t==4u||t==6u||t==8u||t==12u||t==0u;         // LAVA -> A,W,O,G,F,St,Sm,E
    if (s == 11u) return t==3u||t==5u||t==4u||t==6u||t==8u||t==12u||t==0u;                // ACID -> W,O,G,F,St,Sm,E
    if (s == 3u) return t==5u||t==4u||t==6u||t==8u||t==12u||t==0u;                        // WATER -> O,G,F,St,Sm,E
    if (s == 5u) return t==4u||t==6u||t==8u||t==12u||t==0u;                               // OIL  -> G,F,St,Sm,E
    if (s == 6u) return t==0u;                                                            // FIRE -> E
    if (s == 8u) return t==0u;                                                            // STEAM -> E (rises)
    if (s == 12u) return t==0u;                                                           // SMOKE -> E (rises)
    if (s == 4u) return t==0u;                                                            // GAS  -> E
    return false;
}
bool eligible(uint s) {
    if (uGrp == 0) return s==2u||s==7u||s==11u||s==3u||s==5u;        // DOWN: sand,lava,acid,water,oil
    if (uGrp == 1) return s==4u||s==6u||s==8u||s==12u;               // GAS/FIRE/STEAM/SMOKE rise
    return s==7u||s==11u||s==3u||s==5u||s==4u||s==6u||s==8u||s==12u;  // HORIZ: + smoke
}
void main() {
    int x = uX0 + int(gl_GlobalInvocationID.x);
    int y = uY0 + int(gl_GlobalInvocationID.y);
    if (x >= uX1 || y >= uY1) return;
    if (uType == 3) {                                     // per-cell time-varying transforms
        int i = y * uSW + x; uint c = cells[i];
        if (c == 6u) {
            uint h = (uint(x) * 167u + uint(y) * 101u + uint(uFrame) * 131u) & 0xFFu;
            if (h < 12u) cells[i] = (h < 4u) ? 12u : 0u;  // fire -> smoke / empty
        } else if (c == 12u) {
            uint h = (uint(x) * 73u + uint(y) * 179u + uint(uFrame) * 149u) & 0xFFu;
            if (h < 16u) cells[i] = 0u;                   // smoke fades
        } else if (c == 8u) {
            uint h = (uint(x) * 193u + uint(y) * 97u + uint(uFrame) * 111u) & 0xFFu;
            if (h < 5u) cells[i] = 3u;                    // steam -> water
        } else if (c == 11u) {
            uint h = (uint(x) * 211u + uint(y) * 137u + uint(uFrame) * 59u) & 0xFFu;
            if (h < 3u) cells[i] = 0u;                    // acid evaporates
        }
        return;
    }
    if (uType == 4) {                                     // ignite: oil (instant) / wood (slow) by fire/lava
        int i = y * uSW + x;
        bool hot = cells[i-1]==6u||cells[i-1]==7u || cells[i+1]==6u||cells[i+1]==7u ||
                   cells[i-uSW]==6u||cells[i-uSW]==7u || cells[i+uSW]==6u||cells[i+uSW]==7u;
        uint c = cells[i]; uint r = 0u;
        if (c == 5u || c == 10u) r = hot ? 1u : 0u;       // oil & plant: instant
        else if (c == 9u && hot) {                        // wood: smoulders
            uint h = (uint(x)*149u + uint(y)*83u + uint(uFrame)*157u) & 0xFFu;
            r = (h < 28u) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 5) {                                     // ignite: apply marked cells -> fire
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 6u;
        return;
    }
    if (uType == 6) {                                     // water meets hot: mark interface cells
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        bool nW  = cells[i-1]==3u||cells[i+1]==3u||cells[i-uSW]==3u||cells[i+uSW]==3u;
        bool nFL = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u||
                   cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
        if (c == 3u) r = nFL ? 1u : 0u;                   // water touching fire/lava
        else if (c == 6u || c == 7u) r = nW ? 1u : 0u;    // fire/lava touching water
        moved[i] = r;
        return;
    }
    if (uType == 7) {                                     // apply: water->steam, fire->empty, lava->stone
        int i = y * uSW + x;
        if (moved[i] == 1u) { uint c = cells[i]; cells[i] = (c==3u) ? 8u : (c==6u) ? 0u : 1u; }
        return;
    }
    if (uType == 8) {                                     // plant grow: mark empty next to plant+water
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 0u) {
            bool nP = cells[i-1]==10u||cells[i+1]==10u||cells[i-uSW]==10u||cells[i+uSW]==10u;
            bool nW = cells[i-1]==3u||cells[i+1]==3u||cells[i-uSW]==3u||cells[i+uSW]==3u;
            uint h = (uint(x)*113u + uint(y)*191u + uint(uFrame)*71u) & 0xFFu;
            r = (nP && nW && h < 14u) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 9) {                                     // plant grow: apply -> plant
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 10u;
        return;
    }
    if (uType == 10) {                                    // acid: mark dissolvable solid touching acid (11)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        bool solid = (c==1u||c==2u||c==9u||c==10u);       // wall,sand,wood,plant
        bool nA = cells[i-1]==11u||cells[i+1]==11u||cells[i-uSW]==11u||cells[i+uSW]==11u;
        uint h = (uint(x)*53u + uint(y)*199u + uint(uFrame)*89u) & 0xFFu;
        r = (solid && nA && h < 22u) ? 1u : 0u;
        moved[i] = r;
        return;
    }
    if (uType == 11) {                                    // acid: apply -> dissolved (empty)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 0u;
        return;
    }
    if (uType == 12) {                                    // glass: mark sand (2) touching lava (7)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 2u)
            r = (cells[i-1]==7u||cells[i+1]==7u||cells[i-uSW]==7u||cells[i+uSW]==7u) ? 1u : 0u;
        moved[i] = r;
        return;
    }
    if (uType == 13) {                                    // glass: apply -> glass (13)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 13u;
        return;
    }
    int cx = x - uX0;
    bool src = (uType == 0) ? (((y - uY0) & 1) == uParity)   // vertical: row parity
                            : ((cx & 1) == uParity);          // diag/horiz: column parity
    if (uType == 2) {                                         // horizontal boundary skip
        if (uDx < 0 && uParity == 0 && (cx % 16) == 0)  src = false;
        if (uDx > 0 && uParity == 1 && (cx % 16) == 15) src = false;
    }
    if (!src) return;
    int si = y * uSW + x;
    int ti = (y + uDy) * uSW + (x + uDx);
    uint s = cells[si], t = cells[ti];
    if (eligible(s) && canEnter(s, t) && moved[si] == 0u && moved[ti] == 0u) {
        cells[ti] = s; cells[si] = t;
        moved[si] = 1u; moved[ti] = 1u;
    }
}
)GLSL";

static const char* kPresentVert = R"GLSL(
#version 430
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
)GLSL";

static const char* kPresentFrag = R"GLSL(
#version 430
layout(std430, binding = 0) buffer Cells { uint cells[]; };
uniform int uSW, uX0, uY0, uLW, uLH, uRW, uRH, uTick;
uniform int uWinW, uWinH;                                  // window (logical) size
uniform int uPalX0, uPalY0, uPalSW, uPalGap, uPalN, uPalSel;
out vec4 frag;
vec3 matColor(uint m) {
    if (m == 1u) return vec3(0.502, 0.502, 0.502);
    if (m == 2u) return vec3(0.886, 0.784, 0.471);
    if (m == 3u) return vec3(0.267, 0.533, 1.000);
    if (m == 4u) return vec3(0.690, 0.769, 0.871);
    if (m == 5u) return vec3(0.557, 0.267, 0.678);
    if (m == 6u) return vec3(1.000, 0.353, 0.118);
    if (m == 7u) return vec3(0.812, 0.106, 0.043);
    if (m == 8u) return vec3(0.863, 0.894, 0.925);
    if (m == 9u) return vec3(0.545, 0.353, 0.169);
    if (m == 10u) return vec3(0.227, 0.659, 0.290);
    if (m == 11u) return vec3(0.722, 0.941, 0.000);
    if (m == 12u) return vec3(0.345, 0.345, 0.376);
    if (m == 13u) return vec3(0.682, 0.878, 0.910);
    return vec3(0.0);
}
float flick(int lx, int ly, int tick) {                   // matches ui::flicker()
    uint h = uint(lx)*374761393u + uint(ly)*668265263u + uint(tick)*2654435761u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return 0.80 + float(h & 0xFFu) / 255.0 * 0.32;
}
void main() {
    int px = int(gl_FragCoord.x);
    int py = uRH - 1 - int(gl_FragCoord.y);
    // HUD palette, evaluated in window (logical) coordinates so it matches the
    // SDL builds regardless of HiDPI framebuffer scaling.
    int wx = px * uWinW / uRW;
    int wy = py * uWinH / uRH;
    int panelX = uPalX0 - uPalGap, panelY = uPalY0 - uPalGap;
    int panelW = uPalN * (uPalSW + uPalGap) + uPalGap, panelH = uPalSW + 2 * uPalGap;
    if (wx >= panelX && wx < panelX + panelW && wy >= panelY && wy < panelY + panelH) {
        vec3 hud = vec3(0.102);
        int ix = wx - uPalX0, rowy = wy - uPalY0;
        if (rowy >= 0 && rowy < uPalSW && ix >= 0) {
            int slot = ix / (uPalSW + uPalGap);
            int inSlot = ix - slot * (uPalSW + uPalGap);
            if (slot < uPalN && inSlot < uPalSW) hud = matColor(uint(slot));
        }
        if (uPalSel >= 0) {
            int sx = uPalX0 + uPalSel * (uPalSW + uPalGap);
            bool inO = wx >= sx-2 && wx < sx+uPalSW+2 && wy >= uPalY0-2 && wy < uPalY0+uPalSW+2;
            bool inI = wx >= sx && wx < sx+uPalSW && wy >= uPalY0 && wy < uPalY0+uPalSW;
            if (inO && !inI) hud = vec3(1.0);
        }
        frag = vec4(hud, 1.0); return;
    }
    int lx = px * uLW / uRW, ly = py * uLH / uRH;
    uint m = cells[(uY0 + ly) * uSW + (uX0 + lx)] & 0xFFu;
    vec3 c = matColor(m);
    if (m == 6u || m == 7u) c = clamp(c * flick(lx, ly, uTick), 0.0, 1.0);   // fire/lava shimmer
    frag = vec4(c, 1.0);
}
)GLSL";

static GLuint compileShader(GLenum kind, const char* src) {
    GLuint sh = glCreateShader(kind);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof(log), nullptr, log); fprintf(stderr, "shader: %s\n", log); exit(1); }
    return sh;
}
static GLuint linkProgram(std::vector<GLuint> shaders) {
    GLuint p = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(p, s);
    glLinkProgram(p);
    for (GLuint s : shaders) glDeleteShader(s);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof(log), nullptr, log); fprintf(stderr, "link: %s\n", log); exit(1); }
    return p;
}

static GLFWwindow* initGL(bool visible, int w, int h) {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    if (!glfwInit()) { glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM); if (!glfwInit()) return nullptr; }
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(w, h, "sandsim (OpenGL)", nullptr, nullptr);
    if (!win) { fprintf(stderr, "no OpenGL 4.3 context\n"); glfwTerminate(); return nullptr; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    GLenum ge = glewInit();
    if (ge != GLEW_OK && ge != GLEW_ERROR_NO_GLX_DISPLAY) { fprintf(stderr, "glewInit: %s\n", glewGetErrorString(ge)); glfwTerminate(); return nullptr; }
    glGetError();
    return win;
}

// --------------------------------------------------------------------------- world
// Mirrors cpp/sandsim_world.cpp's SimdWorld: a CPU shadow drives identical
// chunk<->disk streaming; the live window also lives in a GPU buffer, where the
// 16-pass step runs. Shadow and GPU buffer are synced only at window changes
// (and summary) -- not per frame -- so the heavy loop stays on the GPU.
class GpuWorld {
public:
    GpuWorld(int gw, int gh, int wbox, int hbox, std::string dir, GLuint computeProg)
        : gw(gw), gh(gh), LW(gw * CHUNK), LH(gh * CHUNK), SW(LW + 2 * PAD), SH(LH + 2 * PAD),
          X0(PAD), X1(PAD + LW), Y0(PAD), Y1(PAD + LH),
          wbox(wbox), hbox(hbox), dir(std::move(dir)), prog(computeProg) {
        std::filesystem::create_directories(this->dir);
        shadow.assign((size_t)SW * SH, WALL);

        glGenBuffers(1, &cellsBuf); glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellsBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (size_t)SW * SH * 4, shadow.data(), GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cellsBuf);
        glGenBuffers(1, &movedBuf); glBindBuffer(GL_SHADER_STORAGE_BUFFER, movedBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (size_t)SW * SH * 4, nullptr, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, movedBuf);

        glUseProgram(prog);
        lSW = glGetUniformLocation(prog, "uSW"); lX0 = glGetUniformLocation(prog, "uX0");
        lX1 = glGetUniformLocation(prog, "uX1"); lY0 = glGetUniformLocation(prog, "uY0");
        lY1 = glGetUniformLocation(prog, "uY1"); lType = glGetUniformLocation(prog, "uType");
        lDx = glGetUniformLocation(prog, "uDx"); lDy = glGetUniformLocation(prog, "uDy");
        lPar = glGetUniformLocation(prog, "uParity"); lGrp = glGetUniformLocation(prog, "uGrp");
        lFrame = glGetUniformLocation(prog, "uFrame");
        glUniform1i(lSW, SW); glUniform1i(lX0, X0); glUniform1i(lX1, X1);
        glUniform1i(lY0, Y0); glUniform1i(lY1, Y1);
    }
    ~GpuWorld() { glDeleteBuffers(1, &cellsBuf); glDeleteBuffers(1, &movedBuf); }

    GLuint buffer() const { return cellsBuf; }
    int winChunksW() const { return gw; }
    int winChunksH() const { return gh; }
    int cellsW() const { return LW; }
    int cellsH() const { return LH; }
    int stride() const { return SW; }
    int originX() const { return X0; }
    int originY() const { return Y0; }

    void generateAllToDisk() {
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) { genBox(cx, cy, buf); writeBox(cx, cy, buf); }
    }

    void setWindow(int camCx, int camCy) {
        if (windowValid && camCx == winCx && camCy == winCy) return;
        syncDown();                                // bring GPU state into the shadow
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        if (windowValid)
            for (int y = 0; y < gh; ++y)
                for (int x = 0; x < gw; ++x) { extractChunk(x, y, buf); writeBox(winCx + x, winCy + y, buf); }
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                if (!readBox(camCx + x, camCy + y, buf)) genBox(camCx + x, camCy + y, buf);
                injectChunk(x, y, buf);
            }
        winCx = camCx; winCy = camCy; windowValid = true;
        residentMax = gw * gh;
        syncUp();                                  // push the new window to the GPU
    }

    void step() {
        glUseProgram(prog);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, movedBuf);
        const uint32_t zero = 0;
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        for (const Pass& p : kPasses) {
            glUniform1i(lType, p.type); glUniform1i(lDx, p.dx); glUniform1i(lDy, p.dy);
            glUniform1i(lPar, p.parity); glUniform1i(lGrp, p.grp);
            glDispatchCompute(LW / 16, LH / 16, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        if (hasReactive) {                          // reactions (gated): see shader pass types
            glUniform1i(lFrame, (int)frame);
            for (int t = 3; t <= 13; ++t) {         // + glass (sand+lava)
                glUniform1i(lType, t);
                glDispatchCompute(LW / 16, LH / 16, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }
        ++frame;
        gpuAhead = true;
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
        if (material == FIRE || material == LAVA || material == STEAM || material == PLANT || material == ACID || material == SMOKE) hasReactive = true;
        syncDown();
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = lx + dx, ny = ly + dy;
                if (nx >= 0 && nx < LW && ny >= 0 && ny < LH && dx * dx + dy * dy <= radius * radius)
                    shadow[(size_t)(ny + Y0) * SW + (nx + X0)] = material;
            }
        syncUp();
    }

    void summary(uint64_t& checksum, uint64_t counts[MATERIAL_COUNT]) {
        syncDown();
        for (int i = 0; i < MATERIAL_COUNT; ++i) counts[i] = 0;
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        uint64_t c = 14695981039346656037ull;
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) {
                bool inWin = windowValid && cx >= winCx && cx < winCx + gw && cy >= winCy && cy < winCy + gh;
                const uint8_t* data;
                if (inWin) { extractChunk(cx - winCx, cy - winCy, buf); data = buf.data(); }
                else { readBox(cx, cy, buf); data = buf.data(); }
                for (int i = 0; i < CHUNK * CHUNK; ++i) { uint8_t v = data[i]; counts[v]++; c = (c ^ v) * 1099511628211ull; }
            }
        checksum = c;
    }

    int residentMaxCount() const { return residentMax; }
    long long diskWrites() const { return nWrites; }
    long long diskReads() const { return nReads; }

private:
    const int gw, gh;
    const int LW, LH;
    const int SW, SH;
    const int X0, X1, Y0, Y1;
    int wbox, hbox;
    std::string dir;
    GLuint prog, cellsBuf = 0, movedBuf = 0;
    std::vector<uint32_t> shadow;        // CPU mirror of the live padded window
    bool gpuAhead = false;               // GPU buffer holds newer data than the shadow
    int winCx = 0, winCy = 0;
    bool windowValid = false;
    int residentMax = 0;
    long long nWrites = 0, nReads = 0;
    uint32_t frame = 0;
    bool hasReactive = false;            // gates the reaction dispatches (fire/lava)
    GLint lSW, lX0, lX1, lY0, lY1, lType, lDx, lDy, lPar, lGrp, lFrame;

    void syncDown() {
        if (!gpuAhead) return;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellsBuf);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (size_t)SW * SH * 4, shadow.data());
        gpuAhead = false;
    }
    void syncUp() {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellsBuf);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (size_t)SW * SH * 4, shadow.data());
        gpuAhead = false;
    }

    void extractChunk(int cgx, int cgy, std::vector<uint8_t>& out) const {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                out[ly * CHUNK + lx] = (uint8_t)(shadow[(size_t)(Y0 + cgy * CHUNK + ly) * SW + (X0 + cgx * CHUNK + lx)] & 0xFFu);
    }
    void injectChunk(int cgx, int cgy, const std::vector<uint8_t>& in) {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx) {
                uint8_t v = in[ly * CHUNK + lx];
                if (v == FIRE || v == LAVA || v == STEAM || v == PLANT || v == ACID || v == SMOKE) hasReactive = true;
                shadow[(size_t)(Y0 + cgy * CHUNK + ly) * SW + (X0 + cgx * CHUNK + lx)] = v;
            }
    }
    void genBox(int cx, int cy, std::vector<uint8_t>& buf) {
        for (int y = 0; y < CHUNK; ++y)
            for (int x = 0; x < CHUNK; ++x)
                buf[y * CHUNK + x] = seedMat(cx * CHUNK + x, cy * CHUNK + y);
    }
    std::string path(int cx, int cy) const {
        char n[64]; std::snprintf(n, sizeof(n), "/b_%d_%d.bin", cx, cy); return dir + n;
    }
    void writeBox(int cx, int cy, const std::vector<uint8_t>& buf) {
        std::ofstream f(path(cx, cy), std::ios::binary);
        f.write((const char*)buf.data(), CHUNK * CHUNK); ++nWrites;
    }
    bool readBox(int cx, int cy, std::vector<uint8_t>& buf) {
        std::ifstream f(path(cx, cy), std::ios::binary);
        if (!f) return false;
        f.read((char*)buf.data(), CHUNK * CHUNK); ++nReads;
        return true;
    }
};

// --------------------------------------------------------------------------- modes
static int runBench(int steps, int wbox, int hbox) {
    const int gw = 4, gh = 4;
    if (wbox < gw) wbox = gw;
    if (hbox < gh) hbox = gh;
    GLFWwindow* win = initGL(false, 64, 64);
    if (!win) return 1;
    GLuint prog = linkProgram({compileShader(GL_COMPUTE_SHADER, kComputeSrc)});

    std::string dir = "/tmp/sandsim_world_gl_" + std::to_string(steps) + "_" +
                      std::to_string(wbox) + "x" + std::to_string(hbox);
    std::filesystem::remove_all(dir);
    GpuWorld world(gw, gh, wbox, hbox, dir, prog);
    world.generateAllToDisk();

    uint64_t startCk, startCnt[MATERIAL_COUNT];
    world.summary(startCk, startCnt);

    int nposX = wbox - gw + 1, nposY = hbox - gh + 1, nWin = nposX * nposY;
    (void)nposY;
    glFinish();
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * nWin / steps);
        if (visit >= nWin) visit = nWin - 1;
        int row = visit / nposX, col = visit % nposX;
        world.setWindow((row % 2 == 0) ? col : (nposX - 1 - col), row);
        world.step();
    }
    glFinish();
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, cnt[MATERIAL_COUNT];
    world.summary(ck, cnt);
    bool conserved = true;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != startCnt[i]) conserved = false;
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double mc = (ms > 0.0) ? (double)world.cellsW() * world.cellsH() * steps / (ms / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=opengl rule=world window=%dx%d wbox=%d hbox=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           gw, gh, wbox, hbox, steps, ms, mc, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           world.residentMaxCount(), world.diskWrites(), world.diskReads(), conserved ? "yes" : "no");
    std::filesystem::remove_all(dir);
    glfwDestroyWindow(win); glfwTerminate();
    return conserved ? 0 : 2;
}

static int runInteractive(ViewCfg cfg) {
    const int PIXEL = cfg.scale;
    const int gw = std::max(1, (cfg.winW / PIXEL) / CHUNK);
    const int gh = std::max(1, (cfg.winH / PIXEL) / CHUNK);
    const int WBOX = 2 * gw, HBOX = 2 * gh;
    const int renderW = gw * CHUNK * PIXEL, renderH = gh * CHUNK * PIXEL;
    GLFWwindow* win = initGL(true, renderW, renderH);
    if (!win) return 1;
    GLuint compute = linkProgram({compileShader(GL_COMPUTE_SHADER, kComputeSrc)});
    GLuint present = linkProgram({compileShader(GL_VERTEX_SHADER, kPresentVert),
                                  compileShader(GL_FRAGMENT_SHADER, kPresentFrag)});
    GLuint vao; glGenVertexArrays(1, &vao);

    std::string dir = "/tmp/sandsim_world_gl_interactive";
    std::filesystem::remove_all(dir);
    GpuWorld world(gw, gh, WBOX, HBOX, dir, compute);
    world.generateAllToDisk();
    int camCx = 0, camCy = 0;
    world.setWindow(camCx, camCy);
    uint8_t current = SAND;

    // The actual framebuffer can differ from the requested window size under a
    // HiDPI / fractional-scaling compositor; map cells across the real pixels so
    // the grid fills the window exactly like the SDL builds' logical-size scaling.
    int fbW = renderW, fbH = renderH;
    glfwGetFramebufferSize(win, &fbW, &fbH);
    glUseProgram(present);
    glUniform1i(glGetUniformLocation(present, "uSW"), world.stride());
    glUniform1i(glGetUniformLocation(present, "uX0"), world.originX());
    glUniform1i(glGetUniformLocation(present, "uY0"), world.originY());
    glUniform1i(glGetUniformLocation(present, "uLW"), world.cellsW());
    glUniform1i(glGetUniformLocation(present, "uLH"), world.cellsH());
    GLint uRW = glGetUniformLocation(present, "uRW"), uRH = glGetUniformLocation(present, "uRH");
    glUniform1i(uRW, fbW); glUniform1i(uRH, fbH);
    fprintf(stderr, "sandsim [opengl]: window %dx%d (framebuffer %dx%d), scale %d, "
            "grid %dx%d chunks = %dx%d cells, %d steps/s\n",
            renderW, renderH, fbW, fbH, PIXEL, gw, gh, gw * CHUNK, gh * CHUNK, cfg.simHz);

    // Material palette HUD: laid out in window/logical coords (the present shader
    // scales it to the framebuffer), matching the SDL builds via the shared ui.h.
    static const uint8_t kSwatch[14] = {EMPTY, WALL, SAND, WATER, GAS, OIL, FIRE, LAVA, STEAM, WOOD, PLANT, ACID, SMOKE, GLASS};
    ui::Palette pal = ui::palette(renderW, 14);
    glUniform1i(glGetUniformLocation(present, "uWinW"), renderW);
    glUniform1i(glGetUniformLocation(present, "uWinH"), renderH);
    glUniform1i(glGetUniformLocation(present, "uPalX0"), pal.x0);
    glUniform1i(glGetUniformLocation(present, "uPalY0"), pal.y0);
    glUniform1i(glGetUniformLocation(present, "uPalSW"), pal.sw);
    glUniform1i(glGetUniformLocation(present, "uPalGap"), pal.gap);
    glUniform1i(glGetUniformLocation(present, "uPalN"), pal.n);
    GLint uPalSel = glGetUniformLocation(present, "uPalSel");
    GLint uTick = glGetUniformLocation(present, "uTick");
    int tick = 0;
    int brushRadius = 4;
    bool painting = false, pMb = false, pLB = false, pRB = false;
    auto selectedIdx = [&]() { for (int i = 0; i < 14; ++i) if (kSwatch[i] == current) return i; return -1; };

    glfwSwapInterval(1);                             // vsync: cap rendering (physics is decoupled)
    const double stepDt = 1.0 / cfg.simHz;          // seconds per simulation step
    double acc = 0.0;
    auto last = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        if (glfwGetKey(win, GLFW_KEY_0) == GLFW_PRESS) current = EMPTY;
        if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) current = WALL;
        if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) current = SAND;
        if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) current = WATER;
        if (glfwGetKey(win, GLFW_KEY_4) == GLFW_PRESS) current = GAS;
        if (glfwGetKey(win, GLFW_KEY_5) == GLFW_PRESS) current = OIL;
        if (glfwGetKey(win, GLFW_KEY_6) == GLFW_PRESS) current = FIRE;
        if (glfwGetKey(win, GLFW_KEY_7) == GLFW_PRESS) current = LAVA;
        if (glfwGetKey(win, GLFW_KEY_8) == GLFW_PRESS) current = STEAM;
        if (glfwGetKey(win, GLFW_KEY_9) == GLFW_PRESS) current = WOOD;
        if (glfwGetKey(win, GLFW_KEY_P) == GLFW_PRESS) current = PLANT;
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) current = ACID;
        if (glfwGetKey(win, GLFW_KEY_M) == GLFW_PRESS) current = SMOKE;
        if (glfwGetKey(win, GLFW_KEY_G) == GLFW_PRESS) current = GLASS;
        static bool pL = false, pR = false, pU = false, pD = false;
        bool l = glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS;
        bool r = glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS;
        bool u = glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS;
        bool d = glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS;
        if (l && !pL && camCx > 0) camCx--;
        if (r && !pR && camCx < WBOX - gw) camCx++;
        if (u && !pU && camCy > 0) camCy--;
        if (d && !pD && camCy < HBOX - gh) camCy++;
        pL = l; pR = r; pU = u; pD = d;
        bool lb = glfwGetKey(win, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS;
        bool rb = glfwGetKey(win, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS;
        if (lb && !pLB && brushRadius > 0)  brushRadius--;
        if (rb && !pRB && brushRadius < 32) brushRadius++;
        pLB = lb; pRB = rb;

        world.setWindow(camCx, camCy);
        bool mb = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        double mx, my; glfwGetCursorPos(win, &mx, &my);
        if (mb && !pMb) {                                // press edge
            int h = ui::hit(pal, (int)mx, (int)my);
            if (h >= 0) current = kSwatch[h];            // clicked a palette swatch
            else painting = true;
        }
        if (!mb) painting = false;
        pMb = mb;
        if (painting) world.paint((int)mx / PIXEL, (int)my / PIXEL, current, brushRadius);
        // Advance the simulation by however much real time elapsed, so physics
        // runs at cfg.simHz steps/s regardless of the render frame rate.
        auto nowT = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double>(nowT - last).count();
        last = nowT;
        for (int n = 0; acc >= stepDt && n < 8; ++n) { world.step(); acc -= stepDt; }
        if (acc > stepDt) acc = stepDt;             // drop backlog after a stall

        glUseProgram(present);
        int cfbW, cfbH; glfwGetFramebufferSize(win, &cfbW, &cfbH);
        if (cfbW != fbW || cfbH != fbH) { fbW = cfbW; fbH = cfbH; glUniform1i(uRW, fbW); glUniform1i(uRH, fbH); }
        glUniform1i(uPalSel, selectedIdx());
        glUniform1i(uTick, ++tick);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, world.buffer());
        glViewport(0, 0, fbW, fbH);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);
    }
    std::filesystem::remove_all(dir);
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        int steps = (argc > 2) ? std::atoi(argv[2]) : 600;
        int wbox  = (argc > 3) ? std::atoi(argv[3]) : 6;
        int hbox  = (argc > 4) ? std::atoi(argv[4]) : 6;
        return runBench(steps, wbox, hbox);
    }
    return runInteractive(parseView(argc, argv));
}
