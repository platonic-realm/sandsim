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

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7, STEAM = 8, WOOD = 9, PLANT = 10, ACID = 11, SMOKE = 12, GLASS = 13, ICE = 14, SPRING = 15, TNT = 16, ASH = 17, VOLCANO = 18, VOID = 19, MUD = 20, VIRUS = 21, SPARK = 22, OBSIDIAN = 23, MATERIAL_COUNT = 24 };
enum { SG_DOWN, SG_GAS, SG_HORIZ };

static constexpr int CHUNK = 64;
static constexpr int PAD = 16;

// Window resolution + virtual-pixel scale + simulation rate. simHz is steps/second,
// decoupled from the render rate so the physics runs at the same wall-clock speed
// on every backend.
struct ViewCfg { int winW = 1024, winH = 768, scale = 3, simHz = 60; };
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

#include "../worldgen.h"   // shared deterministic seedMat() (diverse world, all backends)

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
    if (s == 2u || s == 17u) return t==7u||t==11u||t==3u||t==5u||t==4u||t==6u||t==8u||t==12u||t==0u;  // SAND/ASH -> L,A,W,O,G,F,St,Sm,E
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
    if (uGrp == 0) return s==2u||s==17u||s==7u||s==11u||s==3u||s==5u; // DOWN: sand,ash,lava,acid,water,oil
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
            if (h < 12u) cells[i] = (h < 4u) ? 12u : (h < 6u) ? 17u : 0u;  // fire -> smoke / ash / empty
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
        if (c == 5u || c == 10u || c == 4u) r = hot ? 1u : 0u;  // oil, plant & gas: instant
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
        if (c == 3u || c == 11u) r = nFL ? 1u : 0u;       // water / acid touching fire/lava
        else if (c == 6u || c == 7u) r = nW ? 1u : 0u;    // fire/lava touching water
        moved[i] = r;
        return;
    }
    if (uType == 7) {                                     // apply: water->steam, acid->smoke, fire->empty, lava->obsidian
        int i = y * uSW + x;
        if (moved[i] == 1u) { uint c = cells[i]; cells[i] = (c==3u) ? 8u : (c==11u) ? 12u : (c==6u) ? 0u : 23u; }
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
    if (uType == 14) {                                    // ice: mark ice (14) touching fire(6)/lava(7)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 14u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            uint h = (uint(x)*127u + uint(y)*163u + uint(uFrame)*41u) & 0xFFu;
            r = (hot && h < 18u) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 15) {                                    // ice: apply -> water (3)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 3u;
        return;
    }
    if (uType == 16) {                                    // freeze: mark water (3) touching ice (14)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 3u) {
            bool ice = cells[i-1]==14u||cells[i+1]==14u||cells[i-uSW]==14u||cells[i+uSW]==14u;
            uint h = (uint(x)*181u + uint(y)*67u + uint(uFrame)*103u) & 0xFFu;
            r = (ice && h < 4u) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 17) {                                    // freeze: apply -> ice (14)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 14u;
        return;
    }
    if (uType == 18) {                                    // spring: mark empty (0) touching spring (15)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 0u) {
            bool src = cells[i-1]==15u||cells[i+1]==15u||cells[i-uSW]==15u||cells[i+uSW]==15u;
            uint h = (uint(x)*89u + uint(y)*223u + uint(uFrame)*47u) & 0xFFu;
            r = (src && h < 20u) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 19) {                                    // spring: apply -> water (3)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 3u;
        return;
    }
    if (uType == 20) {                                    // tnt: mark detonators (tnt 16 touching fire/lava)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 16u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            r = hot ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 21) {                                    // tnt: blast detonators + blastable 8-neighbours -> fire
        int i = y * uSW + x;
        bool nearBlast = moved[i-1]==1u||moved[i+1]==1u||moved[i-uSW]==1u||moved[i+uSW]==1u
                       ||moved[i-uSW-1]==1u||moved[i-uSW+1]==1u||moved[i+uSW-1]==1u||moved[i+uSW+1]==1u;
        uint c = cells[i];
        bool soft = c==0u||c==2u||c==5u||c==4u||c==9u||c==10u||c==12u||c==16u;  // E,SAND,OIL,GAS,WOOD,PLANT,SMOKE,TNT
        if (moved[i]==1u || (nearBlast && soft)) cells[i] = 6u;
        return;
    }
    if (uType == 22) {                                    // volcano: mark empty (0) touching volcano (18)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 0u) {
            bool src = cells[i-1]==18u||cells[i+1]==18u||cells[i-uSW]==18u||cells[i+uSW]==18u;
            uint h = (uint(x)*109u + uint(y)*241u + uint(uFrame)*67u) & 0xFFu;
            r = (src && h < 16u) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 23) {                                    // volcano: apply -> lava (7)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 7u;
        return;
    }
    if (uType == 24) {                                    // void: mark cells touching a void (19), except wall/void
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c != 0u && c != 1u && c != 19u) {
            bool nv = cells[i-1]==19u||cells[i+1]==19u||cells[i-uSW]==19u||cells[i+uSW]==19u;
            r = nv ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 25) {                                    // void: apply -> empty (consumed)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 0u;
        return;
    }
    if (uType == 26) {                                    // mud: mark sand->mud (1) by water, mud->sand (2) by hot
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 2u) {
            bool w = cells[i-1]==3u||cells[i+1]==3u||cells[i-uSW]==3u||cells[i+uSW]==3u;
            uint h = (uint(x)*157u + uint(y)*97u + uint(uFrame)*61u) & 0xFFu;
            if (w && h < 10u) r = 1u;
        } else if (c == 20u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            uint h = (uint(x)*83u + uint(y)*173u + uint(uFrame)*109u) & 0xFFu;
            if (hot && h < 14u) r = 2u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 27) {                                    // mud: apply (1 -> mud 20, 2 -> sand 2)
        int i = y * uSW + x; uint m = moved[i];
        if (m == 1u) cells[i] = 20u; else if (m == 2u) cells[i] = 2u;
        return;
    }
    if (uType == 28) {                                    // virus: mark infect (1) / die (2)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 21u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            uint h = (uint(x)*107u + uint(y)*199u + uint(uFrame)*149u) & 0xFFu;
            if (hot || h < 16u) r = 2u;
        } else if (c != 0u && c != 1u && c != 6u && c != 7u && c != 19u) {   // not empty/wall/fire/lava/void
            bool nv = cells[i-1]==21u||cells[i+1]==21u||cells[i-uSW]==21u||cells[i+uSW]==21u;
            uint h = (uint(x)*191u + uint(y)*131u + uint(uFrame)*73u) & 0xFFu;
            if (nv && h < 36u) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 29) {                                    // virus: apply (1 -> virus 21, 2 -> empty)
        int i = y * uSW + x; uint m = moved[i];
        if (m == 1u) cells[i] = 21u; else if (m == 2u) cells[i] = 0u;
        return;
    }
    if (uType == 30) {                                    // spark: mark conduct(1)/water(2)/fizzle(3)/ignite(4)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        bool nw = cells[i-1]==3u||cells[i+1]==3u||cells[i-uSW]==3u||cells[i+uSW]==3u;
        bool ns = cells[i-1]==22u||cells[i+1]==22u||cells[i-uSW]==22u||cells[i+uSW]==22u;
        if (c == 22u) r = nw ? 2u : 3u;
        else if (c == 3u) { if (ns) r = 1u; }
        else if (c == 4u || c == 5u) { if (ns) r = 4u; }
        moved[i] = r;
        return;
    }
    if (uType == 31) {                                    // spark: apply (1->spark 22, 2->steam 8, 3->empty, 4->fire 6)
        int i = y * uSW + x; uint m = moved[i];
        if (m == 1u) cells[i] = 22u; else if (m == 2u) cells[i] = 8u;
        else if (m == 3u) cells[i] = 0u; else if (m == 4u) cells[i] = 6u;
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
uniform int uViewX, uViewY;                                // viewport scroll offset, in cells
uniform int uWinW, uWinH;                                  // window (logical) size
uniform int uPalX0, uPalY0, uPalSW, uPalGap, uPalN, uPalSel, uPalCols;
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
    if (m == 14u) return vec3(0.804, 0.922, 1.0);
    if (m == 15u) return vec3(0.122, 0.710, 0.769);
    if (m == 16u) return vec3(0.800, 0.133, 0.133);
    if (m == 17u) return vec3(0.420, 0.388, 0.345);
    if (m == 18u) return vec3(0.251, 0.165, 0.157);
    if (m == 19u) return vec3(0.235, 0.078, 0.322);
    if (m == 20u) return vec3(0.306, 0.231, 0.141);
    if (m == 21u) return vec3(0.847, 0.118, 0.608);
    if (m == 22u) return vec3(0.980, 0.941, 0.502);
    if (m == 23u) return vec3(0.165, 0.141, 0.220);
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
    int stride = uPalSW + uPalGap;
    int rows = (uPalN + uPalCols - 1) / uPalCols;
    int panelX = uPalX0 - uPalGap, panelY = uPalY0 - uPalGap;
    int panelW = uPalCols * stride + uPalGap, panelH = rows * stride + uPalGap;
    if (wx >= panelX && wx < panelX + panelW && wy >= panelY && wy < panelY + panelH) {
        vec3 hud = vec3(0.102);
        int ix = wx - uPalX0, iy = wy - uPalY0;
        if (ix >= 0 && iy >= 0) {
            int col = ix / stride, row = iy / stride;
            int slot = row * uPalCols + col;
            if (col < uPalCols && slot < uPalN && ix - col * stride < uPalSW && iy - row * stride < uPalSW)
                hud = matColor(uint(slot));
        }
        if (uPalSel >= 0) {
            int sx = uPalX0 + (uPalSel % uPalCols) * stride, sy = uPalY0 + (uPalSel / uPalCols) * stride;
            bool inO = wx >= sx-2 && wx < sx+uPalSW+2 && wy >= sy-2 && wy < sy+uPalSW+2;
            bool inI = wx >= sx && wx < sx+uPalSW && wy >= sy && wy < sy+uPalSW;
            if (inO && !inI) hud = vec3(1.0);
        }
        frag = vec4(hud, 1.0); return;
    }
    int cx = uViewX + px * uLW / uRW, cy = uViewY + py * uLH / uRH;   // world cell under this pixel
    uint m = cells[(uY0 + cy) * uSW + (uX0 + cx)] & 0xFFu;
    vec3 c = matColor(m);
    if (m == 6u || m == 7u) c = clamp(c * flick(cx, cy, uTick), 0.0, 1.0);   // fire/lava shimmer
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
            for (int t = 3; t <= 31; ++t) {         // + ... mud, virus, spark
                glUniform1i(lType, t);
                glDispatchCompute(LW / 16, LH / 16, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }
        ++frame;
        gpuAhead = true;
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
        if (material == FIRE || material == LAVA || material == STEAM || material == PLANT || material == ACID || material == SMOKE || material == ICE || material == SPRING || material == VOLCANO || material == VOID || material == WATER || material == VIRUS || material == SPARK) hasReactive = true;
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
                if (v == FIRE || v == LAVA || v == STEAM || v == PLANT || v == ACID || v == SMOKE || v == ICE || v == SPRING || v == VOLCANO || v == VOID || v == WATER || v == VIRUS || v == SPARK) hasReactive = true;
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
    const int vw = std::max(1, (cfg.winW / PIXEL) / CHUNK);   // viewport, in chunks
    const int vh = std::max(1, (cfg.winH / PIXEL) / CHUNK);
    const int MARGIN = 1;                                    // chunks of live border around the view
    const int WBOX = vw + 2 * MARGIN, HBOX = vh + 2 * MARGIN; // keep the sim small: just the view + a thin alive ring
    const int LWv = vw * CHUNK, LHv = vh * CHUNK;            // viewport, in cells
    const int renderW = LWv * PIXEL, renderH = LHv * PIXEL;
    GLFWwindow* win = initGL(true, renderW, renderH);
    if (!win) return 1;
    GLuint compute = linkProgram({compileShader(GL_COMPUTE_SHADER, kComputeSrc)});
    GLuint present = linkProgram({compileShader(GL_VERTEX_SHADER, kPresentVert),
                                  compileShader(GL_FRAGMENT_SHADER, kPresentFrag)});
    GLuint vao; glGenVertexArrays(1, &vao);

    std::string dir = "/tmp/sandsim_world_gl_interactive";
    std::filesystem::remove_all(dir);
    // Resident window = the view plus a MARGIN-chunk live border, all simulated, so a
    // ring around the view stays alive and panning reveals a living edge -- without
    // paying to simulate the whole surroundings. (The disk-streamed huge world is what
    // --bench shows.)
    GpuWorld world(WBOX, HBOX, WBOX, HBOX, dir, compute);
    world.generateAllToDisk();
    world.setWindow(0, 0);
    const int worldW = world.cellsW(), worldH = world.cellsH();
    int viewX = (worldW - LWv) / 2, viewY = (worldH - LHv) / 2;   // viewport scroll offset, in cells
    const int PAN = CHUNK / 4;                                    // pan step per key press, in cells
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
    glUniform1i(glGetUniformLocation(present, "uLW"), LWv);   // viewport extent in cells
    glUniform1i(glGetUniformLocation(present, "uLH"), LHv);
    GLint uRW = glGetUniformLocation(present, "uRW"), uRH = glGetUniformLocation(present, "uRH");
    glUniform1i(uRW, fbW); glUniform1i(uRH, fbH);
    GLint uViewX = glGetUniformLocation(present, "uViewX"), uViewY = glGetUniformLocation(present, "uViewY");
    fprintf(stderr, "sandsim [opengl]: view %dx%d (framebuffer %dx%d), scale %d, "
            "world %dx%d chunks = %dx%d cells (all simulated), %d steps/s\n",
            renderW, renderH, fbW, fbH, PIXEL, WBOX, HBOX, worldW, worldH, cfg.simHz);

    // Material palette HUD: laid out in window/logical coords (the present shader
    // scales it to the framebuffer), matching the SDL builds via the shared ui.h.
    static const uint8_t kSwatch[24] = {EMPTY, WALL, SAND, WATER, GAS, OIL, FIRE, LAVA, STEAM, WOOD, PLANT, ACID, SMOKE, GLASS, ICE, SPRING, TNT, ASH, VOLCANO, VOID, MUD, VIRUS, SPARK, OBSIDIAN};
    ui::Palette pal = ui::palette(renderW, 24);
    glUniform1i(glGetUniformLocation(present, "uWinW"), renderW);
    glUniform1i(glGetUniformLocation(present, "uWinH"), renderH);
    glUniform1i(glGetUniformLocation(present, "uPalX0"), pal.x0);
    glUniform1i(glGetUniformLocation(present, "uPalY0"), pal.y0);
    glUniform1i(glGetUniformLocation(present, "uPalSW"), pal.sw);
    glUniform1i(glGetUniformLocation(present, "uPalGap"), pal.gap);
    glUniform1i(glGetUniformLocation(present, "uPalN"), pal.n);
    glUniform1i(glGetUniformLocation(present, "uPalCols"), pal.cols);
    GLint uPalSel = glGetUniformLocation(present, "uPalSel");
    GLint uTick = glGetUniformLocation(present, "uTick");
    int tick = 0;
    int brushRadius = 4;
    bool painting = false, pMb = false, pLB = false, pRB = false;
    auto selectedIdx = [&]() { for (int i = 0; i < 24; ++i) if (kSwatch[i] == current) return i; return -1; };

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
        if (glfwGetKey(win, GLFW_KEY_I) == GLFW_PRESS) current = ICE;
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) current = SPRING;
        if (glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS) current = TNT;
        if (glfwGetKey(win, GLFW_KEY_H) == GLFW_PRESS) current = ASH;
        if (glfwGetKey(win, GLFW_KEY_V) == GLFW_PRESS) current = VOLCANO;
        if (glfwGetKey(win, GLFW_KEY_X) == GLFW_PRESS) current = VOID;
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) current = MUD;
        if (glfwGetKey(win, GLFW_KEY_Z) == GLFW_PRESS) current = VIRUS;
        if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) current = SPARK;
        if (glfwGetKey(win, GLFW_KEY_O) == GLFW_PRESS) current = OBSIDIAN;
        // Hold an arrow to scroll the viewport over the living world (smoother than
        // the old edge-triggered chunk step; the whole world is resident so it's free).
        if (glfwGetKey(win, GLFW_KEY_LEFT)  == GLFW_PRESS) viewX -= PAN;
        if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS) viewX += PAN;
        if (glfwGetKey(win, GLFW_KEY_UP)    == GLFW_PRESS) viewY -= PAN;
        if (glfwGetKey(win, GLFW_KEY_DOWN)  == GLFW_PRESS) viewY += PAN;
        viewX = std::max(0, std::min(viewX, worldW - LWv));
        viewY = std::max(0, std::min(viewY, worldH - LHv));
        bool lb = glfwGetKey(win, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS;
        bool rb = glfwGetKey(win, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS;
        if (lb && !pLB && brushRadius > 0)  brushRadius--;
        if (rb && !pRB && brushRadius < 32) brushRadius++;
        pLB = lb; pRB = rb;

        bool mb = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        double mx, my; glfwGetCursorPos(win, &mx, &my);
        if (mb && !pMb) {                                // press edge
            int h = ui::hit(pal, (int)mx, (int)my);
            if (h >= 0) current = kSwatch[h];            // clicked a palette swatch
            else painting = true;
        }
        if (!mb) painting = false;
        pMb = mb;
        if (painting) world.paint(viewX + (int)mx / PIXEL, viewY + (int)my / PIXEL, current, brushRadius);
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
        glUniform1i(uViewX, viewX); glUniform1i(uViewY, viewY);
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
