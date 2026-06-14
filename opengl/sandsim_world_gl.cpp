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

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7, STEAM = 8, WOOD = 9, PLANT = 10, ACID = 11, SMOKE = 12, GLASS = 13, ICE = 14, SPRING = 15, TNT = 16, ASH = 17, VOLCANO = 18, VOID = 19, MUD = 20, VIRUS = 21, SPARK = 22, OBSIDIAN = 23, SALT = 24, SNOW = 25, MERCURY = 26, GUNPOWDER = 27, THERMITE = 28, FROST = 29, WISP = 30, COAL = 31, EMBER = 32, CLONER = 33, CRYSTAL = 34, ANTIMATTER = 35, MOSS = 36, FUMES = 37, WIRE = 38, EHEAD = 39, ETAIL = 40, IGNITER = 41, SENSOR = 42, LIFE = 43, GEYSER = 44, LYE = 45, SODIUM = 46, CORAL = 47, PHOSPHORUS = 48, CEMENT = 49, CHLORINE = 50, BATTERY = 51, FUSE = 52, BURNFUSE = 53, CRYO = 54, LAMP = 55, LAMPLIT = 56, PETRIFY = 57, FIREWORK = 58, LEVITON = 59, SPROUT = 60, BELT = 61, MAGNET = 62, IRON = 63, NITRO = 64, RUST = 65, SEED = 66, LASER = 67, BEAM = 68, MATERIAL_COUNT = 69 };
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
    if (s == 2u || s == 17u || s == 27u || s == 28u || s == 31u || s == 32u || s == 45u || s == 46u || s == 48u || s == 49u || s == 63u || s == 65u || s == 66u) return t==7u||t==11u||t==3u||t==64u||t==5u||t==54u||t==25u||t==37u||t==50u||t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;  // SAND/ASH/GUNPOWDER/THERMITE/COAL/EMBER/LYE/SODIUM/PHOSPHORUS/CEMENT/IRON/RUST/SEED
    if (s == 7u) return t==11u||t==3u||t==64u||t==5u||t==54u||t==25u||t==37u||t==50u||t==4u||t==59u||t==6u||t==8u||t==12u||t==0u; // LAVA -> A,W,O,SNOW,FUMES,G,F,St,Sm,E
    if (s == 11u) return t==3u||t==64u||t==5u||t==54u||t==25u||t==37u||t==50u||t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;        // ACID -> W,O,SNOW,FUMES,G,F,St,Sm,E
    if (s == 3u || s == 64u) return t==5u||t==54u||t==25u||t==37u||t==50u||t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;                // WATER -> O,SNOW,FUMES,G,F,St,Sm,E
    if (s == 5u || s == 54u) return t==25u||t==37u||t==50u||t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;                       // OIL  -> SNOW,FUMES,G,F,St,Sm,E
    if (s == 25u) return t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;                              // SNOW (light powder) -> G,F,St,Sm,E
    if (s == 37u || s == 50u) return t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;                              // FUMES (heavy gas) -> sinks through G,F,St,Sm,E
    if (s == 26u) return t==2u||t==7u||t==11u||t==3u||t==64u||t==5u||t==54u||t==25u||t==37u||t==50u||t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;  // MERCURY (heaviest) -> SAND + everything below
    if (s == 30u) return t==3u||t==64u||t==5u||t==54u||t==11u||t==7u||t==26u||t==4u||t==59u||t==6u||t==8u||t==12u||t==0u;  // WISP (lightest) -> rises through W,O,A,L,MERCURY + gases + E
    if (s == 6u) return t==0u;                                                            // FIRE -> E
    if (s == 8u) return t==0u;                                                            // STEAM -> E (rises)
    if (s == 12u) return t==0u;                                                           // SMOKE -> E (rises)
    if (s == 4u || s == 59u) return t==0u;                                                            // GAS  -> E
    return false;
}
bool eligible(uint s) {
    if (uGrp == 0) return s==2u||s==17u||s==27u||s==28u||s==31u||s==32u||s==45u||s==46u||s==48u||s==49u||s==63u||s==65u||s==66u||s==25u||s==37u||s==50u||s==26u||s==7u||s==11u||s==3u||s==64u||s==5u||s==54u; // DOWN: +lye/sodium/phosphorus/cement/iron/rust/seed
    if (uGrp == 1) return s==4u||s==6u||s==8u||s==12u||s==30u||s==59u;  // +LEVITON rises (up + diag-up, NOT horiz)       // GAS/FIRE/STEAM/SMOKE/WISP rise
    return s==7u||s==26u||s==11u||s==3u||s==64u||s==5u||s==54u||s==4u||s==6u||s==8u||s==12u||s==30u||s==37u||s==50u;  // HORIZ: + mercury, smoke, wisp, fumes
}
bool det(uint m) { return m!=0u && m!=1u && m!=38u && m!=39u && m!=40u && m!=41u && m!=42u; } // SENSOR-detectable: not empty/wall/circuit
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
        if (c == 5u || c == 10u || c == 4u || c == 30u || c == 36u || c == 37u) r = hot ? 1u : 0u;  // oil, plant, gas, wisp, moss & fumes: instant
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
    if (uType == 14) {                                    // ice/snow: mark ice(14)/snow(25) touching fire(6)/lava(7)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 14u || cells[i] == 25u) {
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
    if (uType == 20) {                                    // tnt/gunpowder: mark detonators (16/27 touching fire/lava)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 16u || cells[i] == 27u || cells[i] == 64u) {
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
        bool soft = c==0u||c==2u||c==5u||c==4u||c==9u||c==10u||c==12u||c==16u||c==27u||c==64u;  // E,SAND,OIL,GAS,WOOD,PLANT,SMOKE,TNT,GUNPOWDER
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
        else if (c == 4u || c == 5u || c == 30u || c == 37u) { if (ns) r = 4u; }
        moved[i] = r;
        return;
    }
    if (uType == 31) {                                    // spark: apply (1->spark 22, 2->steam 8, 3->empty, 4->fire 6)
        int i = y * uSW + x; uint m = moved[i];
        if (m == 1u) cells[i] = 22u; else if (m == 2u) cells[i] = 8u;
        else if (m == 3u) cells[i] = 0u; else if (m == 4u) cells[i] = 6u;
        return;
    }
    if (uType == 32) {                                    // salt: mark salt-in-water(1) / ice-by-salt(2)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 24u) {
            bool w = cells[i-1]==3u||cells[i+1]==3u||cells[i-uSW]==3u||cells[i+uSW]==3u;
            uint h = (uint(x)*211u + uint(y)*79u + uint(uFrame)*167u) & 0xFFu;
            if (w && h < 24u) r = 1u;
        } else if (c == 14u) {
            bool s = cells[i-1]==24u||cells[i+1]==24u||cells[i-uSW]==24u||cells[i+uSW]==24u;
            uint h = (uint(x)*71u + uint(y)*187u + uint(uFrame)*113u) & 0xFFu;
            if (s && h < 40u) r = 2u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 33) {                                    // salt: apply (1 -> empty, 2 -> water 3)
        int i = y * uSW + x; uint m = moved[i];
        if (m == 1u) cells[i] = 0u; else if (m == 2u) cells[i] = 3u;
        return;
    }
    if (uType == 34) {                                    // mercury: mark plant(10) touching mercury(26)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 10u) {
            bool merc = cells[i-1]==26u||cells[i+1]==26u||cells[i-uSW]==26u||cells[i+uSW]==26u;
            uint h = (uint(x)*101u + uint(y)*217u + uint(uFrame)*131u) & 0xFFu;
            r = (merc && h < 20u) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 35) {                                    // mercury: apply -> empty (plant poisoned)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 0u;
        return;
    }
    if (uType == 36) {                                    // thermite: mark powder (28) touching fire/lava
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 28u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            r = hot ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 37) {                                    // thermite: marked -> fire; melt adjacent solids -> lava
        int i = y * uSW + x;
        if (moved[i] == 1u) { cells[i] = 6u; return; }
        bool nearBurn = moved[i-1]==1u||moved[i+1]==1u||moved[i-uSW]==1u||moved[i+uSW]==1u;
        uint c = cells[i];
        bool melt = c==1u||c==13u||c==23u||c==2u||c==9u;  // WALL,GLASS,OBSIDIAN,SAND,WOOD
        if (nearBurn && melt) cells[i] = 7u;
        return;
    }
    if (uType == 38) {                                    // frost: mark water->frost(1)/frost->ice(3)/plant->wither(4)/frost->melt(5)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        bool nf = cells[i-1]==29u||cells[i+1]==29u||cells[i-uSW]==29u||cells[i+uSW]==29u;
        if (c == 29u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            r = hot ? 5u : 3u;
        } else if (c == 3u) { if (nf) r = 1u; }
        else if (c == 10u) { if (nf) r = 4u; }
        moved[i] = r;
        return;
    }
    if (uType == 39) {                                    // frost: apply (1->frost 29, 3->ice 14, 4->empty, 5->water 3)
        int i = y * uSW + x; uint m = moved[i];
        if (m == 1u) cells[i] = 29u; else if (m == 3u) cells[i] = 14u;
        else if (m == 4u) cells[i] = 0u; else if (m == 5u) cells[i] = 3u;
        return;
    }
    if (uType == 40) {                                    // coal: mark coal-catches(1, hot/ember nbr) / ember(2)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 31u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            bool ember = cells[i-1]==32u||cells[i+1]==32u||cells[i-uSW]==32u||cells[i+uSW]==32u;
            if (hot || ember) r = 1u;
        } else if (c == 32u) r = 2u;
        moved[i] = r;
        return;
    }
    if (uType == 41) {                                    // coal: apply (1->ember 32, 2->ash 17 or stay, empty next to ember -> fire)
        int i = y * uSW + x; uint m = moved[i];
        if (m == 1u) { cells[i] = 32u; return; }
        if (m == 2u) {                                    // ember ages to ash (else stays ember)
            uint h = (uint(x)*181u + uint(y)*61u + uint(uFrame)*139u) & 0xFFu;
            if (h < 8u) cells[i] = 17u;
            return;
        }
        if (cells[i] == 0u) {                             // empty next to an ember catches flame
            bool litN = moved[i-1]==2u||moved[i+1]==2u||moved[i-uSW]==2u||moved[i+uSW]==2u;
            uint h = (uint(x)*97u + uint(y)*149u + uint(uFrame)*73u) & 0xFFu;
            if (litN && h < 64u) cells[i] = 6u;
        }
        return;
    }
    if (uType == 42) {                                    // cloner: store the cloneable material above each cloner (33) in scratch
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 33u) {
            uint above = cells[i-uSW];
            if (above != 0u && above != 1u && above != 33u) r = above;  // skip EMPTY/WALL/CLONER
        }
        moved[i] = r;
        return;
    }
    if (uType == 43) {                                    // cloner: fill EMPTY below a loaded cloner with the stored material
        int i = y * uSW + x;
        if (cells[i] == 0u && moved[i-uSW] != 0u) cells[i] = moved[i-uSW];
        return;
    }
    if (uType == 44) {                                    // crystal: mark EMPTY with exactly one crystal(34) neighbour-of-8 (frame-hashed)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 0u) {
            int n = int(cells[i-1]==34u)+int(cells[i+1]==34u)+int(cells[i-uSW]==34u)+int(cells[i+uSW]==34u)
                  + int(cells[i-uSW-1]==34u)+int(cells[i-uSW+1]==34u)+int(cells[i+uSW-1]==34u)+int(cells[i+uSW+1]==34u);
            uint h = (uint(x)*127u + uint(y)*89u + uint(uFrame)*167u) & 0xFFu;
            if (n == 1 && h < 10u) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 45) {                                    // crystal: apply (1 -> crystal 34)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 34u;
        return;
    }
    if (uType == 46) {                                    // antimatter: mark each antimatter(35) touching matter (!=empty,!=antimatter)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 35u) {
            bool m = (cells[i-1]!=0u&&cells[i-1]!=35u)||(cells[i+1]!=0u&&cells[i+1]!=35u)
                   ||(cells[i-uSW]!=0u&&cells[i-uSW]!=35u)||(cells[i+uSW]!=0u&&cells[i+uSW]!=35u);
            if (m) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 47) {                                    // antimatter: apply (marked -> fire 6; matter next to one -> empty)
        int i = y * uSW + x;
        if (moved[i] == 1u) { cells[i] = 6u; return; }
        bool nearAnnih = moved[i-1]==1u||moved[i+1]==1u||moved[i-uSW]==1u||moved[i+uSW]==1u;
        if (nearAnnih && cells[i] != 0u && cells[i] != 35u) cells[i] = 0u;
        return;
    }
    if (uType == 48) {                                    // moss: mark EMPTY next to moss(36) AND a stone/wood anchor (1/23/13/9), frame-hashed
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 0u) {
            bool nMoss = cells[i-1]==36u||cells[i+1]==36u||cells[i-uSW]==36u||cells[i+uSW]==36u;
            uint a=cells[i-1],b=cells[i+1],c2=cells[i-uSW],d=cells[i+uSW];
            bool nAnc = (a==1u||a==23u||a==13u||a==9u)||(b==1u||b==23u||b==13u||b==9u)||(c2==1u||c2==23u||c2==13u||c2==9u)||(d==1u||d==23u||d==13u||d==9u);
            uint h = (uint(x)*139u + uint(y)*113u + uint(uFrame)*61u) & 0xFFu;
            if (nMoss && nAnc && h < 7u) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 49) {                                    // moss: apply (1 -> moss 36)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 36u;
        return;
    }
    if (uType == 50) {                                    // wireworld: compute next state into scratch (head39->tail40, tail40->wire38, wire38->head if 1|2 head nbrs)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 39u) r = 40u;
        else if (c == 40u) r = 38u;
        else if (c == 38u) {
            int h = int(cells[i-1]==39u)+int(cells[i+1]==39u)+int(cells[i-uSW]==39u)+int(cells[i+uSW]==39u)
                  + int(cells[i-uSW-1]==39u)+int(cells[i-uSW+1]==39u)+int(cells[i+uSW-1]==39u)+int(cells[i+uSW+1]==39u);
            r = (h == 1 || h == 2) ? 39u : 38u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 51) {                                    // wireworld: apply (scratch != 0 -> become it)
        int i = y * uSW + x;
        if (moved[i] != 0u) cells[i] = moved[i];
        return;
    }
    if (uType == 52) {                                    // igniter: mark each igniter(41) next to an electron head(39)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 41u)
            r = (cells[i-1]==39u||cells[i+1]==39u||cells[i-uSW]==39u||cells[i+uSW]==39u) ? 1u : 0u;
        moved[i] = r;
        return;
    }
    if (uType == 53) {                                    // igniter: empty next to a triggered igniter -> fire
        int i = y * uSW + x;
        if (cells[i] == 0u && (moved[i-1]==1u||moved[i+1]==1u||moved[i-uSW]==1u||moved[i+uSW]==1u)) cells[i] = 6u;
        return;
    }
    if (uType == 54) {                                    // sensor: mark each sensor(42) with a detectable neighbour (not 0/1/38/39/40/41/42)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 42u) {
            uint a=cells[i-1],b=cells[i+1],c2=cells[i-uSW],d=cells[i+uSW];
            r = (det(a)||det(b)||det(c2)||det(d)) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 55) {                                    // sensor: a WIRE(38) next to a marked sensor -> electron head(39)
        int i = y * uSW + x;
        if (cells[i] == 38u && (moved[i-1]==1u||moved[i+1]==1u||moved[i-uSW]==1u||moved[i+uSW]==1u)) cells[i] = 39u;
        return;
    }
    if (uType == 56) {                                    // life: mark each LIFE/EMPTY cell's Conway fate (1=live, 2=empty, 0=leave)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 43u || c == 0u) {
            int n = int(cells[i-1]==43u)+int(cells[i+1]==43u)+int(cells[i-uSW]==43u)+int(cells[i+uSW]==43u)
                  + int(cells[i-uSW-1]==43u)+int(cells[i-uSW+1]==43u)+int(cells[i+uSW-1]==43u)+int(cells[i+uSW+1]==43u);
            if (c == 43u) r = (n == 2 || n == 3) ? 1u : 2u;
            else          r = (n == 3) ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 57) {                                    // life: apply (1 -> LIFE 43, 2 -> EMPTY 0)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 43u; else if (moved[i] == 2u) cells[i] = 0u;
        return;
    }
    if (uType == 58) {                                    // geyser: while erupting, mark EMPTY next to a geyser(44) -> steam (frame-hashed)
        int i = y * uSW + x; uint r = 0u;
        if ((uint(uFrame) % 150u) < 30u) {                // global eruption window
            bool src = cells[i-1]==44u||cells[i+1]==44u||cells[i-uSW]==44u||cells[i+uSW]==44u;
            uint h = (uint(x)*151u + uint(y)*47u + uint(uFrame)*199u) & 0xFFu;
            if (cells[i] == 0u && src && h < 90u) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 59) {                                    // geyser: apply (1 -> steam 8)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 8u;
        return;
    }
    if (uType == 60) {                                    // lye: mark LYE(45) touching acid(11) -> 1 (salt), ACID touching lye -> 2 (water)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 45u) r = (cells[i-1]==11u||cells[i+1]==11u||cells[i-uSW]==11u||cells[i+uSW]==11u) ? 1u : 0u;
        else if (c == 11u) r = (cells[i-1]==45u||cells[i+1]==45u||cells[i-uSW]==45u||cells[i+uSW]==45u) ? 2u : 0u;
        moved[i] = r;
        return;
    }
    if (uType == 61) {                                    // lye: apply (1 -> salt 24, 2 -> water 3)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 24u; else if (moved[i] == 2u) cells[i] = 3u;
        return;
    }
    if (uType == 62) {                                    // sodium: mark SODIUM(46) touching water(3) or fire/lava -> 1
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 46u) {
            bool trig = cells[i-1]==3u||cells[i+1]==3u||cells[i-uSW]==3u||cells[i+uSW]==3u
                      ||cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                      ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            r = trig ? 1u : 0u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 63) {                                    // sodium: apply (marked -> fire 6; water beside a mark -> steam 8)
        int i = y * uSW + x;
        if (moved[i] == 1u) { cells[i] = 6u; return; }
        if (cells[i] == 3u && (moved[i-1]==1u||moved[i+1]==1u||moved[i-uSW]==1u||moved[i+uSW]==1u)) cells[i] = 8u;
        return;
    }
    if (uType == 64) {                                    // coral: mark WATER(3) with exactly one coral(47) of 8 (frame-hashed) -> 1; coral by fire/lava -> 2 (ash)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 3u) {
            int n = int(cells[i-1]==47u)+int(cells[i+1]==47u)+int(cells[i-uSW]==47u)+int(cells[i+uSW]==47u)
                  + int(cells[i-uSW-1]==47u)+int(cells[i-uSW+1]==47u)+int(cells[i+uSW-1]==47u)+int(cells[i+uSW+1]==47u);
            uint h = (uint(x)*151u + uint(y)*101u + uint(uFrame)*181u) & 0xFFu;
            if (n == 1 && h < 8u) r = 1u;
        } else if (cells[i] == 47u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            if (hot) r = 2u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 65) {                                    // coral: apply (1 -> coral 47, 2 -> ash 17)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 47u; else if (moved[i] == 2u) cells[i] = 17u;
        return;
    }
    if (uType == 66) {                                    // phosphorus: mark PHOSPHORUS(48) next to fire/lava (instant) or air (frame-hashed) -> 1
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 48u) {
            bool hot = cells[i-1]==6u||cells[i-1]==7u||cells[i+1]==6u||cells[i+1]==7u
                     ||cells[i-uSW]==6u||cells[i-uSW]==7u||cells[i+uSW]==6u||cells[i+uSW]==7u;
            bool air = cells[i-1]==0u||cells[i+1]==0u||cells[i-uSW]==0u||cells[i+uSW]==0u;
            uint h = (uint(x)*199u + uint(y)*113u + uint(uFrame)*173u) & 0xFFu;
            if (hot || (air && h < 50u)) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 67) {                                    // phosphorus: apply (1 -> fire 6)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 6u;
        return;
    }
    if (uType == 68) {                                    // cement: mark supported CEMENT(49) (cell below non-empty) whose frame-hash fires -> 1
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 49u && cells[i+uSW] != 0u) {
            uint h = (uint(x)*61u + uint(y)*157u + uint(uFrame)*97u) & 0xFFu;
            if (h < 6u) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 69) {                                    // cement: apply (1 -> wall 1)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 1u;
        return;
    }
    if (uType == 70) {                                    // chlorine: mark fate (1 -> SALT, 2 -> EMPTY)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        uint n0=cells[i-1], n1=cells[i+1], n2=cells[i-uSW], n3=cells[i+uSW];
        if (c == 50u) {                                   // CHLORINE
            if (n0==46u||n1==46u||n2==46u||n3==46u) r = 1u;   // + SODIUM -> SALT
            else if (n0==10u||n0==36u||n0==47u||n1==10u||n1==36u||n1==47u||n2==10u||n2==36u||n2==47u||n3==10u||n3==36u||n3==47u) r = 2u;  // bleach PLANT/MOSS/CORAL
            else { uint h=(uint(x)*83u+uint(y)*211u+uint(uFrame)*67u)&0xFFu; if (h<4u) r=2u; }  // disperse
        } else if (c == 46u) {                            // SODIUM next to chlorine -> SALT
            if (n0==50u||n1==50u||n2==50u||n3==50u) r = 1u;
        } else if (c==10u||c==36u||c==47u) {              // PLANT/MOSS/CORAL next to chlorine -> bleached
            if (n0==50u||n1==50u||n2==50u||n3==50u) r = 2u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 71) {                                    // chlorine: apply (1 -> salt 24, 2 -> empty 0)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 24u; else if (moved[i] == 2u) cells[i] = 0u;
        return;
    }
    if (uType == 72) {                                    // battery: mark WIRE(38) touching battery(51) on a pulse frame -> 1
        int i = y * uSW + x; uint r = 0u;
        if ((uint(uFrame) % 12u) == 0u && cells[i] == 38u &&
            (cells[i-1]==51u||cells[i+1]==51u||cells[i-uSW]==51u||cells[i+uSW]==51u)) r = 1u;
        moved[i] = r;
        return;
    }
    if (uType == 73) {                                    // battery: apply (1 -> ehead 39)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 39u;
        return;
    }
    if (uType == 74) {                                    // fuse: mark FUSE(52) catching (-> 1 = burnfuse) or BURNFUSE(53) burning out (-> 2 = fire)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 52u) {
            uint n0=cells[i-1], n1=cells[i+1], n2=cells[i-uSW], n3=cells[i+uSW];
            bool lit = n0==6u||n0==7u||n0==32u||n0==53u||n1==6u||n1==7u||n1==32u||n1==53u
                     ||n2==6u||n2==7u||n2==32u||n2==53u||n3==6u||n3==7u||n3==32u||n3==53u;
            if (lit) r = 1u;
        } else if (c == 53u) { r = 2u; }
        moved[i] = r;
        return;
    }
    if (uType == 75) {                                    // fuse: apply (1 -> burnfuse 53, 2 -> fire 6)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 53u; else if (moved[i] == 2u) cells[i] = 6u;
        return;
    }
    if (uType == 76) {                                    // cryo: mark fate (1 -> ICE, 2 -> EMPTY, 3 -> OBSIDIAN)
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        uint n0=cells[i-1], n1=cells[i+1], n2=cells[i-uSW], n3=cells[i+uSW];
        bool nearCryo = n0==54u||n1==54u||n2==54u||n3==54u;
        if (c == 54u) {
            bool hot = n0==6u||n0==7u||n1==6u||n1==7u||n2==6u||n2==7u||n3==6u||n3==7u;
            uint h=(uint(x)*109u+uint(y)*233u+uint(uFrame)*47u)&0xFFu;
            if (hot || h<3u) r = 2u;
        } else if (c == 3u) { if (nearCryo) r = 1u; }
        else if (c == 6u) { if (nearCryo) r = 2u; }
        else if (c == 7u) { if (nearCryo) r = 3u; }
        moved[i] = r;
        return;
    }
    if (uType == 77) {                                    // cryo: apply (1 -> ice 14, 2 -> empty 0, 3 -> obsidian 23)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 14u; else if (moved[i] == 2u) cells[i] = 0u; else if (moved[i] == 3u) cells[i] = 23u;
        return;
    }
    if (uType == 78) {                                    // lamp: mark LAMP(55) lit by adjacent electron -> 1, LAMPLIT(56) no longer beside one -> 2
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        bool nearE = cells[i-1]==39u||cells[i+1]==39u||cells[i-uSW]==39u||cells[i+uSW]==39u
                   ||cells[i-1]==40u||cells[i+1]==40u||cells[i-uSW]==40u||cells[i+uSW]==40u;
        if (c == 55u) { if (nearE) r = 1u; }
        else if (c == 56u) { if (!nearE) r = 2u; }
        moved[i] = r;
        return;
    }
    if (uType == 79) {                                    // lamp: apply (1 -> lamplit 56, 2 -> lamp 55)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 56u; else if (moved[i] == 2u) cells[i] = 55u;
        return;
    }
    if (uType == 80) {                                    // petrify: mark living(10/9/36/47) beside petrify(57) -> 1, petrify -> 2
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        if (c == 57u) { r = 2u; }
        else if (c==10u||c==9u||c==36u||c==47u) {
            if (cells[i-1]==57u||cells[i+1]==57u||cells[i-uSW]==57u||cells[i+uSW]==57u) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 81) {                                    // petrify: apply (1 -> petrify 57, 2 -> obsidian 23)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 57u; else if (moved[i] == 2u) cells[i] = 23u;
        return;
    }
    if (uType == 82) {                                    // firework: tag rocket rising(1)/waiting(0)/bursting(2)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 58u) {
            uint h = (uint(x)*71u + uint(y)*251u + uint(uFrame)*139u) & 0xFFu;
            if (h < 8u) r = 2u;
            else if (cells[i-uSW] == 0u) r = 1u;
            else if (cells[i-uSW] == 58u) r = 0u;
            else r = 2u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 83) {                                    // firework: advance risers, detonate bursters, spray
        int i = y * uSW + x;
        if (moved[i] == 2u) { cells[i] = 6u; return; }
        if (moved[i] == 1u) { cells[i] = 0u; return; }
        if (cells[i] == 0u) {
            if (moved[i+uSW] == 1u) cells[i] = 58u;
            else if (moved[i-1]==2u||moved[i+1]==2u||moved[i-uSW]==2u||moved[i+uSW]==2u) cells[i] = 6u;
        }
        return;
    }
    if (uType == 84) {                                    // sprout: tag tip climbing(1)/waiting(0)/crowning(2)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 60u) {
            uint h = (uint(x)*167u + uint(y)*59u + uint(uFrame)*233u) & 0xFFu;
            if (h < 14u) r = 2u;
            else if (cells[i-uSW] == 0u) r = 1u;
            else if (cells[i-uSW] == 60u) r = 0u;
            else r = 2u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 85) {                                    // sprout: trunk(9)/crown(10); the tip climbs / canopy unfurls
        int i = y * uSW + x;
        if (moved[i] == 2u) { cells[i] = 10u; return; }
        if (moved[i] == 1u) { cells[i] = 9u; return; }
        if (cells[i] == 0u) {
            if (moved[i+uSW] == 1u) cells[i] = 60u;
            else if (moved[i-1]==2u||moved[i+1]==2u||moved[i-uSW]==2u||moved[i+uSW]==2u) cells[i] = 10u;
        }
        return;
    }
    if (uType == 86) {                                    // conveyor: mark (254 leaving / material-id arriving / 255 no change)
        int i = y * uSW + x; uint c = cells[i]; uint r = 255u;
        if (c == 0u) {
            uint cl = cells[i-1];
            if (cl != 0u && cl != 1u && cl != 61u && cells[i-1+uSW] == 61u) r = cl;   // arrives from the left
        } else if (c != 1u && c != 61u && cells[i+uSW] == 61u && cells[i+1] == 0u) {
            r = 254u;                                                                  // rides off to the right
        }
        moved[i] = r;
        return;
    }
    if (uType == 87) {                                    // conveyor: apply
        int i = y * uSW + x; uint r = moved[i];
        if (r == 254u) cells[i] = 0u;
        else if (r != 255u) cells[i] = r;
        return;
    }
    if (uType == 88) {                                    // magnet: mark IRON(63) touching a MAGNET(62) -> 1
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 63u && (cells[i-1]==62u||cells[i+1]==62u||cells[i-uSW]==62u||cells[i+uSW]==62u)) r = 1u;
        moved[i] = r;
        return;
    }
    if (uType == 89) {                                    // magnet: apply (1 -> magnet 62)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 62u;
        return;
    }
    if (uType == 90) {                                    // rust: mark IRON(63) corroding in water/acid -> 1, RUST(65) smelting near fire/lava -> 2
        int i = y * uSW + x; uint c = cells[i]; uint r = 0u;
        uint n0=cells[i-1], n1=cells[i+1], n2=cells[i-uSW], n3=cells[i+uSW];
        uint h = (uint(x)*131u + uint(y)*79u + uint(uFrame)*197u) & 0xFFu;
        if (c == 63u) {
            bool wet = n0==3u||n1==3u||n2==3u||n3==3u||n0==11u||n1==11u||n2==11u||n3==11u;
            if (wet && h < 4u) r = 1u;
        } else if (c == 65u) {
            bool hot = n0==6u||n0==7u||n1==6u||n1==7u||n2==6u||n2==7u||n3==6u||n3==7u;
            if (hot && h < 4u) r = 2u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 91) {                                    // rust: apply (1 -> rust 65, 2 -> iron 63)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 65u; else if (moved[i] == 2u) cells[i] = 63u;
        return;
    }
    if (uType == 92) {                                    // seed: mark SEED(66) grounded + beside WATER -> 1 (germinate)
        int i = y * uSW + x; uint r = 0u;
        if (cells[i] == 66u && cells[i+uSW] != 0u) {
            uint n0=cells[i-1], n1=cells[i+1], n2=cells[i-uSW], n3=cells[i+uSW];
            bool wet = n0==3u||n1==3u||n2==3u||n3==3u;
            uint h = (uint(x)*89u + uint(y)*197u + uint(uFrame)*151u) & 0xFFu;
            if (wet && h < 8u) r = 1u;
        }
        moved[i] = r;
        return;
    }
    if (uType == 93) {                                    // seed: apply (1 -> sprout 60)
        int i = y * uSW + x;
        if (moved[i] == 1u) cells[i] = 60u;
        return;
    }
    if (uType == 94) {                                    // laser: mark BEAM(68)->1, LASER(67)->2, flammable->3
        int i = y * uSW + x; uint c = cells[i];
        bool burn = c==9u||c==10u||c==5u||c==4u||c==30u||c==36u||c==37u||c==31u||c==66u;
        moved[i] = (c == 68u) ? 1u : (c == 67u) ? 2u : (burn ? 3u : 0u);
        return;
    }
    if (uType == 95) {                                    // laser: apply -- beam travels right, fed from the left
        int i = y * uSW + x; uint left = moved[i-1];
        bool fed = (left == 1u || left == 2u);
        if (cells[i] == 68u)      cells[i] = fed ? 68u : 0u;   // hold the ray only while fed
        else if (cells[i] == 0u)  { if (fed) cells[i] = 68u; } // extend / emit into empty
        else if (moved[i] == 3u)  { if (fed) cells[i] = 6u; }  // burn flammable -> FIRE
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
    if (m == 24u) return vec3(0.929, 0.929, 0.878);
    if (m == 25u) return vec3(0.918, 0.957, 1.0);
    if (m == 26u) return vec3(0.769, 0.784, 0.831);
    if (m == 27u) return vec3(0.227, 0.227, 0.251);
    if (m == 28u) return vec3(0.541, 0.227, 0.122);
    if (m == 29u) return vec3(0.682, 0.941, 1.000);
    if (m == 30u) return vec3(0.620, 0.961, 0.710);
    if (m == 31u) return vec3(0.149, 0.133, 0.118);
    if (m == 32u) return vec3(0.800, 0.267, 0.067);
    if (m == 33u) return vec3(0.604, 0.251, 0.902);
    if (m == 34u) return vec3(0.251, 0.878, 0.753);
    if (m == 35u) return vec3(0.804, 0.627, 1.000);
    if (m == 36u) return vec3(0.431, 0.545, 0.239);
    if (m == 37u) return vec3(0.796, 0.780, 0.353);
    if (m == 38u) return vec3(0.784, 0.525, 0.180);
    if (m == 39u) return vec3(0.502, 0.878, 1.000);
    if (m == 40u) return vec3(0.227, 0.416, 0.690);
    if (m == 41u) return vec3(0.847, 0.565, 0.125);
    if (m == 42u) return vec3(0.690, 0.878, 0.251);
    if (m == 43u) return vec3(0.314, 1.000, 0.565);
    if (m == 44u) return vec3(0.314, 0.565, 0.627);
    if (m == 45u) return vec3(0.784, 0.910, 0.816);
    if (m == 46u) return vec3(0.843, 0.816, 0.690);
    if (m == 47u) return vec3(1.000, 0.549, 0.412);
    if (m == 48u) return vec3(0.937, 0.910, 0.627);
    if (m == 49u) return vec3(0.494, 0.549, 0.600);
    if (m == 50u) return vec3(0.714, 0.878, 0.227);
    if (m == 51u) return vec3(1.000, 0.800, 0.133);
    if (m == 52u) return vec3(0.604, 0.502, 0.314);
    if (m == 53u) return vec3(1.000, 0.816, 0.188);
    if (m == 54u) return vec3(0.533, 0.816, 0.973);
    if (m == 55u) return vec3(0.290, 0.251, 0.188);
    if (m == 56u) return vec3(1.000, 0.941, 0.627);
    if (m == 57u) return vec3(0.690, 0.596, 0.659);
    if (m == 58u) return vec3(1.000, 0.314, 0.753);
    if (m == 59u) return vec3(0.690, 0.376, 1.000);
    if (m == 60u) return vec3(0.439, 0.847, 0.220);
    if (m == 61u) return vec3(0.271, 0.298, 0.314);
    if (m == 62u) return vec3(0.345, 0.471, 0.722);
    if (m == 63u) return vec3(0.471, 0.502, 0.533);
    if (m == 64u) return vec3(0.784, 0.878, 0.439);
    if (m == 65u) return vec3(0.659, 0.314, 0.125);
    if (m == 66u) return vec3(0.710, 0.514, 0.180);
    if (m == 67u) return vec3(0.565, 0.094, 0.094);
    if (m == 68u) return vec3(1.0, 0.188, 0.188);
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
            for (int t = 3; t <= 95; ++t) {         // + ... magnet, rust, seed, laser
                if (!passEnabled(t)) continue;      // skip a paint-only reaction whose material is absent
                glUniform1i(lType, t);
                glDispatchCompute(LW / 16, LH / 16, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }
        ++frame;
        gpuAhead = true;
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
        if (material == FIRE || material == LAVA || material == STEAM || material == PLANT || material == ACID || material == SMOKE || material == ICE || material == SPRING || material == VOLCANO || material == VOID || material == WATER || material == VIRUS || material == SPARK || material == SALT || material == FROST || material == EMBER || material == CLONER || material == CRYSTAL || material == ANTIMATTER || material == MOSS || material == EHEAD || material == ETAIL || material == PHOSPHORUS || material == CEMENT || material == CHLORINE || material == BATTERY || material == BURNFUSE || material == CRYO || material == LAMPLIT || material == PETRIFY || material == FIREWORK || material == SPROUT || material == BELT || material == MAGNET || material == LASER || material == BEAM) hasReactive = true;
        present[material] = true;
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
    bool hasReactive = false;            // gates the reaction dispatches (any reactive material)
    // Per-material "ever resident" latch -> skip the dispatch for a paint-only reaction whose
    // trigger material is absent (a guaranteed no-op). Set, never cleared; safe (a stale flag
    // only costs an extra no-op dispatch). Mirrors the CPU build's gate. See passEnabled().
    bool present[MATERIAL_COUNT] = {false};
    // A gated reaction's two pass types run only while its trigger material(s) can be present.
    bool passEnabled(int t) const {
        switch (t) {
            case 18: case 19: return present[SPRING];
            case 20: case 21: return present[TNT] || present[GUNPOWDER] || present[NITRO];
            case 22: case 23: return present[VOLCANO];
            case 24: case 25: return present[VOID];
            case 28: case 29: return present[VIRUS];
            case 30: case 31: return present[SPARK];
            case 32: case 33: return present[SALT] || present[LYE] || present[CHLORINE];  // LYE/CHLORINE make SALT
            case 34: case 35: return present[MERCURY];
            case 36: case 37: return present[THERMITE];
            case 38: case 39: return present[FROST];
            case 40: case 41: return present[COAL] || present[EMBER];
            case 42: case 43: return present[CLONER];
            case 44: case 45: return present[CRYSTAL];
            case 46: case 47: return present[ANTIMATTER];
            case 48: case 49: return present[MOSS];
            case 50: case 51: return present[EHEAD] || present[ETAIL] || present[SENSOR] || present[BATTERY];  // sensor/battery can create electrons
            case 54: case 55: return present[SENSOR];
            case 56: case 57: return present[LIFE];
            case 58: case 59: return present[GEYSER];
            case 60: case 61: return present[LYE];
            case 62: case 63: return present[SODIUM];
            case 64: case 65: return present[CORAL];
            case 66: case 67: return present[PHOSPHORUS];
            case 68: case 69: return present[CEMENT];
            case 70: case 71: return present[CHLORINE];
            case 72: case 73: return present[BATTERY];
            case 74: case 75: return present[FUSE] || present[BURNFUSE];
            case 76: case 77: return present[CRYO];
            case 78: case 79: return present[LAMP] || present[LAMPLIT];
            case 80: case 81: return present[PETRIFY];
            case 82: case 83: return present[FIREWORK];
            case 84: case 85: return present[SPROUT] || present[SEED];
            case 86: case 87: return present[BELT];
            case 88: case 89: return present[MAGNET];
            case 90: case 91: return present[IRON] || present[RUST];
            case 92: case 93: return present[SEED];
            case 94: case 95: return present[LASER] || present[BEAM];
            case 52: case 53: return present[IGNITER];
            default: return true;   // 3-17, 26-27: always-on core reactions
        }
    }
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
                present[v] = true;
                if (v == FIRE || v == LAVA || v == STEAM || v == PLANT || v == ACID || v == SMOKE || v == ICE || v == SPRING || v == VOLCANO || v == VOID || v == WATER || v == VIRUS || v == SPARK || v == SALT || v == FROST || v == EMBER || v == CLONER || v == CRYSTAL || v == ANTIMATTER || v == MOSS || v == EHEAD || v == ETAIL || v == PHOSPHORUS || v == CEMENT || v == CHLORINE || v == BATTERY || v == BURNFUSE || v == CRYO || v == LAMPLIT || v == PETRIFY || v == FIREWORK || v == SPROUT || v == BELT || v == MAGNET || v == LASER || v == BEAM) hasReactive = true;
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
    // Swatch slot i IS material i (the present shader colours slot i with
    // matColor(i)), so the palette is driven entirely by MATERIAL_COUNT.
    ui::Palette pal = ui::palette(renderW, MATERIAL_COUNT);
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
        if (glfwGetKey(win, GLFW_KEY_L) == GLFW_PRESS) current = SALT;
        if (glfwGetKey(win, GLFW_KEY_N) == GLFW_PRESS) current = SNOW;
        if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) current = MERCURY;
        if (glfwGetKey(win, GLFW_KEY_B) == GLFW_PRESS) current = GUNPOWDER;
        if (glfwGetKey(win, GLFW_KEY_K) == GLFW_PRESS) current = THERMITE;
        if (glfwGetKey(win, GLFW_KEY_F) == GLFW_PRESS) current = FROST;
        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) current = WISP;
        if (glfwGetKey(win, GLFW_KEY_C) == GLFW_PRESS) current = COAL;
        if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) current = EMBER;
        if (glfwGetKey(win, GLFW_KEY_U) == GLFW_PRESS) current = CLONER;
        if (glfwGetKey(win, GLFW_KEY_Y) == GLFW_PRESS) current = CRYSTAL;
        if (glfwGetKey(win, GLFW_KEY_J) == GLFW_PRESS) current = ANTIMATTER;
        if (glfwGetKey(win, GLFW_KEY_SEMICOLON) == GLFW_PRESS) current = MOSS;
        if (glfwGetKey(win, GLFW_KEY_COMMA) == GLFW_PRESS) current = FUMES;
        if (glfwGetKey(win, GLFW_KEY_PERIOD) == GLFW_PRESS) current = WIRE;
        if (glfwGetKey(win, GLFW_KEY_SLASH) == GLFW_PRESS) current = EHEAD;
        if (glfwGetKey(win, GLFW_KEY_APOSTROPHE) == GLFW_PRESS) current = ETAIL;
        if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS) current = IGNITER;
        if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS) current = SENSOR;
        if (glfwGetKey(win, GLFW_KEY_BACKSLASH) == GLFW_PRESS) current = LIFE;
        if (glfwGetKey(win, GLFW_KEY_GRAVE_ACCENT) == GLFW_PRESS) current = GEYSER;
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
            if (h >= 0) current = (uint8_t)h;            // clicked a palette swatch (slot == material id)
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
        glUniform1i(uPalSel, (int)current);            // selected slot == selected material id
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
