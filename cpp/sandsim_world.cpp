/*
 * sandsim - multi-material falling-sand world (CPU SIMD)
 *
 * A huge world (chunked, streamed to/from disk around a camera) simulated on one
 * contiguous, WALL-bordered live grid with the single-grid SIMD technique. The
 * update rule lives in simd_core.h and is order-independent (disjoint even/odd
 * passes), so it produces a bit-identical world on CPU SIMD and the GPU compute
 * backends. The SIMD width is chosen at runtime: AVX2 (32 lanes) if the CPU has
 * it, otherwise SSE4.1 (16 lanes) -- see world_step.h. Both give the same result.
 *
 * The live window is sized at runtime: the interactive view renders each cell as
 * a "virtual pixel" of SCALE x SCALE screen pixels, so the resident region is
 * (winW/SCALE) x (winH/SCALE) cells. Window resolution and scale are configurable
 * (--res WxH / --scale N, or SANDSIM_RES / SANDSIM_SCALE; default 1024x768, 2x2).
 *
 * Modes:
 *   (default)                 SDL2 window; arrows pan the camera by a chunk,
 *                             number keys pick a material, left mouse paints.
 *   --bench [steps] [wch] [hch]   headless streaming benchmark (fixed 4x4 live
 *                             window; whole-world checksum + conserved counts).
 *   --ppm <file> [steps]      render a snapshot of one live window.
 */

#include "materials.h"   // Material enum
#include "world_step.h"  // runtime SSE/AVX2 step dispatch
#include "../ui.h"       // on-screen material palette
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <SDL2/SDL.h>

static StepFn g_step = nullptr;   // selected at startup (AVX2 or SSE)
static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu, 0xFF8E44ADu, 0xFFFF5A1Eu, 0xFFCF1B0Bu, 0xFFDCE4ECu, 0xFF8B5A2Bu, 0xFF3AA84Au, 0xFFB8F000u, 0xFF585860u, 0xFFAEE0E8u, 0xFFCDEBFFu, 0xFF1FB5C4u, 0xFFCC2222u, 0xFF6B6358u, 0xFF402A28u, 0xFF3C1452u, 0xFF4E3B24u, 0xFFD81E9Bu, 0xFFFAF080u, 0xFF2A2438u, 0xFFEDEDE0u, 0xFFEAF4FFu, 0xFFC4C8D4u, 0xFF3A3A40u, 0xFF8A3A1Fu, 0xFFAEF0FFu, 0xFF9EF5B5u, 0xFF26221Eu, 0xFFCC4411u, 0xFF9A40E6u, 0xFF40E0C0u, 0xFFCDA0FFu, 0xFF6E8B3Du, 0xFFCBC75Au, 0xFFC8862Eu, 0xFF80E0FFu, 0xFF3A6AB0u, 0xFFD89020u, 0xFFB0E040u, 0xFF50FF90u, 0xFF5090A0u, 0xFFC8E8D0u, 0xFFD7D0B0u, 0xFFFF8C69u, 0xFFEFE8A0u, 0xFF7E8C99u, 0xFFB6E03Au, 0xFFFFCC22u, 0xFF9A8050u, 0xFFFFD030u, 0xFF88D0F8u, 0xFF4A4030u, 0xFFFFF0A0u, 0xFFB098A8u, 0xFFFF50C0u, 0xFFB060FFu, 0xFF70D838u, 0xFF454C50u, 0xFF5878B8u, 0xFF788088u, 0xFFC8E070u, 0xFFA85020u, 0xFFB5832Eu, 0xFF901818u, 0xFFFF3030u, 0xFFE8F8FFu,
};

// Display names for the HUD/tooltips -- one per material id, in enum order.
static const char* kNames[MATERIAL_COUNT] = {
    "ERASER", "WALL", "SAND", "WATER", "GAS", "OIL", "FIRE", "LAVA", "STEAM", "WOOD",
    "PLANT", "ACID", "SMOKE", "GLASS", "ICE", "SPRING", "TNT", "ASH", "VOLCANO", "VOID",
    "MUD", "VIRUS", "SPARK", "OBSIDIAN", "SALT", "SNOW", "MERCURY", "GUNPOWDER", "THERMITE", "FROST",
    "WISP", "COAL", "EMBER", "CLONER", "CRYSTAL", "ANTIMATTER", "MOSS", "FUMES", "WIRE", "E-HEAD",
    "E-TAIL", "IGNITER", "SENSOR", "LIFE", "GEYSER", "LYE", "SODIUM", "CORAL", "PHOSPHORUS", "CEMENT",
    "CHLORINE", "BATTERY", "FUSE", "BURN-FUSE", "CRYO", "LAMP", "LAMP-LIT", "PETRIFY", "FIREWORK", "LEVITON",
    "SPROUT", "BELT", "MAGNET", "IRON", "NITRO", "RUST", "SEED", "LASER", "BEAM", "ICICLE",
};

// The palette is laid out grouped by category (a navigable toolbox) rather than in raw
// id order. kPaletteOrder is the swatch order (material ids); kSlotCat[i] is the category
// of swatch i; each category has a name and an accent colour drawn under its swatches.
static const char* kCatNames[] = {
    "TOOLS", "SOLIDS", "POWDERS", "LIQUIDS", "GASES", "FIRE & HEAT",
    "EXPLOSIVES", "LIFE & GROWTH", "CIRCUITS", "MACHINES", "SOURCES & MAGIC",
};
static const uint32_t kCatAccent[] = {
    0xFF8890A0u, 0xFF6E6E78u, 0xFFC8A060u, 0xFF4488FFu, 0xFFA6C2D2u, 0xFFFF6633u,
    0xFFFF3344u, 0xFF44C060u, 0xFFFFCC22u, 0xFFB060FFu, 0xFF40E0C0u,
};
static const uint8_t kPaletteOrder[MATERIAL_COUNT] = {
    EMPTY,                                                                       // tools
    WALL, WOOD, GLASS, ICE, OBSIDIAN,                                            // solids
    SAND, ASH, SALT, SNOW, MUD, COAL, LYE, CEMENT, IRON, RUST, LEVITON, SEED,    // powders
    WATER, OIL, ACID, MERCURY, CRYO,                                            // liquids
    GAS, STEAM, SMOKE, WISP, FUMES, CHLORINE,                                   // gases
    FIRE, LAVA, EMBER, FROST,                                                   // fire & heat
    TNT, GUNPOWDER, THERMITE, SODIUM, PHOSPHORUS, NITRO, FUSE, BURNFUSE,        // explosives
    PLANT, CRYSTAL, MOSS, CORAL, SPROUT, ICICLE, VIRUS, LIFE,                   // life & growth
    SPARK, WIRE, EHEAD, ETAIL, IGNITER, SENSOR, BATTERY, LAMP, LAMPLIT,         // circuits
    CLONER, BELT, MAGNET, LASER, BEAM, ANTIMATTER,                             // machines
    SPRING, VOLCANO, GEYSER, VOID, PETRIFY, FIREWORK,                          // sources & magic
};
static const uint8_t kSlotCat[MATERIAL_COUNT] = {
    0,
    1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,
    4,4,4,4,4,4,
    5,5,5,5,
    6,6,6,6,6,6,6,6,
    7,7,7,7,7,7,7,7,
    8,8,8,8,8,8,8,8,8,
    9,9,9,9,9,9,
    10,10,10,10,10,10,
};

// Render-only: how brightly each material glows. Emissive cells cast a soft additive
// bloom into their surroundings so fire, lava, lasers and lamps light up the scene.
static float emissionStrength(uint8_t m) {
    switch (m) {
        case FIRE:       return 1.00f;
        case BEAM:       return 1.00f;
        case SPARK:      return 0.90f;
        case LAVA:       return 0.85f;
        case LAMPLIT:    return 0.80f;
        case EMBER:      return 0.70f;
        case FIREWORK:   return 0.70f;
        case BURNFUSE:   return 0.70f;
        case LASER:      return 0.60f;
        case ANTIMATTER: return 0.60f;
        case EHEAD:      return 0.50f;
        default:         return 0.00f;
    }
}

static constexpr int CHUNK = 64;   // simulation chunk = 64x64 cells
static constexpr int PAD = 16;     // WALL border / SIMD halo

// Window resolution + virtual-pixel scale + simulation rate for the interactive
// view. simHz is steps/second, decoupled from the render rate so the physics runs
// at the same wall-clock speed on every backend.
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

class SimdWorld {
public:
    // gw x gh chunks resident (the live window); the world is wbox x hbox chunks.
    SimdWorld(int gw, int gh, int wbox, int hbox, std::string dir)
        : gw(gw), gh(gh), LW(gw * CHUNK), LH(gh * CHUNK), SW(LW + 2 * PAD), SH(LH + 2 * PAD),
          X0(PAD), X1(PAD + LW), Y0(PAD), Y1(PAD + LH),
          wbox(wbox), hbox(hbox), dir(std::move(dir)) {
        std::filesystem::create_directories(this->dir);
        grid.assign((size_t)SW * SH, WALL);     // everything starts solid (border stays WALL)
        moved.assign((size_t)SW * SH, 0);
    }

    int winChunksW() const { return gw; }
    int winChunksH() const { return gh; }
    int cellsW() const { return LW; }
    int cellsH() const { return LH; }

    void generateAllToDisk() {
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) { genBox(cx, cy, buf); writeBox(cx, cy, buf); }
    }

    // Make the window's top-left chunk (camCx,camCy); save the old window and
    // load the new one into the contiguous interior.
    void setWindow(int camCx, int camCy) {
        if (windowValid && camCx == winCx && camCy == winCy) return;
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
    }

    void step() {
        g_step(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);
        if (hasReactive) {                                              // moved = scratch buffer
            decayFire(grid.data(), SW, X0, X1, Y0, Y1, frame);
            igniteFire(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);
            quench(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);
            growPlant(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);
            dissolveAcid(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);
            makeGlass(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);
            meltIce(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);
            freezeWater(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);
            if (present[SPRING])   emitSpring(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);
            if (present[TNT] || present[GUNPOWDER] || present[NITRO]) detonateTnt(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);
            if (present[VOLCANO])  emitVolcano(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);  // pass 22/23
            if (present[VOID])     consumeVoid(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);         // pass 24/25
            mudCycle(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);     // pass 26/27 (sand pervasive: always on)
            if (present[VIRUS])    spreadVirus(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);  // pass 28/29
            if (present[SPARK])    arcSpark(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);            // pass 30/31
            if (present[SALT] || present[LYE] || present[CHLORINE]) saltCycle(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame); // pass 32/33 (LYE/CHLORINE make SALT)
            if (present[MERCURY])  poisonMercury(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame); // pass 34/35
            if (present[THERMITE]) burnThermite(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);        // pass 36/37
            if (present[FROST])    spreadFrost(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);         // pass 38/39
            if (present[COAL] || present[EMBER]) smoulderCoal(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame); // pass 40/41
            if (present[CLONER])   cloneMaterial(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);       // pass 42/43
            if (present[CRYSTAL])  growCrystal(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);  // pass 44/45
            if (present[ANTIMATTER]) annihilate(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);        // pass 46/47
            if (present[MOSS])     growMoss(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);     // pass 48/49
            if (present[EHEAD] || present[ETAIL] || present[SENSOR] || present[BATTERY]) wireWorld(grid.data(), moved.data(), SW, X0, X1, Y0, Y1); // pass 50/51 (SENSOR/BATTERY can create electrons)
            if (present[IGNITER])  fireIgniter(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);         // pass 52/53
            if (present[SENSOR])   senseWorld(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);          // pass 54/55
            if (present[LIFE])     conwayLife(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);          // pass 56/57
            if (present[GEYSER])   eruptGeyser(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);  // pass 58/59
            if (present[LYE])      neutraliseLye(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);       // pass 60/61
            if (present[SODIUM])   reactSodium(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);         // pass 62/63
            if (present[CORAL])    growCoral(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);    // pass 64/65
            if (present[PHOSPHORUS]) ignitePhosphorus(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame); // pass 66/67
            if (present[CEMENT])   hardenCement(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);     // pass 68/69
            if (present[CHLORINE]) reactChlorine(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);    // pass 70/71
            if (present[BATTERY])  emitBattery(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);      // pass 72/73
            if (present[FUSE] || present[BURNFUSE]) burnFuse(grid.data(), moved.data(), SW, X0, X1, Y0, Y1); // pass 74/75
            if (present[CRYO])     reactCryo(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);        // pass 76/77
            if (present[LAMP] || present[LAMPLIT]) lampLogic(grid.data(), moved.data(), SW, X0, X1, Y0, Y1); // pass 78/79
            if (present[PETRIFY])  petrify(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);                 // pass 80/81
            if (present[FIREWORK]) launchFirework(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);   // pass 82/83
            if (present[SPROUT] || present[SEED]) growTree(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame); // pass 84/85 (SEED germinates into SPROUT)
            if (present[BELT])     runConveyor(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);             // pass 86/87
            if (present[MAGNET])   magnetise(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);               // pass 88/89
            if (present[IRON] || present[RUST]) rustCycle(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame); // pass 90/91
            if (present[SEED])     germinateSeed(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);     // pass 92/93
            if (present[LASER] || present[BEAM]) laserBeam(grid.data(), moved.data(), SW, X0, X1, Y0, Y1);  // pass 94/95
            if (present[ICICLE])   growIcicle(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame);        // pass 96/97
        }
        ++frame;
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
        if (material == FIRE || material == LAVA || material == STEAM || material == PLANT || material == ACID || material == SMOKE || material == ICE || material == SPRING || material == VOLCANO || material == VOID || material == WATER || material == VIRUS || material == SPARK || material == SALT || material == FROST || material == EMBER || material == CLONER || material == CRYSTAL || material == ANTIMATTER || material == MOSS || material == EHEAD || material == ETAIL || material == SENSOR || material == LIFE || material == GEYSER || material == PHOSPHORUS || material == CEMENT || material == CHLORINE || material == BATTERY || material == BURNFUSE || material == CRYO || material == LAMPLIT || material == PETRIFY || material == FIREWORK || material == SPROUT || material == BELT || material == MAGNET || material == LASER || material == BEAM || material == ICICLE) hasReactive = true;
        present[material] = true;
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = lx + dx, ny = ly + dy;
                if (nx >= 0 && nx < LW && ny >= 0 && ny < LH && dx * dx + dy * dy <= radius * radius)
                    grid[(size_t)(ny + Y0) * SW + (nx + X0)] = material;
            }
    }
    uint8_t viewCell(int lx, int ly) const { return grid[(size_t)(ly + Y0) * SW + (lx + X0)]; }

    // Wipe the whole resident live region back to empty air (keeps the WALL halo).
    void clearView() {
        for (int y = Y0; y < Y1; ++y)
            for (int x = X0; x < X1; ++x)
                grid[(size_t)y * SW + x] = EMPTY;
    }

    void summary(uint64_t& checksum, uint64_t counts[MATERIAL_COUNT]) {
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
    const int gw, gh;                 // live window in chunks
    const int LW, LH;                 // live window in cells
    const int SW, SH;                 // padded stride / height
    const int X0, X1, Y0, Y1;         // interior cell range
    int wbox, hbox;
    std::string dir;
    std::vector<uint8_t> grid;   // padded contiguous live region
    std::vector<uint8_t> moved;
    int winCx = 0, winCy = 0;
    bool windowValid = false;
    uint32_t frame = 0;
    int residentMax = 0;
    long long nWrites = 0, nReads = 0;
    bool hasReactive = false;      // gates the reaction passes; set when any reactive material enters the grid
    // Per-material "has this ever been resident?" latch. A reaction whose trigger material is
    // neither loaded, painted, nor *created by another reaction* (so its only source is itself
    // or nothing) is a guaranteed no-op while its flag is false, so it can be skipped -- this
    // is the perf gate for the many paint-only reactions (virus, frost, crystal, ...). Set, never
    // cleared: a stale flag only costs an extra (still no-op) pass, never a wrong result.
    bool present[MATERIAL_COUNT] = {false};

    // --- chunk <-> interior, disk -------------------------------------------
    void extractChunk(int cgx, int cgy, std::vector<uint8_t>& out) const {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                out[ly * CHUNK + lx] = grid[(size_t)(Y0 + cgy * CHUNK + ly) * SW + (X0 + cgx * CHUNK + lx)];
    }
    void injectChunk(int cgx, int cgy, const std::vector<uint8_t>& in) {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx) {
                uint8_t v = in[ly * CHUNK + lx];
                present[v] = true;
                if (v == FIRE || v == LAVA || v == STEAM || v == PLANT || v == ACID || v == SMOKE || v == ICE || v == SPRING || v == VOLCANO || v == VOID || v == WATER || v == VIRUS || v == SPARK || v == SALT || v == FROST || v == EMBER || v == CLONER || v == CRYSTAL || v == ANTIMATTER || v == MOSS || v == EHEAD || v == ETAIL || v == SENSOR || v == LIFE || v == GEYSER || v == PHOSPHORUS || v == CEMENT || v == CHLORINE || v == BATTERY || v == BURNFUSE || v == CRYO || v == LAMPLIT || v == PETRIFY || v == FIREWORK || v == SPROUT || v == BELT || v == MAGNET || v == LASER || v == BEAM || v == ICICLE) hasReactive = true;
                grid[(size_t)(Y0 + cgy * CHUNK + ly) * SW + (X0 + cgx * CHUNK + lx)] = v;
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
    bool readBox(int cx, int cy, std::vector<uint8_t>& buf) const {
        std::ifstream f(path(cx, cy), std::ios::binary);
        if (!f) return false;
        f.read((char*)buf.data(), CHUNK * CHUNK);
        const_cast<SimdWorld*>(this)->nReads++;
        return true;
    }
};

// ---------------------------------------------------------------------------
static int runBench(int steps, int wbox, int hbox) {
    const int gw = 4, gh = 4;   // fixed live window for the bit-identical reference
    if (wbox < gw) wbox = gw;
    if (hbox < gh) hbox = gh;
    std::string dir = "/tmp/sandsim_world_simd_" + std::to_string(steps) + "_" +
                      std::to_string(wbox) + "x" + std::to_string(hbox);
    std::filesystem::remove_all(dir);
    SimdWorld world(gw, gh, wbox, hbox, dir);
    world.generateAllToDisk();

    uint64_t startCk, startCnt[MATERIAL_COUNT];
    world.summary(startCk, startCnt);

    int nposX = wbox - gw + 1, nposY = hbox - gh + 1, nWin = nposX * nposY;
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * nWin / steps);
        if (visit >= nWin) visit = nWin - 1;
        int row = visit / nposX, col = visit % nposX;
        world.setWindow((row % 2 == 0) ? col : (nposX - 1 - col), row);
        world.step();
    }
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, cnt[MATERIAL_COUNT];
    world.summary(ck, cnt);
    bool conserved = true;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != startCnt[i]) conserved = false;
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double cells = (double)world.cellsW() * world.cellsH() * steps;
    double mc = (ms > 0.0) ? cells / (ms / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=cpp_%s rule=world window=%dx%d wbox=%d hbox=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           simdName(), gw, gh, wbox, hbox, steps, ms, mc, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           world.residentMaxCount(), world.diskWrites(), world.diskReads(), conserved ? "yes" : "no");
    std::filesystem::remove_all(dir);
    return conserved ? 0 : 2;
}

static int runPPM(const char* pathOut, int steps) {
    const int gw = 4, gh = 4;
    std::string dir = "/tmp/sandsim_world_simd_ppm";
    std::filesystem::remove_all(dir);
    SimdWorld world(gw, gh, gw, gh, dir);   // world == one live window
    world.generateAllToDisk();
    world.setWindow(0, 0);
    for (int s = 0; s < steps; ++s) world.step();
    int LW = world.cellsW(), LH = world.cellsH();
    FILE* f = fopen(pathOut, "wb");
    if (!f) return 1;
    fprintf(f, "P6\n%d %d\n255\n", LW, LH);
    std::vector<uint8_t> row((size_t)LW * 3);
    for (int y = 0; y < LH; ++y) {
        for (int x = 0; x < LW; ++x) {
            uint32_t c = kColors[world.viewCell(x, y)];
            row[x*3+0] = (c>>16)&0xFF; row[x*3+1] = (c>>8)&0xFF; row[x*3+2] = c&0xFF;
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
    printf("wrote %s (%dx%d, %d steps)\n", pathOut, LW, LH, steps);
    std::filesystem::remove_all(dir);
    return 0;
}

static int runInteractive(ViewCfg cfg) {
    const int PIXEL = cfg.scale;
    const int vw = std::max(1, (cfg.winW / PIXEL) / CHUNK);   // viewport, in chunks
    const int vh = std::max(1, (cfg.winH / PIXEL) / CHUNK);
    const int MARGIN = 1;                                    // chunks of live border around the view
    const int WBOX = vw + 2 * MARGIN, HBOX = vh + 2 * MARGIN; // keep the sim small: just the view + a thin alive ring
    const char* DIR = "/tmp/sandsim_world_simd_interactive";
    // Resident window = the view plus a MARGIN-chunk live border, all simulated, so
    // a ring around the view keeps bubbling and panning reveals a living edge -- but
    // we don't pay to simulate the whole surroundings (that made the CPU crawl). The
    // disk-streamed huge world is what --bench demonstrates.
    SimdWorld world(WBOX, HBOX, WBOX, HBOX, DIR);
    const int worldW = world.cellsW(), worldH = world.cellsH();
    const int LWv = vw * CHUNK, LHv = vh * CHUNK;            // viewport, in cells
    const int renderW = LWv * PIXEL, renderH = LHv * PIXEL;
    std::filesystem::remove_all(DIR);
    std::filesystem::create_directories(DIR);
    world.generateAllToDisk();
    world.setWindow(0, 0);
    int viewX = (worldW - LWv) / 2, viewY = (worldH - LHv) / 2;   // viewport scroll offset, in cells
    const int PAN = CHUNK / 4;                                    // pan step per key press, in cells
    uint8_t current = SAND;

    // The palette shows every material in id order -- swatch slot i IS material i --
    // so it's driven entirely by MATERIAL_COUNT and kColors, with no per-material
    // swatch list or counts to keep in sync as materials are added.
    ui::Palette pal = ui::palette(renderW, MATERIAL_COUNT);
    // Swatch colours in category order, and a material id -> swatch slot reverse map.
    std::vector<uint32_t> orderedColors(MATERIAL_COUNT);
    int slotOf[MATERIAL_COUNT];
    for (int i = 0; i < MATERIAL_COUNT; ++i) { orderedColors[i] = kColors[kPaletteOrder[i]]; slotOf[kPaletteOrder[i]] = i; }
    auto catOfMat = [&](uint8_t m) { return kCatNames[kSlotCat[slotOf[m]]]; };
    int brushRadius = 4;
    bool painting = false;
    bool paused = false;
    bool stepOnce = false;        // advance exactly one frame while paused (TAB)
    uint8_t paintMat = SAND;      // material laid down while dragging (current, or EMPTY when erasing)

    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);

    // Emissive-bloom buffers (cell resolution) + a precomputed radial falloff kernel.
    const int GR = 3;                                   // glow radius in cells
    const float GLOW = 0.85f;                           // how strongly the bloom is added
    std::vector<float> glowR((size_t)LWv * LHv), glowG((size_t)LWv * LHv), glowB((size_t)LWv * LHv);
    float kern[(2 * GR + 1) * (2 * GR + 1)];
    for (int dy = -GR; dy <= GR; ++dy)
        for (int dx = -GR; dx <= GR; ++dx) {
            float w = 1.0f - std::sqrt((float)(dx * dx + dy * dy)) / (GR + 1);
            kern[(dy + GR) * (2 * GR + 1) + (dx + GR)] = (w > 0.0f) ? w * w : 0.0f;
        }
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "sandsim - paint with the mouse, pick from the palette, hover to identify, SPACE to pause",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderW, renderH, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_RenderSetLogicalSize(renderer, renderW, renderH);    // scales content to the actual output
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING, renderW, renderH);
    int outW = renderW, outH = renderH; SDL_GetRendererOutputSize(renderer, &outW, &outH);
    fprintf(stderr, "sandsim [cpp_%s]: view %dx%d (output %dx%d), scale %d, "
            "world %dx%d chunks = %dx%d cells (all simulated), %d steps/s\n",
            simdName(), renderW, renderH, outW, outH, PIXEL, WBOX, HBOX, worldW, worldH, cfg.simHz);
    bool quit = false;
    int mouseX = 0, mouseY = 0;
    SDL_Event e;
    const double stepDt = 1.0 / cfg.simHz;          // seconds per simulation step
    double acc = 0.0;
    double fpsEMA = 0.0;                             // smoothed render frame rate for the HUD
    auto last = std::chrono::steady_clock::now();
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            else if (e.type == SDL_MOUSEBUTTONDOWN) {
                float flx, fly; SDL_RenderWindowToLogical(renderer, e.button.x, e.button.y, &flx, &fly);
                int h = ui::hit(pal, (int)flx, (int)fly);
                int cellX = viewX + (int)flx / PIXEL, cellY = viewY + (int)fly / PIXEL;
                if (h >= 0) {                          // over the palette
                    if (e.button.button == SDL_BUTTON_LEFT) current = kPaletteOrder[h];  // left-click picks a swatch
                } else if (e.button.button == SDL_BUTTON_MIDDLE) {
                    current = world.viewCell(cellX, cellY);    // eyedropper: pick the material under the cursor
                } else {                               // left paints, right erases
                    painting = true;
                    paintMat = (e.button.button == SDL_BUTTON_RIGHT) ? (uint8_t)EMPTY : current;
                    world.paint(cellX, cellY, paintMat, brushRadius);
                }
            }
            else if (e.type == SDL_MOUSEBUTTONUP) painting = false;
            else if (e.type == SDL_MOUSEWHEEL) {       // scroll to resize the brush
                brushRadius += e.wheel.y;
                if (brushRadius < 0)  brushRadius = 0;
                if (brushRadius > 32) brushRadius = 32;
            }
            else if (e.type == SDL_MOUSEMOTION) SDL_GetMouseState(&mouseX, &mouseY);
            else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_0: current = EMPTY; break;
                    case SDLK_1: current = WALL;  break;
                    case SDLK_2: current = SAND;  break;
                    case SDLK_3: current = WATER; break;
                    case SDLK_4: current = GAS;   break;
                    case SDLK_5: current = OIL;   break;
                    case SDLK_6: current = FIRE;  break;
                    case SDLK_7: current = LAVA;  break;
                    case SDLK_8: current = STEAM; break;
                    case SDLK_9: current = WOOD;  break;
                    case SDLK_p: current = PLANT; break;
                    case SDLK_a: current = ACID;  break;
                    case SDLK_m: current = SMOKE; break;
                    case SDLK_g: current = GLASS; break;
                    case SDLK_i: current = ICE; break;
                    case SDLK_s: current = SPRING; break;
                    case SDLK_t: current = TNT; break;
                    case SDLK_h: current = ASH; break;
                    case SDLK_v: current = VOLCANO; break;
                    case SDLK_x: current = VOID; break;
                    case SDLK_d: current = MUD; break;
                    case SDLK_z: current = VIRUS; break;
                    case SDLK_e: current = SPARK; break;
                    case SDLK_o: current = OBSIDIAN; break;
                    case SDLK_l: current = SALT; break;
                    case SDLK_n: current = SNOW; break;
                    case SDLK_q: current = MERCURY; break;
                    case SDLK_b: current = GUNPOWDER; break;
                    case SDLK_k: current = THERMITE; break;
                    case SDLK_f: current = FROST; break;
                    case SDLK_w: current = WISP; break;
                    case SDLK_c: current = COAL; break;
                    case SDLK_r: current = EMBER; break;
                    case SDLK_u: current = CLONER; break;
                    case SDLK_y: current = CRYSTAL; break;
                    case SDLK_j: current = ANTIMATTER; break;
                    case SDLK_SEMICOLON: current = MOSS; break;
                    case SDLK_COMMA: current = FUMES; break;
                    case SDLK_PERIOD: current = WIRE; break;
                    case SDLK_SLASH:  current = EHEAD; break;
                    case SDLK_QUOTE:  current = ETAIL; break;
                    case SDLK_MINUS:  current = IGNITER; break;
                    case SDLK_EQUALS: current = SENSOR; break;
                    case SDLK_BACKSLASH: current = LIFE; break;
                    case SDLK_BACKQUOTE: current = GEYSER; break;
                    case SDLK_LEFTBRACKET:  if (brushRadius > 0)  brushRadius--; break;
                    case SDLK_RIGHTBRACKET: if (brushRadius < 32) brushRadius++; break;
                    case SDLK_SPACE: paused = !paused; break;        // freeze/resume the simulation
                    case SDLK_TAB: if (paused) stepOnce = true; break; // step one frame while paused
                    case SDLK_BACKSPACE:
                    case SDLK_DELETE: world.clearView(); break;      // wipe the canvas to empty air
                    case SDLK_LEFT:  viewX -= PAN; if (viewX < 0) viewX = 0; break;
                    case SDLK_RIGHT: viewX += PAN; if (viewX > worldW - LWv) viewX = worldW - LWv; break;
                    case SDLK_UP:    viewY -= PAN; if (viewY < 0) viewY = 0; break;
                    case SDLK_DOWN:  viewY += PAN; if (viewY > worldH - LHv) viewY = worldH - LHv; break;
                }
            }
        }
        if (painting) {
            float flx, fly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &flx, &fly);
            world.paint(viewX + (int)flx / PIXEL, viewY + (int)fly / PIXEL, paintMat, brushRadius);
        }
        // Advance the simulation by however much real time elapsed, so physics
        // runs at cfg.simHz steps/s regardless of the render frame rate.
        auto nowT = std::chrono::steady_clock::now();
        double frameDt = std::chrono::duration<double>(nowT - last).count();
        acc += frameDt;
        last = nowT;
        if (frameDt > 0.0) fpsEMA = (fpsEMA <= 0.0) ? 1.0 / frameDt : fpsEMA * 0.92 + (1.0 / frameDt) * 0.08;
        for (int n = 0; !paused && acc >= stepDt && n < 8; ++n) { world.step(); acc -= stepDt; }
        if (acc > stepDt) acc = stepDt;             // drop backlog after a stall
        if (paused && stepOnce) { world.step(); stepOnce = false; }   // single-frame advance
        static int tick = 0; ++tick;                // render clock for the flame/lava flicker

        // Scatter each emissive cell's light into a cell-resolution bloom buffer.
        std::fill(glowR.begin(), glowR.end(), 0.0f);
        std::fill(glowG.begin(), glowG.end(), 0.0f);
        std::fill(glowB.begin(), glowB.end(), 0.0f);
        for (int y = 0; y < LHv; ++y)
            for (int x = 0; x < LWv; ++x) {
                float s = emissionStrength(world.viewCell(viewX + x, viewY + y));
                if (s <= 0.0f) continue;
                uint32_t c = kColors[world.viewCell(viewX + x, viewY + y)];
                float cr = (c >> 16) & 0xFF, cg = (c >> 8) & 0xFF, cb = c & 0xFF;
                for (int dy = -GR; dy <= GR; ++dy) {
                    int ny = y + dy; if (ny < 0 || ny >= LHv) continue;
                    for (int dx = -GR; dx <= GR; ++dx) {
                        int nx = x + dx; if (nx < 0 || nx >= LWv) continue;
                        float w = kern[(dy + GR) * (2 * GR + 1) + (dx + GR)] * s;
                        if (w <= 0.0f) continue;
                        size_t ni = (size_t)ny * LWv + nx;
                        glowR[ni] += cr * w; glowG[ni] += cg * w; glowB[ni] += cb * w;
                    }
                }
            }

        for (int y = 0; y < LHv; ++y)
            for (int x = 0; x < LWv; ++x) {
                int wxc = viewX + x, wyc = viewY + y;       // world cell under this viewport pixel
                uint8_t m = world.viewCell(wxc, wyc);
                uint32_t color = kColors[m];
                if (m == FIRE || m == LAVA) color = ui::flicker(color, wxc, wyc, tick);
                size_t gi = (size_t)y * LWv + x;            // add the accumulated bloom (additive, clamped)
                if (glowR[gi] + glowG[gi] + glowB[gi] > 0.0f) {
                    int rr = (int)((color >> 16) & 0xFF) + (int)(glowR[gi] * GLOW);
                    int gg = (int)((color >> 8)  & 0xFF) + (int)(glowG[gi] * GLOW);
                    int bb = (int)( color        & 0xFF) + (int)(glowB[gi] * GLOW);
                    if (rr > 255) rr = 255;
                    if (gg > 255) gg = 255;
                    if (bb > 255) bb = 255;
                    color = 0xFF000000u | ((uint32_t)rr << 16) | ((uint32_t)gg << 8) | (uint32_t)bb;
                }
                for (int dy = 0; dy < PIXEL; ++dy)
                    for (int dx = 0; dx < PIXEL; ++dx)
                        pixels[(size_t)(y * PIXEL + dy) * renderW + (x * PIXEL + dx)] = color;
            }
        ui::draw(pixels.data(), renderW, renderH, pal, orderedColors.data(), slotOf[current]);
        { // a thin category-coloured accent in the gap under each swatch, so groups read at a glance
            int stride = pal.sw + pal.gap;
            for (int i = 0; i < pal.n; ++i) {
                int sx = pal.x0 + (i % pal.cols) * stride, sy = pal.y0 + (i / pal.cols) * stride;
                ui::fillRect(pixels.data(), renderW, renderH, sx, sy + pal.sw + 1, pal.sw, 2, kCatAccent[kSlotCat[i]]);
            }
        }

        // ---- mouse hover: brush cursor ring on the canvas + a material tooltip ----
        float hlx_f, hly_f;
        SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &hlx_f, &hly_f);
        int hlx = (int)hlx_f, hly = (int)hly_f;
        int hov = ui::hit(pal, hlx, hly);
        bool onCanvas = (hlx >= 0 && hly >= 0 && hlx < renderW && hly < renderH);
        if (hov < 0 && onCanvas) {
            int ccx = hlx / PIXEL, ccy = hly / PIXEL, r = brushRadius;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    if (dx*dx + dy*dy > r*r) continue;                       // inside the circular brush
                    bool edge = (dx-1)*(dx-1)+dy*dy > r*r || (dx+1)*(dx+1)+dy*dy > r*r
                              || dx*dx+(dy-1)*(dy-1) > r*r || dx*dx+(dy+1)*(dy+1) > r*r;
                    if (!edge) continue;                                    // keep only the ring
                    int cx = ccx + dx, cy = ccy + dy;
                    if (cx >= 0 && cy >= 0 && cx < LWv && cy < LHv)
                        ui::fillRect(pixels.data(), renderW, renderH, cx*PIXEL, cy*PIXEL, PIXEL, PIXEL, 0xFFFFFFFFu);
                }
        }
        { // tooltip naming whatever is under the cursor (a palette swatch, or a cell): "NAME  CATEGORY"
            int tipMat = -1;
            if (hov >= 0) tipMat = kPaletteOrder[hov];
            else if (onCanvas) tipMat = world.viewCell(viewX + hlx / PIXEL, viewY + hly / PIXEL);
            if (tipMat >= 0 && kNames[tipMat] && *kNames[tipMat]) {
                char tip[64];
                std::snprintf(tip, sizeof tip, "%s  %s", kNames[tipMat], catOfMat((uint8_t)tipMat));
                int tx = hlx + 16, ty = hly + 16, tw = ui::textWidth(tip, 2);
                if (tx + tw + 6 > renderW) tx = renderW - tw - 6;
                if (ty + 22 > renderH - 26) ty = hly - 24;
                if (tx < 4) tx = 4;
                if (ty < 4) ty = 4;
                ui::label(pixels.data(), renderW, renderH, tx, ty, tip, 2, 0xFFFFFFFFu);
            }
        }

        // ---- bottom info bar: selected material swatch + name + brush + controls ----
        {
            int barH = 24, by = renderH - barH;
            ui::fillRect(pixels.data(), renderW, renderH, 0, by, renderW, barH, 0xFF121218u);
            ui::fillRect(pixels.data(), renderW, renderH, 0, by, renderW, 1, 0xFF3A3A44u);
            ui::fillRect(pixels.data(), renderW, renderH, 6, by + 5, 14, 14, kColors[current] | 0xFF000000u);
            ui::outline(pixels.data(), renderW, renderH, 6, by + 5, 14, 14, 1, 0xFF000000u);
            char buf[128];
            std::snprintf(buf, sizeof buf, "%s  %s   BRUSH %d   %d FPS", kNames[current], catOfMat(current), brushRadius, (int)(fpsEMA + 0.5));
            ui::text(pixels.data(), renderW, renderH, 26, by + 6, buf, 2, 0xFFFFFFFFu);
            const char* help = "LMB PAINT  RMB ERASE  MMB PICK  WHEEL BRUSH  SPACE PAUSE  TAB STEP  DEL CLEAR  ARROWS PAN";
            int hw = ui::textWidth(help, 1);
            ui::text(pixels.data(), renderW, renderH, renderW - hw - 6, by + 9, help, 1, 0xFFB0B0BEu);
        }
        if (paused) {  // centred banner so it's unmistakable
            const char* pw = "PAUSED  -  SPACE RESUME  TAB STEP";
            int s = 3, w = ui::textWidth(pw, s);
            ui::label(pixels.data(), renderW, renderH, (renderW - w) / 2, renderH / 2 - 10, pw, s, 0xFFFFE060u);
        }

        SDL_UpdateTexture(texture, nullptr, pixels.data(), renderW * (int)sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::filesystem::remove_all("/tmp/sandsim_world_simd_interactive");
    return 0;
}

int main(int argc, char* argv[]) {
    g_step = selectStep();   // pick AVX2 or SSE based on the running CPU
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        int steps = (argc > 2) ? std::atoi(argv[2]) : 600;
        int wbox  = (argc > 3) ? std::atoi(argv[3]) : 6;
        int hbox  = (argc > 4) ? std::atoi(argv[4]) : 6;
        return runBench(steps, wbox, hbox);
    }
    if (argc > 2 && std::strcmp(argv[1], "--ppm") == 0) {
        int steps = (argc > 3) ? std::atoi(argv[3]) : 400;
        return runPPM(argv[2], steps);
    }
    return runInteractive(parseView(argc, argv));
}
