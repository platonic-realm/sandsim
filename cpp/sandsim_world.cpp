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
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <SDL2/SDL.h>

static StepFn g_step = nullptr;   // selected at startup (AVX2 or SSE)
static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu, 0xFF8E44ADu, 0xFFFF5A1Eu, 0xFFCF1B0Bu, 0xFFDCE4ECu, 0xFF8B5A2Bu, 0xFF3AA84Au, 0xFFB8F000u, 0xFF585860u, 0xFFAEE0E8u, 0xFFCDEBFFu,
};

static constexpr int CHUNK = 64;   // simulation chunk = 64x64 cells
static constexpr int PAD = 16;     // WALL border / SIMD halo

// Window resolution + virtual-pixel scale + simulation rate for the interactive
// view. simHz is steps/second, decoupled from the render rate so the physics runs
// at the same wall-clock speed on every backend.
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
        }
        ++frame;
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
        if (material == FIRE || material == LAVA || material == STEAM || material == PLANT || material == ACID || material == SMOKE || material == ICE) hasReactive = true;
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = lx + dx, ny = ly + dy;
                if (nx >= 0 && nx < LW && ny >= 0 && ny < LH && dx * dx + dy * dy <= radius * radius)
                    grid[(size_t)(ny + Y0) * SW + (nx + X0)] = material;
            }
    }
    uint8_t viewCell(int lx, int ly) const { return grid[(size_t)(ly + Y0) * SW + (lx + X0)]; }

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
    bool hasReactive = false;      // gates the reaction passes; set when fire/lava enters the grid

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
                if (v == FIRE || v == LAVA || v == STEAM || v == PLANT || v == ACID || v == SMOKE || v == ICE) hasReactive = true;
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
    const int gw = std::max(1, (cfg.winW / PIXEL) / CHUNK);
    const int gh = std::max(1, (cfg.winH / PIXEL) / CHUNK);
    const int WBOX = 2 * gw, HBOX = 2 * gh;            // room to pan around the world
    SimdWorld world(gw, gh, WBOX, HBOX, "/tmp/sandsim_world_simd_interactive");
    const int LW = world.cellsW(), LH = world.cellsH();
    const int renderW = LW * PIXEL, renderH = LH * PIXEL;
    std::filesystem::remove_all("/tmp/sandsim_world_simd_interactive");
    std::filesystem::create_directories("/tmp/sandsim_world_simd_interactive");
    world.generateAllToDisk();
    int camCx = 0, camCy = 0;
    world.setWindow(camCx, camCy);
    uint8_t current = SAND;

    static const uint8_t kSwatch[15] = {EMPTY, WALL, SAND, WATER, GAS, OIL, FIRE, LAVA, STEAM, WOOD, PLANT, ACID, SMOKE, GLASS, ICE};
    uint32_t swatchCol[15];
    for (int i = 0; i < 15; ++i) swatchCol[i] = kColors[kSwatch[i]];
    ui::Palette pal = ui::palette(renderW, 15);
    int brushRadius = 4;
    bool painting = false;
    auto selectedIdx = [&]() { for (int i = 0; i < 15; ++i) if (kSwatch[i] == current) return i; return -1; };

    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "Connected SIMD World - arrows pan  [1]Wall [2]Sand [3]Water [4]Gas [5]Oil [0]Eraser",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderW, renderH, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_RenderSetLogicalSize(renderer, renderW, renderH);    // scales content to the actual output
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING, renderW, renderH);
    int outW = renderW, outH = renderH; SDL_GetRendererOutputSize(renderer, &outW, &outH);
    fprintf(stderr, "sandsim [cpp_%s]: window %dx%d (output %dx%d), scale %d, "
            "grid %dx%d chunks = %dx%d cells, %d steps/s\n",
            simdName(), renderW, renderH, outW, outH, PIXEL, gw, gh, LW, LH, cfg.simHz);
    bool quit = false;
    int mouseX = 0, mouseY = 0;
    SDL_Event e;
    const double stepDt = 1.0 / cfg.simHz;          // seconds per simulation step
    double acc = 0.0;
    auto last = std::chrono::steady_clock::now();
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            else if (e.type == SDL_MOUSEBUTTONDOWN) {
                float flx, fly; SDL_RenderWindowToLogical(renderer, e.button.x, e.button.y, &flx, &fly);
                int h = ui::hit(pal, (int)flx, (int)fly);
                if (h >= 0) current = kSwatch[h];     // clicked a palette swatch
                else { painting = true; world.paint((int)flx / PIXEL, (int)fly / PIXEL, current, brushRadius); }
            }
            else if (e.type == SDL_MOUSEBUTTONUP) painting = false;
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
                    case SDLK_LEFTBRACKET:  if (brushRadius > 0)  brushRadius--; break;
                    case SDLK_RIGHTBRACKET: if (brushRadius < 32) brushRadius++; break;
                    case SDLK_LEFT:  if (camCx > 0) camCx--; break;
                    case SDLK_RIGHT: if (camCx < WBOX - gw) camCx++; break;
                    case SDLK_UP:    if (camCy > 0) camCy--; break;
                    case SDLK_DOWN:  if (camCy < HBOX - gh) camCy++; break;
                }
            }
        }
        world.setWindow(camCx, camCy);
        if (painting) {
            float flx, fly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &flx, &fly);
            world.paint((int)flx / PIXEL, (int)fly / PIXEL, current, brushRadius);
        }
        // Advance the simulation by however much real time elapsed, so physics
        // runs at cfg.simHz steps/s regardless of the render frame rate.
        auto nowT = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double>(nowT - last).count();
        last = nowT;
        for (int n = 0; acc >= stepDt && n < 8; ++n) { world.step(); acc -= stepDt; }
        if (acc > stepDt) acc = stepDt;             // drop backlog after a stall
        static int tick = 0; ++tick;                // render clock for the flame/lava flicker
        for (int y = 0; y < LH; ++y)
            for (int x = 0; x < LW; ++x) {
                uint8_t m = world.viewCell(x, y);
                uint32_t color = kColors[m];
                if (m == FIRE || m == LAVA) color = ui::flicker(color, x, y, tick);
                for (int dy = 0; dy < PIXEL; ++dy)
                    for (int dx = 0; dx < PIXEL; ++dx)
                        pixels[(size_t)(y * PIXEL + dy) * renderW + (x * PIXEL + dx)] = color;
            }
        ui::draw(pixels.data(), renderW, renderH, pal, swatchCol, selectedIdx());
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
