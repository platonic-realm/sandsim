/*
 * sandsim - SIMD multi-box streamed world, all materials (SSE)
 *
 * Combines three ideas:
 *
 *  - the chunked, disk-streamed "big world" of sandsim_world.cpp,
 *  - the multi-buffer SIMD technique from sandsim_sse_mb.cpp, where the
 *    interleaved layout cells[(y*BOX + x)*LANES + lane] lets one 128-bit load
 *    read the same in-box cell across LANES boxes, so a single SSE instruction
 *    advances LANES boxes at once, and
 *  - the multi-material rule of sandsim_materials.cpp (EMPTY / WALL / SAND /
 *    WATER / GAS with density-based swaps).
 *
 * The live region is a GRIDW x GRIDH window of boxes (LANES = 16, arranged 4x4)
 * kept resident around a camera; the rest of the world lives on disk. Each
 * frame all 16 boxes advance through the full materials rule in parallel SSE
 * lanes: for every directional move attempt we compute, vectorised across the
 * lanes, which lanes hold an eligible mover whose target is enterable, and swap
 * those lanes with _mm_blendv_epi8. A per-cell "moved" mask reproduces the
 * one-move-per-frame semantics of the scalar engine. Box edges are solid, so
 * each lane is an independent box (the "multiple boxes" model); material flows
 * fully inside a box.
 *
 * Modes:
 *   (default)                 SDL2 window showing the 4x4 live boxes; arrows pan
 *                             the camera by a box, number keys pick a material,
 *                             left mouse paints.
 *   --bench [steps] [wbox] [hbox]
 *                             headless: a finite wbox x hbox world of boxes, a
 *                             deterministic camera sweep that streams boxes to
 *                             disk and back, then a RESULT line with a
 *                             whole-world checksum, conserved per-material
 *                             counts, and streaming stats.
 */

#include <emmintrin.h>
#include <smmintrin.h>   // SSE4.1 (_mm_blendv_epi8)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <SDL2/SDL.h>

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };

static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu,
};

static constexpr int BOX = 64;
static constexpr int LANES = 16;           // SSE: 16 bytes per __m128i
static constexpr int GRIDW = 4, GRIDH = 4; // 16 live boxes as a 4x4 window

static inline uint32_t hashCoord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}
// Deterministic multi-material terrain: perforated wall shelves, with bands of
// sand, then water, then gas. Same generator as the scalar world (minus border).
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
    SimdWorld(int wbox, int hbox, std::string dir)
        : wbox(wbox), hbox(hbox), dir(std::move(dir)) {
        std::filesystem::create_directories(this->dir);
        cells.assign((size_t)BOX * BOX * LANES, EMPTY);
        moved.assign((size_t)BOX * BOX * LANES, 0);
        for (int i = 0; i < LANES; ++i) laneValid[i] = false;
    }

    void generateAllToDisk() {
        std::vector<uint8_t> buf((size_t)BOX * BOX);
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) { genBox(cx, cy, buf); writeBox(cx, cy, buf); }
    }

    void streamWindow(int camCx, int camCy) {
        std::vector<uint8_t> buf((size_t)BOX * BOX);
        for (int dy = 0; dy < GRIDH; ++dy)
            for (int dx = 0; dx < GRIDW; ++dx) {
                int lane = dy * GRIDW + dx;
                int wantCx = camCx + dx, wantCy = camCy + dy;
                if (laneValid[lane] && laneCx[lane] == wantCx && laneCy[lane] == wantCy) continue;
                if (laneValid[lane]) { extractLane(lane, buf); writeBox(laneCx[lane], laneCy[lane], buf); }
                if (!readBox(wantCx, wantCy, buf)) genBox(wantCx, wantCy, buf);
                injectLane(lane, buf);
                laneCx[lane] = wantCx; laneCy[lane] = wantCy; laneValid[lane] = true;
            }
        if (LANES > residentMax) residentMax = LANES;
    }

    // Advance all 16 live boxes one materials step, in parallel SSE lanes.
    void step() {
        const __m128i vE = _mm_set1_epi8(EMPTY);
        const __m128i vS = _mm_set1_epi8(SAND);
        const __m128i vW = _mm_set1_epi8(WATER);
        const __m128i vG = _mm_set1_epi8(GAS);
        const __m128i vZero = _mm_setzero_si128();
        std::memset(moved.data(), 0, moved.size());

        enum Grp { DOWN, GASUP, HORIZ };
        // One masked swap attempt from (x,y) in direction (dx,dy). Updates cur &
        // mvCur (the source) and writes the target cell + its moved mask.
        auto attempt = [&](__m128i& cur, __m128i& mvCur, int x, int y, int dx, int dy, Grp grp) {
            int tx = x + dx, ty = y + dy;
            if (tx < 0 || tx >= BOX || ty < 0 || ty >= BOX) return;  // box edge = solid
            size_t ti = (size_t)(ty * BOX + tx) * LANES;
            __m128i target = _mm_loadu_si128((const __m128i*)&cells[ti]);
            __m128i mvTgt  = _mm_loadu_si128((const __m128i*)&moved[ti]);

            __m128i isSand  = _mm_cmpeq_epi8(cur, vS);
            __m128i isWater = _mm_cmpeq_epi8(cur, vW);
            __m128i isGas   = _mm_cmpeq_epi8(cur, vG);
            __m128i tE = _mm_cmpeq_epi8(target, vE);
            __m128i tW = _mm_cmpeq_epi8(target, vW);
            __m128i tG = _mm_cmpeq_epi8(target, vG);
            // canEnter(mover, target): sand->{E,W,G}, water->{E,G}, gas->{E}
            __m128i canEnter = _mm_or_si128(
                _mm_or_si128(_mm_and_si128(isSand, _mm_or_si128(_mm_or_si128(tE, tW), tG)),
                             _mm_and_si128(isWater, _mm_or_si128(tE, tG))),
                _mm_and_si128(isGas, tE));
            // which materials move in this direction
            __m128i eligible = (grp == DOWN)  ? _mm_or_si128(isSand, isWater)
                             : (grp == GASUP) ? isGas
                                              : _mm_or_si128(isWater, isGas);
            __m128i notCur = _mm_cmpeq_epi8(mvCur, vZero);
            __m128i notTgt = _mm_cmpeq_epi8(mvTgt, vZero);
            __m128i m = _mm_and_si128(_mm_and_si128(eligible, canEnter),
                                      _mm_and_si128(notCur, notTgt));
            // swap source<->target where m is set
            _mm_storeu_si128((__m128i*)&cells[ti], _mm_blendv_epi8(target, cur, m));
            _mm_storeu_si128((__m128i*)&moved[ti], _mm_or_si128(mvTgt, m));
            cur   = _mm_blendv_epi8(cur, target, m);
            mvCur = _mm_or_si128(mvCur, m);
        };

        for (int y = BOX - 1; y >= 0; --y) {           // bottom-to-top
            for (int x = 0; x < BOX; ++x) {
                size_t si = (size_t)(y * BOX + x) * LANES;
                __m128i cur   = _mm_loadu_si128((const __m128i*)&cells[si]);
                __m128i mvCur = _mm_loadu_si128((const __m128i*)&moved[si]);
                bool leftFirst = (((x + y + (int)frame) & 1) == 0);
                int d1 = leftFirst ? -1 : 1, d2 = -d1;
                attempt(cur, mvCur, x, y, 0,  1, DOWN);     // sand/water fall
                attempt(cur, mvCur, x, y, d1, 1, DOWN);
                attempt(cur, mvCur, x, y, d2, 1, DOWN);
                attempt(cur, mvCur, x, y, 0, -1, GASUP);    // gas rises
                attempt(cur, mvCur, x, y, d1, -1, GASUP);
                attempt(cur, mvCur, x, y, d2, -1, GASUP);
                attempt(cur, mvCur, x, y, d1, 0, HORIZ);    // water/gas spread
                attempt(cur, mvCur, x, y, d2, 0, HORIZ);
                _mm_storeu_si128((__m128i*)&cells[si], cur);
                _mm_storeu_si128((__m128i*)&moved[si], mvCur);
            }
        }
        ++frame;
    }

    void paintLane(int lane, int lx, int ly, uint8_t material, int radius) {
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = lx + dx, ny = ly + dy;
                if (nx >= 0 && nx < BOX && ny >= 0 && ny < BOX && dx * dx + dy * dy <= radius * radius)
                    cells[(size_t)(ny * BOX + nx) * LANES + lane] = material;
            }
    }

    uint8_t cellInLane(int lane, int lx, int ly) const {
        return cells[(size_t)(ly * BOX + lx) * LANES + lane];
    }

    void summary(uint64_t& checksum, uint64_t counts[MATERIAL_COUNT]) {
        for (int i = 0; i < MATERIAL_COUNT; ++i) counts[i] = 0;
        std::unordered_map<long long, int> lane;
        for (int i = 0; i < LANES; ++i)
            if (laneValid[i]) lane[((long long)laneCx[i] << 20) | laneCy[i]] = i;
        std::vector<uint8_t> buf((size_t)BOX * BOX);
        uint64_t c = 14695981039346656037ull;
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) {
                auto it = lane.find(((long long)cx << 20) | cy);
                const uint8_t* data;
                if (it != lane.end()) { extractLane(it->second, buf); data = buf.data(); }
                else { readBox(cx, cy, buf); data = buf.data(); }
                for (int i = 0; i < BOX * BOX; ++i) {
                    uint8_t v = data[i];
                    counts[v]++;
                    c = (c ^ (uint64_t)v) * 1099511628211ull;
                }
            }
        checksum = c;
    }

    int residentMaxCount() const { return residentMax; }
    long long diskWrites() const { return nWrites; }
    long long diskReads() const { return nReads; }
    int boxesW() const { return wbox; }
    int boxesH() const { return hbox; }

private:
    int wbox, hbox;
    std::string dir;
    std::vector<uint8_t> cells;     // interleaved: [(y*BOX+x)*LANES + lane]
    std::vector<uint8_t> moved;     // per-frame "already moved" flags (same layout)
    int laneCx[LANES], laneCy[LANES];
    bool laneValid[LANES];
    uint32_t frame = 0;
    int residentMax = 0;
    long long nWrites = 0, nReads = 0;

    void extractLane(int lane, std::vector<uint8_t>& out) const {
        for (int i = 0; i < BOX * BOX; ++i) out[i] = cells[(size_t)i * LANES + lane];
    }
    void injectLane(int lane, const std::vector<uint8_t>& in) {
        for (int i = 0; i < BOX * BOX; ++i) cells[(size_t)i * LANES + lane] = in[i];
    }
    void genBox(int cx, int cy, std::vector<uint8_t>& buf) {
        for (int y = 0; y < BOX; ++y)
            for (int x = 0; x < BOX; ++x)
                buf[y * BOX + x] = seedMat(cx * BOX + x, cy * BOX + y);
    }
    std::string path(int cx, int cy) const {
        char name[64];
        std::snprintf(name, sizeof(name), "/b_%d_%d.bin", cx, cy);
        return dir + name;
    }
    void writeBox(int cx, int cy, const std::vector<uint8_t>& buf) {
        std::ofstream f(path(cx, cy), std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf.data()), BOX * BOX);
        ++nWrites;
    }
    bool readBox(int cx, int cy, std::vector<uint8_t>& buf) const {
        std::ifstream f(path(cx, cy), std::ios::binary);
        if (!f) return false;
        f.read(reinterpret_cast<char*>(buf.data()), BOX * BOX);
        const_cast<SimdWorld*>(this)->nReads++;
        return true;
    }
};

// ---------------------------------------------------------------------------
static int runBench(int steps, int wbox, int hbox) {
    if (wbox < GRIDW) wbox = GRIDW;
    if (hbox < GRIDH) hbox = GRIDH;
    std::string dir = "/tmp/sandsim_world_simd_" + std::to_string(steps) + "_" +
                      std::to_string(wbox) + "x" + std::to_string(hbox);
    std::filesystem::remove_all(dir);
    SimdWorld world(wbox, hbox, dir);
    world.generateAllToDisk();

    uint64_t startCk, startCnt[MATERIAL_COUNT];
    world.summary(startCk, startCnt);

    int nposX = wbox - GRIDW + 1, nposY = hbox - GRIDH + 1, nWin = nposX * nposY;
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * nWin / steps);
        if (visit >= nWin) visit = nWin - 1;
        int row = visit / nposX, col = visit % nposX;
        int camCx = (row % 2 == 0) ? col : (nposX - 1 - col);
        world.streamWindow(camCx, row);
        world.step();
    }
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, cnt[MATERIAL_COUNT];
    world.summary(ck, cnt);
    bool conserved = true;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != startCnt[i]) conserved = false;
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    double cells = (double)BOX * BOX * LANES * steps;
    double mcells = (elapsedMs > 0.0) ? cells / (elapsedMs / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=cpp_world_simd rule=world_simd lanes=%d wbox=%d hbox=%d box=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           LANES, wbox, hbox, BOX, steps, elapsedMs, mcells, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           world.residentMaxCount(), world.diskWrites(), world.diskReads(),
           conserved ? "yes" : "no");

    std::filesystem::remove_all(dir);
    return conserved ? 0 : 2;
}

// Render the 4x4 live region to a PPM after N steps (camera fixed, whole world
// resident), so the materials physics can be inspected without a display.
static int runPPM(const char* path, int steps) {
    std::string dir = "/tmp/sandsim_world_simd_ppm";
    std::filesystem::remove_all(dir);
    SimdWorld world(GRIDW, GRIDH, dir);            // world == one live window
    world.generateAllToDisk();
    world.streamWindow(0, 0);
    for (int s = 0; s < steps; ++s) world.step();

    int W = GRIDW * BOX, H = GRIDH * BOX;
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<uint8_t> row((size_t)W * 3);
    for (int gy = 0; gy < GRIDH; ++gy)
        for (int ly = 0; ly < BOX; ++ly) {
            for (int gx = 0; gx < GRIDW; ++gx) {
                int lane = gy * GRIDW + gx;
                for (int lx = 0; lx < BOX; ++lx) {
                    uint32_t c = kColors[world.cellInLane(lane, lx, ly)];
                    size_t px = (size_t)(gx * BOX + lx) * 3;
                    row[px + 0] = (c >> 16) & 0xFF; row[px + 1] = (c >> 8) & 0xFF; row[px + 2] = c & 0xFF;
                }
            }
            fwrite(row.data(), 1, row.size(), f);
        }
    fclose(f);
    printf("wrote %s (%dx%d, %d steps, 16 boxes)\n", path, W, H, steps);
    std::filesystem::remove_all(dir);
    return 0;
}

static int runInteractive() {
    static const int PIXEL = 2;
    static const int VIEW = GRIDW * BOX;
    static const int WBOX = 16, HBOX = 16;
    int renderW = VIEW * PIXEL, renderH = GRIDH * BOX * PIXEL;

    std::string dir = "/tmp/sandsim_world_simd_interactive";
    std::filesystem::remove_all(dir);
    SimdWorld world(WBOX, HBOX, dir);
    world.generateAllToDisk();
    int camCx = 0, camCy = 0;
    world.streamWindow(camCx, camCy);
    uint8_t current = SAND;

    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "SIMD Multi-Box Materials World - arrows pan  [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderW, renderH, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    // Map rendering and the cursor through a fixed logical size, so painting
    // lands under the pointer even when a tiling compositor (e.g. niri) resizes
    // the window away from the requested size.
    SDL_RenderSetLogicalSize(renderer, renderW, renderH);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING, renderW, renderH);
    bool quit = false, mouseDown = false;
    int mouseX = 0, mouseY = 0;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            else if (e.type == SDL_MOUSEBUTTONDOWN) mouseDown = true;
            else if (e.type == SDL_MOUSEBUTTONUP) mouseDown = false;
            else if (e.type == SDL_MOUSEMOTION) SDL_GetMouseState(&mouseX, &mouseY);
            else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_0: current = EMPTY; break;
                    case SDLK_1: current = WALL;  break;
                    case SDLK_2: current = SAND;  break;
                    case SDLK_3: current = WATER; break;
                    case SDLK_4: current = GAS;   break;
                    case SDLK_LEFT:  if (camCx > 0) camCx--; break;
                    case SDLK_RIGHT: if (camCx < WBOX - GRIDW) camCx++; break;
                    case SDLK_UP:    if (camCy > 0) camCy--; break;
                    case SDLK_DOWN:  if (camCy < HBOX - GRIDH) camCy++; break;
                }
            }
        }
        world.streamWindow(camCx, camCy);
        if (mouseDown) {
            float flx, fly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &flx, &fly);
            int vx = (int)flx / PIXEL, vy = (int)fly / PIXEL;
            if (vx >= 0 && vx < GRIDW * BOX && vy >= 0 && vy < GRIDH * BOX) {
                int lane = (vy / BOX) * GRIDW + (vx / BOX);
                world.paintLane(lane, vx % BOX, vy % BOX, current, 4);
            }
        }
        world.step();

        for (int gy = 0; gy < GRIDH; ++gy)
            for (int gx = 0; gx < GRIDW; ++gx) {
                int lane = gy * GRIDW + gx;
                for (int ly = 0; ly < BOX; ++ly)
                    for (int lx = 0; lx < BOX; ++lx) {
                        uint32_t color = kColors[world.cellInLane(lane, lx, ly)];
                        int vx = gx * BOX + lx, vy = gy * BOX + ly;
                        for (int dy = 0; dy < PIXEL; ++dy)
                            for (int dx = 0; dx < PIXEL; ++dx)
                                pixels[(size_t)(vy * PIXEL + dy) * renderW + (vx * PIXEL + dx)] = color;
                    }
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
    std::filesystem::remove_all(dir);
    return 0;
}

int main(int argc, char* argv[]) {
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
    return runInteractive();
}
