/*
 * sandsim - SIMD multi-box streamed world (SSE)
 *
 * Combines two ideas:
 *
 *  - the chunked, disk-streamed "big world" of sandsim_world.cpp, and
 *  - the multi-buffer SIMD technique from sandsim_sse_mb.cpp, where the
 *    interleaved layout cells[(y*BOX + x)*LANES + lane] lets one 128-bit load
 *    read the same in-box cell across LANES boxes, so a single SSE instruction
 *    advances LANES boxes at once.
 *
 * The live region is a GRIDW x GRIDH window of boxes (LANES = 16 for SSE,
 * arranged 4x4) kept resident around a camera; the rest of the world lives on
 * disk and is streamed in/out as the camera moves. Every frame, all 16 live
 * boxes advance in parallel SIMD lanes.
 *
 * To keep the SIMD lanes clean (the whole point of the multi-buffer trick), each
 * box has solid edges and does not exchange material with its neighbours --
 * exactly the "independent buffers" model of the original multi-buffer code,
 * here used as the streamed live boxes. The fully *connected* multi-material
 * world is sandsim_world.cpp; this variant trades cross-box flow for 16-wide
 * SIMD throughput. Sand only (0/1), matching the multi-buffer reference.
 *
 * Modes:
 *   (default)                 SDL2 window showing the 4x4 live boxes; arrows pan
 *                             the camera by a box, left mouse paints sand.
 *   --bench [steps] [wbox] [hbox]
 *                             headless: a finite wbox x hbox world of boxes, a
 *                             deterministic camera sweep that streams boxes to
 *                             disk and back, then a RESULT line with a
 *                             whole-world checksum, conserved sand count, and
 *                             streaming stats.
 */

#include <emmintrin.h>
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

static constexpr int BOX = 64;
static constexpr int LANES = 16;           // SSE: 16 bytes per __m128i
static constexpr int GRIDW = 4, GRIDH = 4; // 16 live boxes as a 4x4 window

static inline uint32_t hashCoord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}
static inline uint8_t seedSand(int gx, int gy) {
    return (hashCoord(gx, gy) % 100u) < 25u ? 1u : 0u;
}

class SimdWorld {
public:
    SimdWorld(int wbox, int hbox, std::string dir)
        : wbox(wbox), hbox(hbox), dir(std::move(dir)) {
        std::filesystem::create_directories(this->dir);
        cells.assign((size_t)BOX * BOX * LANES, 0);
        for (int i = 0; i < LANES; ++i) laneValid[i] = false;
    }

    void generateAllToDisk() {
        std::vector<uint8_t> buf((size_t)BOX * BOX);
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) {
                genBox(cx, cy, buf);
                writeBox(cx, cy, buf);
            }
    }

    // Keep the GRIDW x GRIDH window with top-left box (camCx,camCy) resident.
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

    // Advance all 16 live boxes one step, in parallel SSE lanes. This is the
    // multi-buffer fall (down, then down-left, then down-right) with solid box
    // edges (no wrap), so each lane is an independent box.
    void step() {
        const __m128i SAND = _mm_set1_epi8(1);
        const __m128i EMPTY = _mm_setzero_si128();
        for (int y = BOX - 2; y >= 0; --y) {
            for (int x = 0; x < BOX; ++x) {
                __m128i cur = load(x, y);
                __m128i below = load(x, y + 1);

                __m128i isSand = _mm_cmpeq_epi8(cur, SAND);
                __m128i canDown = _mm_and_si128(isSand, _mm_cmpeq_epi8(below, EMPTY));
                below = _mm_or_si128(below, _mm_and_si128(canDown, SAND));
                cur = _mm_andnot_si128(canDown, cur);

                if (x > 0) {
                    __m128i left = load(x - 1, y + 1);
                    isSand = _mm_cmpeq_epi8(cur, SAND);
                    __m128i canL = _mm_and_si128(isSand, _mm_cmpeq_epi8(left, EMPTY));
                    left = _mm_or_si128(left, _mm_and_si128(canL, SAND));
                    cur = _mm_andnot_si128(canL, cur);
                    store(x - 1, y + 1, left);
                }
                if (x < BOX - 1) {
                    __m128i right = load(x + 1, y + 1);
                    isSand = _mm_cmpeq_epi8(cur, SAND);
                    __m128i canR = _mm_and_si128(isSand, _mm_cmpeq_epi8(right, EMPTY));
                    right = _mm_or_si128(right, _mm_and_si128(canR, SAND));
                    cur = _mm_andnot_si128(canR, cur);
                    store(x + 1, y + 1, right);
                }
                store(x, y, cur);
                store(x, y + 1, below);
            }
        }
    }

    void paintLane(int lane, int lx, int ly, int radius) {
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = lx + dx, ny = ly + dy;
                if (nx >= 0 && nx < BOX && ny >= 0 && ny < BOX && dx * dx + dy * dy <= radius * radius)
                    cells[(size_t)(ny * BOX + nx) * LANES + lane] = 1;
            }
    }

    uint8_t cellInLane(int lane, int lx, int ly) const {
        return cells[(size_t)(ly * BOX + lx) * LANES + lane];
    }
    bool laneIsValid(int lane) const { return laneValid[lane]; }

    void summary(uint64_t& checksum, uint64_t& sand) {
        // map currently-resident boxes to their lane
        std::unordered_map<long long, int> lane;
        for (int i = 0; i < LANES; ++i)
            if (laneValid[i]) lane[((long long)laneCx[i] << 20) | laneCy[i]] = i;
        std::vector<uint8_t> buf((size_t)BOX * BOX);
        uint64_t c = 14695981039346656037ull, s = 0;
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) {
                auto it = lane.find(((long long)cx << 20) | cy);
                const uint8_t* data;
                if (it != lane.end()) { extractLane(it->second, buf); data = buf.data(); }
                else { readBox(cx, cy, buf); data = buf.data(); }
                for (int i = 0; i < BOX * BOX; ++i) {
                    uint8_t v = data[i] & 1u;
                    s += v;
                    c = (c ^ (uint64_t)v) * 1099511628211ull;
                }
            }
        checksum = c; sand = s;
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
    int laneCx[LANES], laneCy[LANES];
    bool laneValid[LANES];
    int residentMax = 0;
    long long nWrites = 0, nReads = 0;

    __m128i load(int x, int y) const {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(&cells[(size_t)(y * BOX + x) * LANES]));
    }
    void store(int x, int y, __m128i v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&cells[(size_t)(y * BOX + x) * LANES]), v);
    }
    void extractLane(int lane, std::vector<uint8_t>& out) const {
        for (int i = 0; i < BOX * BOX; ++i) out[i] = cells[(size_t)i * LANES + lane];
    }
    void injectLane(int lane, const std::vector<uint8_t>& in) {
        for (int i = 0; i < BOX * BOX; ++i) cells[(size_t)i * LANES + lane] = in[i];
    }
    void genBox(int cx, int cy, std::vector<uint8_t>& buf) {
        for (int y = 0; y < BOX; ++y)
            for (int x = 0; x < BOX; ++x)
                buf[y * BOX + x] = seedSand(cx * BOX + x, cy * BOX + y);
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

    uint64_t startCk, startSand;
    world.summary(startCk, startSand);

    int nposX = wbox - GRIDW + 1, nposY = hbox - GRIDH + 1, nWin = nposX * nposY;
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * nWin / steps);
        if (visit >= nWin) visit = nWin - 1;
        int row = visit / nposX, col = visit % nposX;
        int camCx = (row % 2 == 0) ? col : (nposX - 1 - col);
        int camCy = row;
        world.streamWindow(camCx, camCy);
        world.step();
    }
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, sand;
    world.summary(ck, sand);
    bool conserved = (sand == startSand);
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    double cells = (double)BOX * BOX * LANES * steps;
    double mcells = (elapsedMs > 0.0) ? cells / (elapsedMs / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=cpp_world_simd rule=world_simd lanes=%d wbox=%d hbox=%d box=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx sand=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           LANES, wbox, hbox, BOX, steps, elapsedMs, mcells, (unsigned long long)ck,
           (unsigned long long)sand, world.residentMaxCount(),
           world.diskWrites(), world.diskReads(), conserved ? "yes" : "no");

    std::filesystem::remove_all(dir);
    return conserved ? 0 : 2;
}

static int runInteractive() {
    static const int PIXEL = 2;
    static const int VIEW = GRIDW * BOX;               // 256 cells across
    static const int WBOX = 16, HBOX = 16;
    int renderW = VIEW * PIXEL, renderH = GRIDH * BOX * PIXEL;

    std::string dir = "/tmp/sandsim_world_simd_interactive";
    std::filesystem::remove_all(dir);
    SimdWorld world(WBOX, HBOX, dir);
    world.generateAllToDisk();
    int camCx = 0, camCy = 0;
    world.streamWindow(camCx, camCy);

    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("SIMD Multi-Box Streamed World - arrows pan, mouse paints sand",
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
                if (e.key.keysym.sym == SDLK_LEFT  && camCx > 0) camCx--;
                if (e.key.keysym.sym == SDLK_RIGHT && camCx < WBOX - GRIDW) camCx++;
                if (e.key.keysym.sym == SDLK_UP    && camCy > 0) camCy--;
                if (e.key.keysym.sym == SDLK_DOWN  && camCy < HBOX - GRIDH) camCy++;
            }
        }
        world.streamWindow(camCx, camCy);
        if (mouseDown) {
            float flx, fly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &flx, &fly);
            int vx = (int)flx / PIXEL, vy = (int)fly / PIXEL;   // view cell
            if (vx >= 0 && vx < GRIDW * BOX && vy >= 0 && vy < GRIDH * BOX) {
                int lane = (vy / BOX) * GRIDW + (vx / BOX);
                world.paintLane(lane, vx % BOX, vy % BOX, 4);
            }
        }
        world.step();

        for (int gy = 0; gy < GRIDH; ++gy)
            for (int gx = 0; gx < GRIDW; ++gx) {
                int lane = gy * GRIDW + gx;
                for (int ly = 0; ly < BOX; ++ly)
                    for (int lx = 0; lx < BOX; ++lx) {
                        uint32_t color = world.cellInLane(lane, lx, ly) ? 0xFFFFFF00u : 0xFF000000u;
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
    return runInteractive();
}
