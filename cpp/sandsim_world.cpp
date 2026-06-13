/*
 * sandsim - C++ chunked streaming world (Noita-style)
 *
 * Simulates a world far larger than the screen by keeping only a few "live
 * boxes" (chunks) resident around a camera and streaming the rest to disk. See
 * WORLD.md for the design and the Noita techniques it adapts.
 *
 *   - The world is a sparse grid of CHUNK x CHUNK chunks of material ids.
 *   - Only chunks within a live region around the camera are resident; chunks
 *     leaving the region are saved to disk and freed, chunks entering are read
 *     back (or generated if never visited).
 *   - Each chunk keeps a dirty rectangle; once a chunk settles (no cell moved)
 *     it sleeps and is skipped. A particle crossing a border wakes the neighbor.
 *   - The materials rule (EMPTY/WALL/SAND/WATER/GAS) runs over global cell
 *     coordinates, so material flows across chunk borders naturally.
 *
 * Modes:
 *   (default)                 SDL2 window over a large streamed world; WASD or
 *                             arrows pan the camera, number keys paint.
 *   --bench [steps] [wch] [hch]
 *                             headless: a finite wall-bordered world of wch x hch
 *                             chunks, a deterministic camera sweep that forces
 *                             most chunks out to disk and back, then one RESULT
 *                             line with a whole-world checksum, conserved
 *                             per-material counts, and streaming stats.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <fstream>
#include <filesystem>
#include <SDL2/SDL.h>

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };

static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu,
};

static constexpr int CHUNK = 64;
static constexpr int CHUNK_MASK = CHUNK - 1;
static constexpr int CHUNK_SHIFT = 6;

// --- shared rule (identical to the materials engine) -----------------------
static inline bool canEnter(uint8_t mover, uint8_t target) {
    if (target == WALL) return false;
    switch (mover) {
        case SAND:  return target == EMPTY || target == WATER || target == GAS;
        case WATER: return target == EMPTY || target == GAS;
        case GAS:   return target == EMPTY;
        default:    return false;
    }
}

static inline uint32_t hashCoord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}

// Deterministic world generation in global coordinates (identical across ports).
static inline uint8_t genCell(int gx, int gy, int wcells, int hcells) {
    if (gx == 0 || gy == 0 || gx == wcells - 1 || gy == hcells - 1) return WALL;
    if (gy % 40 == 39 && (gx % 11 != 0)) return WALL;             // perforated shelves
    uint32_t r = hashCoord(gx, gy) % 100u;
    switch ((gy / 40) % 3) {                                       // vertical bands
        case 0:  return (r < 35u) ? SAND  : EMPTY;
        case 1:  return (r < 30u) ? WATER : EMPTY;
        default: return (r < 18u) ? GAS   : EMPTY;
    }
}

struct Chunk {
    std::vector<uint8_t> cells;   // CHUNK*CHUNK material ids
    std::vector<uint8_t> moved;   // per-frame "already moved" flags
    // dirty rect for THIS frame (local coords); empty when minx > maxx
    int dminx, dminy, dmaxx, dmaxy;
    // accumulator for NEXT frame's dirty rect
    int nminx, nminy, nmaxx, nmaxy;

    Chunk() : cells(CHUNK * CHUNK, EMPTY), moved(CHUNK * CHUNK, 0) { fullDirty(); clearNext(); }
    void fullDirty() { dminx = 0; dminy = 0; dmaxx = CHUNK - 1; dmaxy = CHUNK - 1; }
    void clearNext() { nminx = CHUNK; nminy = CHUNK; nmaxx = -1; nmaxy = -1; }
    bool awake() const { return dminx <= dmaxx && dminy <= dmaxy; }
    void wakeLocal(int lx, int ly) {
        if (lx < nminx) nminx = lx;
        if (ly < nminy) nminy = ly;
        if (lx > nmaxx) nmaxx = lx;
        if (ly > nmaxy) nmaxy = ly;
    }
    void commitNext() { dminx = nminx; dminy = nminy; dmaxx = nmaxx; dmaxy = nmaxy; clearNext(); }
};

class World {
public:
    World(int wch, int hch, std::string dir, bool finite)
        : wchunks(wch), hchunks(hch), wcells(wch * CHUNK), hcells(hch * CHUNK),
          finite(finite), dir(std::move(dir)) {
        std::filesystem::create_directories(this->dir);
    }

    int cellsW() const { return wcells; }
    int cellsH() const { return hcells; }

    // --- cell access in global coordinates ---------------------------------
    uint8_t get(int gx, int gy) {
        if (finite && (gx < 0 || gy < 0 || gx >= wcells || gy >= hcells)) return WALL;
        Chunk* c = residentAt(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT);
        if (!c) return WALL;                       // unresident = solid
        return c->cells[(gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK)];
    }
    void set(int gx, int gy, uint8_t m) {
        Chunk* c = residentAt(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT);
        if (c) c->cells[(gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK)] = m;
    }

    // --- generation / streaming --------------------------------------------
    // Pre-generate the whole finite world and flush every chunk to disk.
    void generateAllToDisk() {
        for (int cy = 0; cy < hchunks; ++cy)
            for (int cx = 0; cx < wchunks; ++cx) {
                Chunk ch;
                for (int ly = 0; ly < CHUNK; ++ly)
                    for (int lx = 0; lx < CHUNK; ++lx)
                        ch.cells[ly * CHUNK + lx] = genCell(cx * CHUNK + lx, cy * CHUNK + ly, wcells, hcells);
                writeChunk(cx, cy, ch);
            }
    }

    // Keep chunks within `radius` (Chebyshev) of (camCx,camCy) resident; evict
    // the rest to disk. Returns nothing; updates stats.
    void streamAround(int camCx, int camCy, int radius) {
        // evict out-of-range residents
        std::vector<long long> toEvict;
        for (auto& kv : resident) {
            int cx = (int)(kv.first >> 20), cy = (int)(kv.first & 0xFFFFF);
            if (std::abs(cx - camCx) > radius || std::abs(cy - camCy) > radius)
                toEvict.push_back(kv.first);
        }
        for (long long k : toEvict) {
            int cx = (int)(k >> 20), cy = (int)(k & 0xFFFFF);
            writeChunk(cx, cy, resident[k]);
            resident.erase(k);
        }
        // load/generate in-range chunks
        for (int cy = camCy - radius; cy <= camCy + radius; ++cy)
            for (int cx = camCx - radius; cx <= camCx + radius; ++cx) {
                if (cx < 0 || cy < 0 || cx >= wchunks || cy >= hchunks) continue;
                if (residentAt(cx, cy)) continue;
                loadOrGenerate(cx, cy);
            }
        if ((int)resident.size() > residentMax) residentMax = (int)resident.size();
    }

    // --- one simulation frame ----------------------------------------------
    void step() {
        // collect awake resident chunks, reset their moved flags
        std::vector<std::pair<int,int>> awake;
        for (auto& kv : resident) {
            int cx = (int)(kv.first >> 20), cy = (int)(kv.first & 0xFFFFF);
            std::fill(kv.second.moved.begin(), kv.second.moved.end(), 0);
            if (kv.second.awake()) awake.push_back({cx, cy});
        }
        // bottom chunks first, then left-to-right -> globally bottom-up order
        std::sort(awake.begin(), awake.end(), [](auto a, auto b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });

        for (auto [cx, cy] : awake) {
            Chunk* c = residentAt(cx, cy);
            int baseX = cx * CHUNK, baseY = cy * CHUNK;
            for (int ly = c->dmaxy; ly >= c->dminy; --ly) {       // bottom-to-top
                for (int lx = c->dminx; lx <= c->dmaxx; ++lx) {
                    if (c->moved[ly * CHUNK + lx]) continue;
                    int gx = baseX + lx, gy = baseY + ly;
                    uint8_t m = c->cells[ly * CHUNK + lx];
                    if (m == EMPTY || m == WALL) continue;
                    bool left = (((uint32_t)gx + (uint32_t)gy + frame) & 1u) == 0u;
                    int d1 = left ? -1 : 1, d2 = -d1;
                    if (m == SAND || m == WATER) {
                        if (tryMove(gx, gy, gx, gy + 1)) continue;
                        if (tryMove(gx, gy, gx + d1, gy + 1)) continue;
                        if (tryMove(gx, gy, gx + d2, gy + 1)) continue;
                        if (m == WATER) {
                            if (tryMove(gx, gy, gx + d1, gy)) continue;
                            if (tryMove(gx, gy, gx + d2, gy)) continue;
                        }
                    } else { // GAS
                        if (tryMove(gx, gy, gx, gy - 1)) continue;
                        if (tryMove(gx, gy, gx + d1, gy - 1)) continue;
                        if (tryMove(gx, gy, gx + d2, gy - 1)) continue;
                        if (tryMove(gx, gy, gx + d1, gy)) continue;
                        if (tryMove(gx, gy, gx + d2, gy)) continue;
                    }
                }
            }
        }
        // settle dirty rects: each resident chunk adopts its accumulated next rect
        for (auto& kv : resident) kv.second.commitNext();
        ++frame;
    }

    // The world border (generated as WALL) is the indestructible solid shell,
    // including the bottom floor: painting never modifies it.
    bool indestructible(int gx, int gy) const {
        return gx == 0 || gy == 0 || gx == wcells - 1 || gy == hcells - 1;
    }

    void paint(int gx, int gy, uint8_t material, int radius) {
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = gx + dx, ny = gy + dy;
                if (dx * dx + dy * dy > radius * radius) continue;
                if (indestructible(nx, ny)) continue;
                Chunk* c = residentAt(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT);
                if (!c) continue;
                c->cells[(ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK)] = material;
                wake(nx, ny);
            }
    }

    // --- whole-world summary (resident + on-disk) --------------------------
    void summary(uint64_t& checksum, uint64_t counts[MATERIAL_COUNT]) {
        for (int i = 0; i < MATERIAL_COUNT; ++i) counts[i] = 0;
        uint64_t c = 14695981039346656037ull;
        std::vector<uint8_t> buf(CHUNK * CHUNK);
        for (int cy = 0; cy < hchunks; ++cy)
            for (int cx = 0; cx < wchunks; ++cx) {
                const uint8_t* cells;
                Chunk* res = residentAt(cx, cy);
                if (res) cells = res->cells.data();
                else { readChunkRaw(cx, cy, buf); cells = buf.data(); }
                for (int i = 0; i < CHUNK * CHUNK; ++i) {
                    uint8_t v = cells[i];
                    counts[v]++;
                    c = (c ^ (uint64_t)v) * 1099511628211ull;
                }
            }
        checksum = c;
    }

    // Render the whole world (resident + on-disk chunks) as a PPM image.
    void writePPM(const char* path) {
        FILE* f = fopen(path, "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); return; }
        fprintf(f, "P6\n%d %d\n255\n", wcells, hcells);
        std::vector<uint8_t> row((size_t)wcells * 3);
        std::vector<uint8_t> buf(CHUNK * CHUNK);
        for (int cy = 0; cy < hchunks; ++cy)
            for (int ly = 0; ly < CHUNK; ++ly) {
                for (int cx = 0; cx < wchunks; ++cx) {
                    const uint8_t* cells;
                    Chunk* res = residentAt(cx, cy);
                    if (res) cells = res->cells.data();
                    else { readChunkRaw(cx, cy, buf); cells = buf.data(); }
                    for (int lx = 0; lx < CHUNK; ++lx) {
                        uint32_t c = kColors[cells[ly * CHUNK + lx]];
                        size_t px = (size_t)(cx * CHUNK + lx) * 3;
                        row[px + 0] = (c >> 16) & 0xFF;
                        row[px + 1] = (c >> 8) & 0xFF;
                        row[px + 2] = c & 0xFF;
                    }
                }
                fwrite(row.data(), 1, row.size(), f);
            }
        fclose(f);
    }

    int residentCount() const { return (int)resident.size(); }
    int residentMaxCount() const { return residentMax; }
    long long diskWrites() const { return nDiskWrites; }
    long long diskReads() const { return nDiskReads; }
    long long generated() const { return nGenerated; }

private:
    int wchunks, hchunks, wcells, hcells;
    bool finite;
    std::string dir;
    std::unordered_map<long long, Chunk> resident;  // key = (cx<<20)|cy
    uint32_t frame = 0;
    int residentMax = 0;
    long long nDiskWrites = 0, nDiskReads = 0, nGenerated = 0;

    static long long key(int cx, int cy) { return ((long long)cx << 20) | (long long)cy; }

    Chunk* residentAt(int cx, int cy) {
        auto it = resident.find(key(cx, cy));
        return it == resident.end() ? nullptr : &it->second;
    }

    std::string path(int cx, int cy) const {
        char name[64];
        std::snprintf(name, sizeof(name), "/c_%d_%d.bin", cx, cy);
        return dir + name;
    }
    void writeChunk(int cx, int cy, const Chunk& ch) {
        std::ofstream f(path(cx, cy), std::ios::binary);
        f.write(reinterpret_cast<const char*>(ch.cells.data()), CHUNK * CHUNK);
        ++nDiskWrites;
    }
    bool readChunkRaw(int cx, int cy, std::vector<uint8_t>& out) {
        std::ifstream f(path(cx, cy), std::ios::binary);
        if (!f) return false;
        f.read(reinterpret_cast<char*>(out.data()), CHUNK * CHUNK);
        return true;
    }
    void loadOrGenerate(int cx, int cy) {
        Chunk ch;
        if (readChunkRaw(cx, cy, ch.cells)) ++nDiskReads;
        else {
            for (int ly = 0; ly < CHUNK; ++ly)
                for (int lx = 0; lx < CHUNK; ++lx)
                    ch.cells[ly * CHUNK + lx] = genCell(cx * CHUNK + lx, cy * CHUNK + ly, wcells, hcells);
            ++nGenerated;
        }
        ch.fullDirty();          // freshly resident chunks simulate until settled
        resident[key(cx, cy)] = std::move(ch);
    }

    // Wake the chunk(s) around a changed global cell for next frame.
    void wake(int gx, int gy) {
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = gx + dx, ny = gy + dy;
                Chunk* c = residentAt(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT);
                if (c) c->wakeLocal(nx & CHUNK_MASK, ny & CHUNK_MASK);
            }
    }

    bool tryMove(int gx, int gy, int nx, int ny) {
        if (finite && (nx < 0 || ny < 0 || nx >= wcells || ny >= hcells)) return false;
        Chunk* tc = residentAt(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT);
        if (!tc) return false;                       // unresident neighbor = solid
        int ti = (ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK);
        if (tc->moved[ti]) return false;
        uint8_t target = tc->cells[ti];
        Chunk* sc = residentAt(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT);
        int si = (gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK);
        uint8_t self = sc->cells[si];
        if (!canEnter(self, target)) return false;
        tc->cells[ti] = self;                        // swap conserves materials
        sc->cells[si] = target;
        tc->moved[ti] = 1;
        sc->moved[si] = 1;
        wake(gx, gy);
        wake(nx, ny);
        return true;
    }
};

// ---------------------------------------------------------------------------
// Headless benchmark: finite world, deterministic camera sweep, streaming.
// ---------------------------------------------------------------------------
static int runBench(int steps, int wch, int hch) {
    std::string dir = "/tmp/sandsim_world_cpp_" + std::to_string((long)steps) + "_" +
                      std::to_string(wch) + "x" + std::to_string(hch);
    std::filesystem::remove_all(dir);
    World world(wch, hch, dir, /*finite=*/true);
    world.generateAllToDisk();

    uint64_t startCk, startCnt[MATERIAL_COUNT];
    world.summary(startCk, startCnt);

    int cells = wch * hch;
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * cells / steps);
        if (visit >= cells) visit = cells - 1;
        int row = visit / wch, col = visit % wch;
        int camCx = (row % 2 == 0) ? col : (wch - 1 - col);     // snake sweep
        int camCy = row;
        world.streamAround(camCx, camCy, /*radius=*/1);
        world.step();
    }
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, cnt[MATERIAL_COUNT];
    world.summary(ck, cnt);
    bool conserved = true;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != startCnt[i]) conserved = false;

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    printf("RESULT impl=cpp_world rule=world wchunks=%d hchunks=%d chunk=%d steps=%d "
           "elapsed_ms=%.3f checksum=%016llx empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           wch, hch, CHUNK, steps, elapsedMs, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           world.residentMaxCount(), world.diskWrites(), world.diskReads(),
           conserved ? "yes" : "no");

    std::filesystem::remove_all(dir);
    return conserved ? 0 : 2;
}

// Run the deterministic streaming sim, then write the whole world as a PPM.
static int runPPM(const char* path, int steps, int wch, int hch) {
    std::string dir = "/tmp/sandsim_world_cpp_ppm";
    std::filesystem::remove_all(dir);
    World world(wch, hch, dir, /*finite=*/true);
    world.generateAllToDisk();
    int cells = wch * hch;
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * cells / steps);
        if (visit >= cells) visit = cells - 1;
        int row = visit / wch, col = visit % wch;
        int camCx = (row % 2 == 0) ? col : (wch - 1 - col);
        world.streamAround(camCx, row, 1);
        world.step();
    }
    world.writePPM(path);
    printf("wrote %s (%dx%d cells, %d steps, %dx%d chunks)\n",
           path, wch * CHUNK, hch * CHUNK, steps, wch, hch);
    std::filesystem::remove_all(dir);
    return 0;
}

// ---------------------------------------------------------------------------
// Interactive: a large streamed world viewed through a moving camera.
// ---------------------------------------------------------------------------
static const char* materialName(uint8_t m) {
    switch (m) { case EMPTY: return "Eraser"; case WALL: return "Wall"; case SAND: return "Sand";
                 case WATER: return "Water"; case GAS: return "Gas"; default: return "?"; }
}

static int runInteractive() {
    static const int PIXEL = 2;
    static const int VIEW_W = 320, VIEW_H = 240;       // cells shown
    static const int WCH = 64, HCH = 64;               // 4096x4096-cell world
    int renderW = VIEW_W * PIXEL, renderH = VIEW_H * PIXEL;

    std::string dir = "/tmp/sandsim_world_cpp_interactive";
    std::filesystem::remove_all(dir);
    World world(WCH, HCH, dir, /*finite=*/true);

    int camX = WCH * CHUNK / 2 - VIEW_W / 2;            // top-left global cell of view
    int camY = HCH * CHUNK / 2 - VIEW_H / 2;
    uint8_t current = SAND;

    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "Streamed World - WASD/arrows pan  [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser",
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
                    case SDLK_w: case SDLK_UP:    camY -= 16; break;
                    case SDLK_s: case SDLK_DOWN:  camY += 16; break;
                    case SDLK_a: case SDLK_LEFT:  camX -= 16; break;
                    case SDLK_d: case SDLK_RIGHT: camX += 16; break;
                }
                char title[160];
                snprintf(title, sizeof(title),
                         "Streamed World - painting %s - camera (%d,%d) - resident chunks %d",
                         materialName(current), camX, camY, world.residentCount());
                SDL_SetWindowTitle(window, title);
            }
        }
        // keep chunks under the viewport (plus margin) resident
        int camCx = (camX + VIEW_W / 2) >> CHUNK_SHIFT;
        int camCy = (camY + VIEW_H / 2) >> CHUNK_SHIFT;
        world.streamAround(camCx, camCy, 3);

        if (mouseDown) {
            float lx, ly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &lx, &ly);
            world.paint(camX + (int)lx / PIXEL, camY + (int)ly / PIXEL, current, 4);
        }
        world.step();

        for (int vy = 0; vy < VIEW_H; ++vy)
            for (int vx = 0; vx < VIEW_W; ++vx) {
                uint32_t color = kColors[world.get(camX + vx, camY + vy)];
                for (int dy = 0; dy < PIXEL; ++dy)
                    for (int dx = 0; dx < PIXEL; ++dx)
                        pixels[(size_t)(vy * PIXEL + dy) * renderW + (vx * PIXEL + dx)] = color;
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
        int wch   = (argc > 3) ? std::atoi(argv[3]) : 6;
        int hch   = (argc > 4) ? std::atoi(argv[4]) : 6;
        return runBench(steps, wch, hch);
    }
    if (argc > 2 && std::strcmp(argv[1], "--ppm") == 0) {
        int steps = (argc > 3) ? std::atoi(argv[3]) : 600;
        int wch   = (argc > 4) ? std::atoi(argv[4]) : 6;
        int hch   = (argc > 5) ? std::atoi(argv[5]) : 6;
        return runPPM(argv[2], steps, wch, hch);
    }
    return runInteractive();
}
