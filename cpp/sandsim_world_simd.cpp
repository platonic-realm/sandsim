/*
 * sandsim - connected SIMD streamed world, all materials (SSE single-grid)
 *
 * This connects the world while keeping the parallelism in SIMD -- no scalar
 * border pass. The earlier multi-buffer approach packed independent boxes into
 * SIMD lanes, which can never connect (lanes don't communicate). Here the lanes
 * are 16 ADJACENT cells of ONE contiguous grid -- the single-grid technique of
 * sandsim_sse_sb.cpp -- so material flows freely across the whole region.
 *
 * The trick that makes multi-material SIMD conflict-free: each directional move
 * maps a source column x to a DISTINCT target column x+dx, so a whole
 * directional pass over a row updates 16 columns at once with no two lanes
 * writing the same cell. The materials rule is decomposed into per-direction
 * SSE passes (swap with _mm_blendv_epi8, a per-cell `moved` mask for one-move-
 * per-frame priority):
 *   down / down-left / down-right     (sand, water)   -- conflict-free
 *   up   / up-left   / up-right       (gas)            -- conflict-free
 *   horizontal-left / -right          (water, gas)     -- even/odd phases, since
 *       a same-row swap chains lane to lane; splitting by column parity breaks
 *       the chain (even sources -> odd targets are disjoint), still all-SIMD.
 *
 * The live region is a contiguous (GW*CHUNK)x(GH*CHUNK) grid with a 1-cell WALL
 * border (padding lets the SIMD offset-loads run off the edge safely). It is a
 * window into a larger world that is streamed to/from disk a chunk at a time as
 * the camera moves.
 *
 * Modes:
 *   (default)                 SDL2 window; arrows pan the camera by a chunk,
 *                             number keys pick a material, left mouse paints.
 *   --bench [steps] [wch] [hch]   headless streaming benchmark (whole-world
 *                             checksum + conserved per-material counts).
 *   --ppm <file> [steps]      render a snapshot.
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
#include <SDL2/SDL.h>

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };
static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu,
};

static constexpr int CHUNK = 64;
static constexpr int GW = 4, GH = 4;             // live window: 4x4 chunks
static constexpr int LW = GW * CHUNK, LH = GH * CHUNK;   // 256 x 256 cells
static constexpr int PAD = 16;                   // WALL border / SIMD halo
static constexpr int SW = LW + 2 * PAD;          // padded stride
static constexpr int SH = LH + 2 * PAD;          // padded height
static constexpr int X0 = PAD, X1 = PAD + LW;    // interior column range
static constexpr int Y0 = PAD, Y1 = PAD + LH;    // interior row range

enum Group { DOWN, GASUP, HORIZ };

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
    SimdWorld(int wbox, int hbox, std::string dir)
        : wbox(wbox), hbox(hbox), dir(std::move(dir)) {
        std::filesystem::create_directories(this->dir);
        grid.assign((size_t)SW * SH, WALL);     // everything starts solid (border stays WALL)
        moved.assign((size_t)SW * SH, 0);
    }

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
            for (int gy = 0; gy < GH; ++gy)
                for (int gx = 0; gx < GW; ++gx) { extractChunk(gx, gy, buf); writeBox(winCx + gx, winCy + gy, buf); }
        for (int gy = 0; gy < GH; ++gy)
            for (int gx = 0; gx < GW; ++gx) {
                if (!readBox(camCx + gx, camCy + gy, buf)) genBox(camCx + gx, camCy + gy, buf);
                injectChunk(gx, gy, buf);
            }
        winCx = camCx; winCy = camCy; windowValid = true;
        residentMax = GW * GH;
    }

    void step() {
        std::memset(moved.data(), 0, moved.size());
        bool flip = frame & 1;
        int dA = flip ? -1 : 1, dB = -dA;
        vertPass(0,  1, DOWN,  false);          // sand/water fall
        vertPass(dA, 1, DOWN,  false);
        vertPass(dB, 1, DOWN,  false);
        vertPass(0, -1, GASUP, true);           // gas rises
        vertPass(dA, -1, GASUP, true);
        vertPass(dB, -1, GASUP, true);
        horizPass(dA, HORIZ);                    // water/gas spread (level out)
        horizPass(dB, HORIZ);
        ++frame;
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
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
                bool inWin = windowValid && cx >= winCx && cx < winCx + GW && cy >= winCy && cy < winCy + GH;
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
    int wbox, hbox;
    std::string dir;
    std::vector<uint8_t> grid;   // padded contiguous live region
    std::vector<uint8_t> moved;
    int winCx = 0, winCy = 0;
    bool windowValid = false;
    uint32_t frame = 0;
    int residentMax = 0;
    long long nWrites = 0, nReads = 0;

    // --- the materials rule, vectorised --------------------------------------
    static __m128i moveMask(__m128i cur, __m128i tgt, __m128i mvCur, __m128i mvTgt, int grp) {
        const __m128i vE = _mm_setzero_si128();
        const __m128i vS = _mm_set1_epi8(SAND), vW = _mm_set1_epi8(WATER), vG = _mm_set1_epi8(GAS);
        __m128i isSand = _mm_cmpeq_epi8(cur, vS), isWater = _mm_cmpeq_epi8(cur, vW), isGas = _mm_cmpeq_epi8(cur, vG);
        __m128i tE = _mm_cmpeq_epi8(tgt, vE), tW = _mm_cmpeq_epi8(tgt, vW), tG = _mm_cmpeq_epi8(tgt, vG);
        __m128i canEnter = _mm_or_si128(
            _mm_or_si128(_mm_and_si128(isSand, _mm_or_si128(_mm_or_si128(tE, tW), tG)),
                         _mm_and_si128(isWater, _mm_or_si128(tE, tG))),
            _mm_and_si128(isGas, tE));
        __m128i eligible = (grp == DOWN)  ? _mm_or_si128(isSand, isWater)
                         : (grp == GASUP) ? isGas
                                          : _mm_or_si128(isWater, isGas);
        __m128i notC = _mm_cmpeq_epi8(mvCur, vE), notT = _mm_cmpeq_epi8(mvTgt, vE);
        return _mm_and_si128(_mm_and_si128(eligible, canEnter), _mm_and_si128(notC, notT));
    }

    // Vertical or diagonal pass: source col x -> target col x+dx in row y+dy.
    // Distinct target columns => conflict-free; full 16-wide.
    void vertPass(int dx, int dy, int grp, bool topDown) {
        int yStart = topDown ? Y0 : (Y1 - 1), yEnd = topDown ? Y1 : (Y0 - 1), yStep = topDown ? 1 : -1;
        for (int y = yStart; y != yEnd; y += yStep)
            for (int x = X0; x < X1; x += 16) {
                uint8_t* gs = &grid[(size_t)y * SW + x];
                uint8_t* gt = &grid[(size_t)(y + dy) * SW + (x + dx)];
                uint8_t* ms = &moved[(size_t)y * SW + x];
                uint8_t* mt = &moved[(size_t)(y + dy) * SW + (x + dx)];
                __m128i cur = _mm_loadu_si128((__m128i*)gs), tgt = _mm_loadu_si128((__m128i*)gt);
                __m128i mc = _mm_loadu_si128((__m128i*)ms), mtt = _mm_loadu_si128((__m128i*)mt);
                __m128i m = moveMask(cur, tgt, mc, mtt, grp);
                _mm_storeu_si128((__m128i*)gt, _mm_blendv_epi8(tgt, cur, m));
                _mm_storeu_si128((__m128i*)gs, _mm_blendv_epi8(cur, tgt, m));
                _mm_storeu_si128((__m128i*)mt, _mm_or_si128(mtt, m));
                _mm_storeu_si128((__m128i*)ms, _mm_or_si128(mc, m));
            }
    }

    // Horizontal spread: same-row swaps chain lane-to-lane, so split by column
    // parity (even sources -> odd targets are disjoint). All targets land inside
    // the source register, so one store per block; conserving.
    void horizPass(int dx, int grp) { horizPhase(dx, true, grp); horizPhase(dx, false, grp); }
    void horizPhase(int dx, bool evenPhase, int grp) {
        const __m128i EVEN = _mm_set_epi8(0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1);
        const __m128i ODD  = _mm_set_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0);
        const __m128i NOT_L0  = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0);
        const __m128i NOT_L15 = _mm_set_epi8(0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
        __m128i laneMask = evenPhase ? EVEN : ODD;
        if (dx < 0 && evenPhase)  laneMask = _mm_and_si128(laneMask, NOT_L0);   // leftmost can't go left
        if (dx > 0 && !evenPhase) laneMask = _mm_and_si128(laneMask, NOT_L15);  // rightmost can't go right
        for (int y = Y0; y < Y1; ++y)
            for (int x = X0; x < X1; x += 16) {
                uint8_t* gs = &grid[(size_t)y * SW + x];
                uint8_t* gn = &grid[(size_t)y * SW + (x + dx)];
                uint8_t* ms = &moved[(size_t)y * SW + x];
                uint8_t* mn = &moved[(size_t)y * SW + (x + dx)];
                __m128i cur = _mm_loadu_si128((__m128i*)gs), nbr = _mm_loadu_si128((__m128i*)gn);
                __m128i mc = _mm_loadu_si128((__m128i*)ms), mn_ = _mm_loadu_si128((__m128i*)mn);
                __m128i m = _mm_and_si128(moveMask(cur, nbr, mc, mn_, grp), laneMask);
                __m128i shCur, shM;
                if (dx < 0) { shCur = _mm_srli_si128(cur, 1); shM = _mm_srli_si128(m, 1); }
                else        { shCur = _mm_slli_si128(cur, 1); shM = _mm_slli_si128(m, 1); }
                __m128i out = _mm_blendv_epi8(cur, nbr, m);     // movers take the neighbour
                out = _mm_blendv_epi8(out, shCur, shM);         // targets take the source
                _mm_storeu_si128((__m128i*)gs, out);
                _mm_storeu_si128((__m128i*)ms, _mm_or_si128(_mm_or_si128(mc, m), shM));
            }
    }

    // --- chunk <-> interior, disk -------------------------------------------
    void extractChunk(int gx, int gy, std::vector<uint8_t>& out) const {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                out[ly * CHUNK + lx] = grid[(size_t)(Y0 + gy * CHUNK + ly) * SW + (X0 + gx * CHUNK + lx)];
    }
    void injectChunk(int gx, int gy, const std::vector<uint8_t>& in) {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                grid[(size_t)(Y0 + gy * CHUNK + ly) * SW + (X0 + gx * CHUNK + lx)] = in[ly * CHUNK + lx];
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
    if (wbox < GW) wbox = GW;
    if (hbox < GH) hbox = GH;
    std::string dir = "/tmp/sandsim_world_simd_" + std::to_string(steps) + "_" +
                      std::to_string(wbox) + "x" + std::to_string(hbox);
    std::filesystem::remove_all(dir);
    SimdWorld world(wbox, hbox, dir);
    world.generateAllToDisk();

    uint64_t startCk, startCnt[MATERIAL_COUNT];
    world.summary(startCk, startCnt);

    int nposX = wbox - GW + 1, nposY = hbox - GH + 1, nWin = nposX * nposY;
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
    double cells = (double)LW * LH * steps;
    double mc = (ms > 0.0) ? cells / (ms / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=cpp_world_simd rule=world_simd window=%dx%d wbox=%d hbox=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           GW, GH, wbox, hbox, steps, ms, mc, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           world.residentMaxCount(), world.diskWrites(), world.diskReads(), conserved ? "yes" : "no");
    std::filesystem::remove_all(dir);
    return conserved ? 0 : 2;
}

static int runPPM(const char* pathOut, int steps) {
    std::string dir = "/tmp/sandsim_world_simd_ppm";
    std::filesystem::remove_all(dir);
    SimdWorld world(GW, GH, dir);           // world == one live window
    world.generateAllToDisk();
    world.setWindow(0, 0);
    for (int s = 0; s < steps; ++s) world.step();
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

static int runInteractive() {
    static const int PIXEL = 2;
    static const int WBOX = 16, HBOX = 16;
    int renderW = LW * PIXEL, renderH = LH * PIXEL;
    std::string dir = "/tmp/sandsim_world_simd_interactive";
    std::filesystem::remove_all(dir);
    SimdWorld world(WBOX, HBOX, dir);
    world.generateAllToDisk();
    int camCx = 0, camCy = 0;
    world.setWindow(camCx, camCy);
    uint8_t current = SAND;

    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "Connected SIMD World - arrows pan  [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderW, renderH, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
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
                    case SDLK_RIGHT: if (camCx < WBOX - GW) camCx++; break;
                    case SDLK_UP:    if (camCy > 0) camCy--; break;
                    case SDLK_DOWN:  if (camCy < HBOX - GH) camCy++; break;
                }
            }
        }
        world.setWindow(camCx, camCy);
        if (mouseDown) {
            float flx, fly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &flx, &fly);
            world.paint((int)flx / PIXEL, (int)fly / PIXEL, current, 4);
        }
        world.step();
        for (int y = 0; y < LH; ++y)
            for (int x = 0; x < LW; ++x) {
                uint32_t color = kColors[world.viewCell(x, y)];
                for (int dy = 0; dy < PIXEL; ++dy)
                    for (int dx = 0; dx < PIXEL; ++dx)
                        pixels[(size_t)(y * PIXEL + dy) * renderW + (x * PIXEL + dx)] = color;
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
