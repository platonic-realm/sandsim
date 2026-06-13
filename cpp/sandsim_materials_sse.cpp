/*
 * sandsim - SIMD multi-material engine (SSE single-grid)
 *
 * The materials rule (EMPTY/WALL/SAND/WATER/GAS) on one connected, fixed grid,
 * vectorised with the single-grid SSE technique in simd_core.h: the lanes are
 * 16 adjacent cells, each directional move maps a column to a distinct target
 * column (conflict-free), and horizontal spread uses even/odd column phases.
 * The grid is padded with a WALL border so material is contained (conserved)
 * and the SIMD offset-loads can run off the edge safely.
 *
 * The AVX2 build (sandsim_materials_avx.cpp) is the same file with Ops = AvxOps;
 * the two produce identical results (the SIMD width does not change the rule).
 *
 * Modes:
 *   (default)                 SDL2 window; number keys pick a material, mouse paints.
 *   --bench [steps] [w] [h]   headless: deterministic scene, time the update
 *                             loop, print a checksum and conserved counts.
 *   --ppm <file> [steps] [w] [h]   render a snapshot.
 */

#include "simd_core.h"   // SimdOps + simdStep, Material enum
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <SDL2/SDL.h>

using Ops = SseOps;
static const char* kImpl = "cpp_materials_sse";
static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu,
};
static constexpr int PAD = 16;

static inline uint32_t hashCoord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}
// Deterministic scene: perforated wall shelves with sand / water / gas bands.
static inline uint8_t seedMat(int gx, int gy) {
    if (gy % 40 == 39 && (gx % 11 != 0)) return WALL;
    uint32_t r = hashCoord(gx, gy) % 100u;
    switch ((gy / 40) % 3) {
        case 0:  return (r < 35u) ? SAND  : EMPTY;
        case 1:  return (r < 30u) ? WATER : EMPTY;
        default: return (r < 18u) ? GAS   : EMPTY;
    }
}

class MaterialGrid {
public:
    MaterialGrid(int w, int h)
        : W((w + 31) & ~31), H(h),            // round width up to a multiple of 32
          SW(W + 2 * PAD), SH(H + 2 * PAD),
          X0(PAD), X1(PAD + W), Y0(PAD), Y1(PAD + H),
          grid((size_t)SW * SH, WALL), moved((size_t)SW * SH, 0) {}

    void seedScene() {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                grid[(size_t)(y + Y0) * SW + (x + X0)] = seedMat(x, y);
    }
    void step() { simdStep<Ops>(grid.data(), moved.data(), SW, X0, X1, Y0, Y1, frame); ++frame; }

    void paint(int x, int y, uint8_t material, int radius) {
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < W && ny >= 0 && ny < H && dx * dx + dy * dy <= radius * radius)
                    grid[(size_t)(ny + Y0) * SW + (nx + X0)] = material;
            }
    }
    void clear() {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) grid[(size_t)(y + Y0) * SW + (x + X0)] = EMPTY;
    }
    uint8_t cell(int x, int y) const { return grid[(size_t)(y + Y0) * SW + (x + X0)]; }
    int w() const { return W; }
    int h() const { return H; }

    void summary(uint64_t& checksum, uint64_t counts[MATERIAL_COUNT]) const {
        for (int i = 0; i < MATERIAL_COUNT; ++i) counts[i] = 0;
        uint64_t c = 14695981039346656037ull;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                uint8_t v = cell(x, y);
                counts[v]++;
                c = (c ^ v) * 1099511628211ull;
            }
        checksum = c;
    }

private:
    int W, H, SW, SH, X0, X1, Y0, Y1;
    std::vector<uint8_t> grid, moved;
    uint32_t frame = 0;
};

// ---------------------------------------------------------------------------
static int runBench(int steps, int w, int h) {
    MaterialGrid g(w, h);
    g.seedScene();
    uint64_t startCk, startCnt[MATERIAL_COUNT];
    g.summary(startCk, startCnt);

    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) g.step();
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, cnt[MATERIAL_COUNT];
    g.summary(ck, cnt);
    bool conserved = true;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != startCnt[i]) conserved = false;
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double mc = (ms > 0.0) ? (double)g.w() * g.h() * steps / (ms / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=%s rule=materials_simd width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu conserved=%s\n",
           kImpl, g.w(), g.h(), steps, ms, mc, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           conserved ? "yes" : "no");
    return conserved ? 0 : 2;
}

static int runPPM(const char* pathOut, int steps, int w, int h) {
    MaterialGrid g(w, h);
    g.seedScene();
    for (int s = 0; s < steps; ++s) g.step();
    FILE* f = fopen(pathOut, "wb");
    if (!f) return 1;
    fprintf(f, "P6\n%d %d\n255\n", g.w(), g.h());
    std::vector<uint8_t> row((size_t)g.w() * 3);
    for (int y = 0; y < g.h(); ++y) {
        for (int x = 0; x < g.w(); ++x) {
            uint32_t c = kColors[g.cell(x, y)];
            row[x*3+0] = (c>>16)&0xFF; row[x*3+1] = (c>>8)&0xFF; row[x*3+2] = c&0xFF;
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
    printf("wrote %s (%dx%d, %d steps)\n", pathOut, g.w(), g.h(), steps);
    return 0;
}

static int runInteractive() {
    static const int PIXEL = 2, W = 320, H = 240;
    int renderW = W * PIXEL, renderH = H * PIXEL;
    MaterialGrid g(W, H);
    g.seedScene();
    uint8_t current = SAND;
    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "SIMD Materials - [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser, C clear",
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
                    case SDLK_c: g.clear(); break;
                }
            }
        }
        if (mouseDown) {
            float lx, ly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &lx, &ly);
            g.paint((int)lx / PIXEL, (int)ly / PIXEL, current, 4);
        }
        g.step();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                uint32_t color = kColors[g.cell(x, y)];
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
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        int steps = (argc > 2) ? std::atoi(argv[2]) : 1000;
        int w = (argc > 3) ? std::atoi(argv[3]) : 256;
        int h = (argc > 4) ? std::atoi(argv[4]) : 256;
        return runBench(steps, w, h);
    }
    if (argc > 2 && std::strcmp(argv[1], "--ppm") == 0) {
        int steps = (argc > 3) ? std::atoi(argv[3]) : 400;
        int w = (argc > 4) ? std::atoi(argv[4]) : 256;
        int h = (argc > 5) ? std::atoi(argv[5]) : 256;
        return runPPM(argv[2], steps, w, h);
    }
    return runInteractive();
}
