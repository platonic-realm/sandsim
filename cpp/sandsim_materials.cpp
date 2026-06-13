/*
 * sandsim - C++ multi-material engine (Noita-style)
 *
 * Extends the single-material falling-sand simulation to several material
 * classes that interact by density:
 *
 *   EMPTY  air / nothing
 *   WALL   solid, never moves (stone)
 *   SAND   powder: falls down, then piles diagonally
 *   WATER  liquid: falls, then spreads horizontally to find its level
 *   GAS    gas: rises, then spreads horizontally under ceilings
 *
 * Movement is a SWAP between a cell and its target, so every material is
 * conserved (only EMPTY shuffles around). A material may swap into a target
 * only if the target is strictly "lighter" in the relevant direction:
 *
 *   SAND  may enter EMPTY, WATER, GAS   (sinks through water and gas)
 *   WATER may enter EMPTY, GAS          (sinks through gas; floats on sand)
 *   GAS   may enter EMPTY               (rises through air; bubbles handled by
 *                                        WATER sinking, so GAS never re-enters
 *                                        WATER -> no oscillation)
 *
 * The grid is a single buffer updated in place. A per-frame "moved" flag stops
 * any cell that took part in a swap from being processed twice, which keeps the
 * update deterministic for both up-movers (gas) and down-movers (sand/water).
 *
 * Modes:
 *   (default)                 SDL2 window; pick a material with number keys and
 *                             paint with the left mouse button.
 *   --bench [steps] [w] [h]   headless: deterministic multi-material scene,
 *                             time the update loop, print one RESULT line with
 *                             a checksum and per-material counts (which are
 *                             invariant, a built-in conservation check).
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <SDL2/SDL.h>

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };

static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, // EMPTY  black
    0xFF808080u, // WALL   gray
    0xFFE2C878u, // SAND   tan
    0xFF4488FFu, // WATER  blue
    0xFFB0C4DEu, // GAS    pale steel blue
};

// ---------------------------------------------------------------------------
// Shared, language-independent helpers (must match the other ports).
// ---------------------------------------------------------------------------

// May a mover swap into a cell currently holding `target`?
static inline bool canEnter(uint8_t mover, uint8_t target) {
    if (target == WALL) return false;
    switch (mover) {
        case SAND:  return target == EMPTY || target == WATER || target == GAS;
        case WATER: return target == EMPTY || target == GAS;
        case GAS:   return target == EMPTY;
        default:    return false;
    }
}

// Deterministic per-cell material for --bench. Pure integer math so every
// language seeds an identical scene: walls frame the box with a perforated
// shelf across the middle; sand on top, water below it, gas at the bottom.
static inline uint8_t seedMaterial(int x, int y, int w, int h) {
    if (x == 0 || x == w - 1 || y == h - 1) return WALL;       // box walls
    if (y == h / 2 && (x % 20 != 0)) return WALL;              // perforated shelf
    uint32_t hsh = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    hsh = (hsh ^ (hsh >> 13)) * 1274126177u;
    uint32_t r = hsh % 100u;
    if (y < h / 3)        return (r < 40u) ? SAND  : EMPTY;
    else if (y < 2 * h / 3) return (r < 35u) ? WATER : EMPTY;
    else                  return (r < 20u) ? GAS   : EMPTY;
}

static uint64_t checksum(const std::vector<uint8_t>& grid) {
    uint64_t c = 14695981039346656037ull;
    for (uint8_t cell : grid) c = (c ^ (uint64_t)cell) * 1099511628211ull;
    return c;
}

// ---------------------------------------------------------------------------
// The engine.
// ---------------------------------------------------------------------------
class MaterialSim {
public:
    MaterialSim(int w, int h) : width(w), height(h),
                                grid(static_cast<size_t>(w) * h, EMPTY),
                                moved(static_cast<size_t>(w) * h, 0) {}

    void seedBenchScene() {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                grid[idx(x, y)] = seedMaterial(x, y, width, height);
    }

    // One update step. `frame` only affects the left/right tie-break so motion
    // is unbiased yet reproducible.
    void update(uint32_t frame) {
        std::fill(moved.begin(), moved.end(), 0);
        for (int y = height - 1; y >= 0; --y) {       // bottom-to-top
            for (int x = 0; x < width; ++x) {
                if (moved[idx(x, y)]) continue;
                uint8_t m = grid[idx(x, y)];
                if (m == EMPTY || m == WALL) continue;
                bool left = (((uint32_t)x + (uint32_t)y + frame) & 1u) == 0u;
                int d1 = left ? -1 : 1, d2 = -d1;

                if (m == SAND || m == WATER) {
                    if (tryMove(x, y, x, y + 1)) continue;
                    if (tryMove(x, y, x + d1, y + 1)) continue;
                    if (tryMove(x, y, x + d2, y + 1)) continue;
                    if (m == WATER) {
                        if (tryMove(x, y, x + d1, y)) continue;
                        if (tryMove(x, y, x + d2, y)) continue;
                    }
                } else { // GAS
                    if (tryMove(x, y, x, y - 1)) continue;
                    if (tryMove(x, y, x + d1, y - 1)) continue;
                    if (tryMove(x, y, x + d2, y - 1)) continue;
                    if (tryMove(x, y, x + d1, y)) continue;
                    if (tryMove(x, y, x + d2, y)) continue;
                }
            }
        }
    }

    void paint(int px, int py, uint8_t material, int radius, int pixelSize) {
        int cx = px / pixelSize, cy = py / pixelSize;
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = cx + dx, ny = cy + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                    dx * dx + dy * dy <= radius * radius)
                    grid[idx(nx, ny)] = material;
            }
    }

    void clear() { std::fill(grid.begin(), grid.end(), (uint8_t)EMPTY); }

    const std::vector<uint8_t>& cells() const { return grid; }
    int w() const { return width; }
    int h() const { return height; }

private:
    int width, height;
    std::vector<uint8_t> grid;
    std::vector<uint8_t> moved;

    inline size_t idx(int x, int y) const { return (size_t)y * width + x; }

    bool tryMove(int x, int y, int nx, int ny) {
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) return false;
        if (moved[idx(nx, ny)]) return false;
        uint8_t target = grid[idx(nx, ny)];
        if (!canEnter(grid[idx(x, y)], target)) return false;
        grid[idx(nx, ny)] = grid[idx(x, y)];
        grid[idx(x, y)] = target;           // swap (conserves all materials)
        moved[idx(nx, ny)] = 1;
        moved[idx(x, y)] = 1;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Headless benchmark.
// ---------------------------------------------------------------------------
static int runBench(int steps, int width, int height) {
    MaterialSim sim(width, height);
    sim.seedBenchScene();

    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) sim.update((uint32_t)s);
    auto end = std::chrono::steady_clock::now();

    const auto& g = sim.cells();
    uint64_t counts[MATERIAL_COUNT] = {0};
    for (uint8_t c : g) counts[c]++;

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    double cells = (double)width * height * steps;
    double mcells = (elapsedMs > 0.0) ? cells / (elapsedMs / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=cpp_materials rule=materials width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu\n",
           width, height, steps, elapsedMs, mcells,
           (unsigned long long)checksum(g),
           (unsigned long long)counts[EMPTY], (unsigned long long)counts[WALL],
           (unsigned long long)counts[SAND], (unsigned long long)counts[WATER],
           (unsigned long long)counts[GAS]);
    return 0;
}

// ---------------------------------------------------------------------------
// Headless image snapshot (PPM): run the deterministic scene and write one
// pixel per cell, so the physics can be inspected without a display.
// ---------------------------------------------------------------------------
static void writePPM(const char* path, const std::vector<uint8_t>& grid, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        uint32_t c = kColors[grid[i]];
        rgb[i * 3 + 0] = (c >> 16) & 0xFF;
        rgb[i * 3 + 1] = (c >> 8) & 0xFF;
        rgb[i * 3 + 2] = c & 0xFF;
    }
    fwrite(rgb.data(), 1, rgb.size(), f);
    fclose(f);
}

static int runPPM(const char* path, int steps, int width, int height) {
    MaterialSim sim(width, height);
    sim.seedBenchScene();
    for (int s = 0; s < steps; ++s) sim.update((uint32_t)s);
    writePPM(path, sim.cells(), width, height);
    printf("wrote %s (%dx%d, %d steps)\n", path, width, height, steps);
    return 0;
}

// ---------------------------------------------------------------------------
// Interactive mode.
// ---------------------------------------------------------------------------
static const char* materialName(uint8_t m) {
    switch (m) { case EMPTY: return "Eraser"; case WALL: return "Wall";
                 case SAND: return "Sand"; case WATER: return "Water";
                 case GAS: return "Gas"; default: return "?"; }
}

static int runInteractive(int width, int height) {
    static const int PIXEL = 2;
    int renderW = width * PIXEL, renderH = height * PIXEL;
    MaterialSim sim(width, height);
    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);
    std::mt19937 rng(std::random_device{}());

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Materials Sand Simulation - [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser",
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderW, renderH, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    // Map rendering and the cursor through a fixed logical size, so painting
    // lands under the pointer even when a tiling compositor (e.g. niri) resizes
    // the window away from the requested size.
    SDL_RenderSetLogicalSize(renderer, renderW, renderH);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING, renderW, renderH);

    uint8_t current = SAND;
    bool quit = false, mouseDown = false;
    int mouseX = 0, mouseY = 0;
    uint32_t frame = 0;
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
                    case SDLK_c: sim.clear(); break;
                    case SDLK_r: {
                        sim.clear();
                        std::uniform_int_distribution<int> md(SAND, GAS);
                        std::uniform_real_distribution<float> pd(0, 1);
                        // sprinkle a random mix in the top half
                        for (int y = 0; y < height / 2; ++y)
                            for (int x = 0; x < width; ++x)
                                if (pd(rng) < 0.2f) sim.paint(x * PIXEL, y * PIXEL, (uint8_t)md(rng), 0, PIXEL);
                        break;
                    }
                }
                char title[128];
                snprintf(title, sizeof(title), "Materials Sand Simulation - current: %s", materialName(current));
                SDL_SetWindowTitle(window, title);
            }
        }
        if (mouseDown) {
            float lx, ly;
            SDL_RenderWindowToLogical(renderer, mouseX, mouseY, &lx, &ly);
            sim.paint((int)lx, (int)ly, current, 4, PIXEL);
        }

        sim.update(frame++);

        const auto& g = sim.cells();
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                uint32_t color = kColors[g[(size_t)y * width + x]];
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
        int steps  = (argc > 2) ? std::atoi(argv[2]) : 1000;
        int width  = (argc > 3) ? std::atoi(argv[3]) : 400;
        int height = (argc > 4) ? std::atoi(argv[4]) : 300;
        return runBench(steps, width, height);
    }
    if (argc > 2 && std::strcmp(argv[1], "--ppm") == 0) {
        int steps  = (argc > 3) ? std::atoi(argv[3]) : 300;
        int width  = (argc > 4) ? std::atoi(argv[4]) : 200;
        int height = (argc > 5) ? std::atoi(argv[5]) : 150;
        return runPPM(argv[2], steps, width, height);
    }
    return runInteractive(400, 300);
}
