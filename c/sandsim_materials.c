/*
 * sandsim - C multi-material engine (Noita-style)
 *
 * Port of cpp/sandsim_materials.cpp. Materials: EMPTY, WALL, SAND (powder),
 * WATER (liquid), GAS. Movement is a swap, so every material is conserved; a
 * mover may swap into a target only if the target is strictly lighter in the
 * relevant direction (SAND sinks through WATER and GAS; WATER sinks through
 * GAS; GAS rises into EMPTY). A per-frame "moved" flag keeps the in-place
 * update deterministic for both up- and down-movers.
 *
 * Modes:
 *   (default)                 SDL2 window; number keys pick a material, the
 *                             left mouse button paints it.
 *   --bench [steps] [w] [h]   headless: deterministic scene, print one RESULT
 *                             line whose checksum matches the other ports.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <SDL2/SDL.h>

enum { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };
#define PIXEL_SIZE 2

static const uint32_t COLORS[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu,
};

/* ------------------------------------------------------------------ */
/* Shared helpers (identical across all material ports).              */
/* ------------------------------------------------------------------ */
static inline int can_enter(uint8_t mover, uint8_t target) {
    if (target == WALL) return 0;
    switch (mover) {
        case SAND:  return target == EMPTY || target == WATER || target == GAS;
        case WATER: return target == EMPTY || target == GAS;
        case GAS:   return target == EMPTY;
        default:    return 0;
    }
}

static inline uint8_t seed_material(int x, int y, int w, int h) {
    if (x == 0 || x == w - 1 || y == h - 1) return WALL;
    if (y == h / 2 && (x % 20 != 0)) return WALL;
    uint32_t hsh = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    hsh = (hsh ^ (hsh >> 13)) * 1274126177u;
    uint32_t r = hsh % 100u;
    if (y < h / 3)         return (r < 40u) ? SAND  : EMPTY;
    else if (y < 2 * h / 3) return (r < 35u) ? WATER : EMPTY;
    else                   return (r < 20u) ? GAS   : EMPTY;
}

static uint64_t checksum(const uint8_t *grid, size_t n) {
    uint64_t c = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i) c = (c ^ (uint64_t)grid[i]) * 1099511628211ULL;
    return c;
}

typedef struct {
    int w, h;
    uint8_t *grid;
    uint8_t *moved;
} Sim;

static inline int idx(const Sim *s, int x, int y) { return y * s->w + x; }

static int try_move(Sim *s, int x, int y, int nx, int ny) {
    if (nx < 0 || nx >= s->w || ny < 0 || ny >= s->h) return 0;
    int ni = idx(s, nx, ny);
    if (s->moved[ni]) return 0;
    int i = idx(s, x, y);
    uint8_t target = s->grid[ni];
    if (!can_enter(s->grid[i], target)) return 0;
    s->grid[ni] = s->grid[i];
    s->grid[i] = target;        /* swap conserves all materials */
    s->moved[ni] = 1;
    s->moved[i] = 1;
    return 1;
}

static void update(Sim *s, uint32_t frame) {
    memset(s->moved, 0, (size_t)s->w * s->h);
    for (int y = s->h - 1; y >= 0; --y) {
        for (int x = 0; x < s->w; ++x) {
            if (s->moved[idx(s, x, y)]) continue;
            uint8_t m = s->grid[idx(s, x, y)];
            if (m == EMPTY || m == WALL) continue;
            int left = (((uint32_t)x + (uint32_t)y + frame) & 1u) == 0u;
            int d1 = left ? -1 : 1, d2 = -d1;
            if (m == SAND || m == WATER) {
                if (try_move(s, x, y, x, y + 1)) continue;
                if (try_move(s, x, y, x + d1, y + 1)) continue;
                if (try_move(s, x, y, x + d2, y + 1)) continue;
                if (m == WATER) {
                    if (try_move(s, x, y, x + d1, y)) continue;
                    if (try_move(s, x, y, x + d2, y)) continue;
                }
            } else { /* GAS */
                if (try_move(s, x, y, x, y - 1)) continue;
                if (try_move(s, x, y, x + d1, y - 1)) continue;
                if (try_move(s, x, y, x + d2, y - 1)) continue;
                if (try_move(s, x, y, x + d1, y)) continue;
                if (try_move(s, x, y, x + d2, y)) continue;
            }
        }
    }
}

static void paint(Sim *s, int px, int py, uint8_t material, int radius) {
    int cx = px / PIXEL_SIZE, cy = py / PIXEL_SIZE;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = cx + dx, ny = cy + dy;
            if (nx >= 0 && nx < s->w && ny >= 0 && ny < s->h && dx * dx + dy * dy <= radius * radius)
                s->grid[idx(s, nx, ny)] = material;
        }
}

/* ------------------------------------------------------------------ */
/* Headless benchmark.                                                */
/* ------------------------------------------------------------------ */
static int run_bench(int steps, int width, int height) {
    size_t n = (size_t)width * height;
    Sim s = {width, height, malloc(n), malloc(n)};
    if (!s.grid || !s.moved) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            s.grid[idx(&s, x, y)] = seed_material(x, y, width, height);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int step = 0; step < steps; ++step) update(&s, (uint32_t)step);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    uint64_t counts[MATERIAL_COUNT] = {0};
    for (size_t i = 0; i < n; ++i) counts[s.grid[i]]++;
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1.0e6;
    double cells = (double)width * height * steps;
    double mcells = (elapsed_ms > 0.0) ? cells / (elapsed_ms / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=c_materials rule=materials width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu\n",
           width, height, steps, elapsed_ms, mcells,
           (unsigned long long)checksum(s.grid, n),
           (unsigned long long)counts[EMPTY], (unsigned long long)counts[WALL],
           (unsigned long long)counts[SAND], (unsigned long long)counts[WATER],
           (unsigned long long)counts[GAS]);
    free(s.grid);
    free(s.moved);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Interactive SDL2 mode.                                             */
/* ------------------------------------------------------------------ */
static int run_interactive(int width, int height) {
    int rw = width * PIXEL_SIZE, rh = height * PIXEL_SIZE;
    size_t n = (size_t)width * height;
    Sim s = {width, height, calloc(n, 1), calloc(n, 1)};
    uint32_t *pixels = calloc((size_t)rw * rh, sizeof(uint32_t));
    if (!s.grid || !s.moved || !pixels) { fprintf(stderr, "out of memory\n"); return 1; }

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow(
        "Materials Sand Simulation - [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, rw, rh, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    /* Map rendering and the cursor through a fixed logical size, so painting
       lands under the pointer even when a tiling compositor (e.g. niri) resizes
       the window away from the requested size. */
    SDL_RenderSetLogicalSize(renderer, rw, rh);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING, rw, rh);

    uint8_t current = SAND;
    int quit = 0, mouse_down = 0, mx = 0, my = 0;
    uint32_t frame = 0;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = 1;
            else if (e.type == SDL_MOUSEBUTTONDOWN) mouse_down = 1;
            else if (e.type == SDL_MOUSEBUTTONUP) mouse_down = 0;
            else if (e.type == SDL_MOUSEMOTION) SDL_GetMouseState(&mx, &my);
            else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_0: current = EMPTY; break;
                    case SDLK_1: current = WALL;  break;
                    case SDLK_2: current = SAND;  break;
                    case SDLK_3: current = WATER; break;
                    case SDLK_4: current = GAS;   break;
                    case SDLK_c: memset(s.grid, 0, n); break;
                }
            }
        }
        if (mouse_down) {
            float lx, ly;
            SDL_RenderWindowToLogical(renderer, mx, my, &lx, &ly);
            paint(&s, (int)lx, (int)ly, current, 4);
        }
        update(&s, frame++);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                uint32_t color = COLORS[s.grid[idx(&s, x, y)]];
                for (int dy = 0; dy < PIXEL_SIZE; ++dy)
                    for (int dx = 0; dx < PIXEL_SIZE; ++dx)
                        pixels[(size_t)(y * PIXEL_SIZE + dy) * rw + (x * PIXEL_SIZE + dx)] = color;
            }
        SDL_UpdateTexture(texture, NULL, pixels, rw * (int)sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    free(s.grid); free(s.moved); free(pixels);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        int steps  = (argc > 2) ? atoi(argv[2]) : 1000;
        int width  = (argc > 3) ? atoi(argv[3]) : 400;
        int height = (argc > 4) ? atoi(argv[4]) : 300;
        return run_bench(steps, width, height);
    }
    return run_interactive(400, 300);
}
