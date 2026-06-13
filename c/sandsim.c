/*
 * sandsim - C implementation
 *
 * A falling-sand cellular automaton. This is the canonical "scalar" rule,
 * a direct port of cpp/sandsim_scalar_sb.cpp:
 *   for each cell bottom-to-top, sand falls straight down, else down-left,
 *   else down-right, else stays.
 *
 * Two modes:
 *   (default)         SDL2 window, 400x300 grid rendered at 2x (800x600).
 *   --bench [steps] [w] [h]   headless: deterministic seed, time the update
 *                             loop, print one machine-readable RESULT line.
 *
 * The --bench seed and checksum match every other implementation, so all of
 * the scalar-rule ports produce an identical checksum.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <SDL2/SDL.h>

#define PIXEL_SIZE 2
#define SAND 1
#define EMPTY 0

/* ------------------------------------------------------------------ */
/* Shared, language-independent helpers used by --bench.              */
/* ------------------------------------------------------------------ */

/* Deterministic per-cell seed (~30% sand), pure u32 wraparound math. */
static inline uint8_t seed_cell(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (h % 100u) < 30u ? SAND : EMPTY;
}

/* FNV-1a over the grid, row-major, u64 wraparound. */
static uint64_t checksum(const uint8_t *grid, size_t n) {
    uint64_t c = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i) {
        c = (c ^ (uint64_t)grid[i]) * 1099511628211ULL;
    }
    return c;
}

/* The one true update step. Both modes call this. */
static void update(uint8_t *buf, int width, int height) {
    for (int y = height - 2; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            if (buf[y * width + x] == SAND) {
                if (buf[(y + 1) * width + x] == EMPTY) {
                    buf[(y + 1) * width + x] = SAND;
                    buf[y * width + x] = EMPTY;
                } else if (x > 0 && buf[(y + 1) * width + (x - 1)] == EMPTY) {
                    buf[(y + 1) * width + (x - 1)] = SAND;
                    buf[y * width + x] = EMPTY;
                } else if (x < width - 1 && buf[(y + 1) * width + (x + 1)] == EMPTY) {
                    buf[(y + 1) * width + (x + 1)] = SAND;
                    buf[y * width + x] = EMPTY;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Headless benchmark.                                                */
/* ------------------------------------------------------------------ */

static int run_bench(int steps, int width, int height) {
    size_t n = (size_t)width * (size_t)height;
    uint8_t *buf = (uint8_t *)malloc(n);
    if (!buf) { fprintf(stderr, "out of memory\n"); return 1; }

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            buf[y * width + x] = seed_cell(x, y);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int step = 0; step < steps; ++step)
        update(buf, width, height);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1.0e6;
    double cells = (double)width * height * steps;
    double mcells = (elapsed_ms > 0.0) ? cells / (elapsed_ms / 1000.0) / 1e6 : 0.0;

    printf("RESULT impl=c rule=scalar width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx\n",
           width, height, steps, elapsed_ms, mcells,
           (unsigned long long)checksum(buf, n));

    free(buf);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Interactive SDL2 mode.                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    int width, height;
    int render_w, render_h;
    uint8_t *buf;
    uint32_t *pixels;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
} Sim;

static void add_sand(Sim *s, int px, int py, int radius) {
    int x = px / PIXEL_SIZE;
    int y = py / PIXEL_SIZE;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < s->width && ny >= 0 && ny < s->height &&
                dx * dx + dy * dy <= radius * radius) {
                s->buf[ny * s->width + nx] = SAND;
            }
        }
    }
}

static void randomize(Sim *s, double density) {
    for (int i = 0; i < s->width * s->height; ++i)
        s->buf[i] = ((double)rand() / RAND_MAX < density) ? SAND : EMPTY;
}

static void render(Sim *s) {
    for (int y = 0; y < s->height; ++y) {
        for (int x = 0; x < s->width; ++x) {
            uint32_t color = s->buf[y * s->width + x] ? 0xFFFFFF00u : 0xFF000000u;
            for (int dy = 0; dy < PIXEL_SIZE; ++dy)
                for (int dx = 0; dx < PIXEL_SIZE; ++dx) {
                    int rx = x * PIXEL_SIZE + dx;
                    int ry = y * PIXEL_SIZE + dy;
                    s->pixels[ry * s->render_w + rx] = color;
                }
        }
    }
    SDL_UpdateTexture(s->texture, NULL, s->pixels, s->render_w * (int)sizeof(uint32_t));
    SDL_RenderClear(s->renderer);
    SDL_RenderCopy(s->renderer, s->texture, NULL, NULL);
    SDL_RenderPresent(s->renderer);
}

static int run_interactive(int width, int height) {
    Sim s;
    s.width = width;
    s.height = height;
    s.render_w = width * PIXEL_SIZE;
    s.render_h = height * PIXEL_SIZE;
    s.buf = (uint8_t *)calloc((size_t)width * height, 1);
    s.pixels = (uint32_t *)calloc((size_t)s.render_w * s.render_h, sizeof(uint32_t));
    if (!s.buf || !s.pixels) { fprintf(stderr, "out of memory\n"); return 1; }
    srand((unsigned)time(NULL));

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    s.window = SDL_CreateWindow("C Sand Simulation",
                                SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                s.render_w, s.render_h, 0);
    s.renderer = SDL_CreateRenderer(s.window, -1, SDL_RENDERER_ACCELERATED);
    s.texture = SDL_CreateTexture(s.renderer, SDL_PIXELFORMAT_ARGB8888,
                                  SDL_TEXTUREACCESS_STREAMING, s.render_w, s.render_h);

    int quit = 0, mouse_down = 0, mouse_x = 0, mouse_y = 0;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = 1;
            else if (e.type == SDL_MOUSEBUTTONDOWN) mouse_down = 1;
            else if (e.type == SDL_MOUSEBUTTONUP) mouse_down = 0;
            else if (e.type == SDL_MOUSEMOTION) SDL_GetMouseState(&mouse_x, &mouse_y);
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_c) memset(s.buf, 0, (size_t)width * height);
                else if (e.key.keysym.sym == SDLK_r) randomize(&s, 0.3);
            }
        }
        if (mouse_down) add_sand(&s, mouse_x, mouse_y, 5);
        update(s.buf, s.width, s.height);
        render(&s);
        SDL_Delay(16);
    }

    SDL_DestroyTexture(s.texture);
    SDL_DestroyRenderer(s.renderer);
    SDL_DestroyWindow(s.window);
    SDL_Quit();
    free(s.buf);
    free(s.pixels);
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
