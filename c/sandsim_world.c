/*
 * sandsim - C chunked streaming world (Noita-style)
 *
 * Port of cpp/sandsim_world.cpp: a world larger than memory, with only a few
 * "live boxes" (chunks) resident around a camera and the rest streamed to disk.
 * See WORLD.md. Resident chunks are held in a dense 2D pointer array and
 * processed bottom-chunk-first, which reproduces the C++ reference checksum.
 *
 * Modes:
 *   (default)                 SDL2 window; WASD/arrows pan, number keys paint.
 *   --bench [steps] [wch] [hch]   headless deterministic streaming benchmark.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <SDL2/SDL.h>

enum { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };
#define CHUNK 64
#define CHUNK_MASK 63
#define CHUNK_SHIFT 6
#define PIXEL_SIZE 2

static const uint32_t COLORS[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu,
};

/* --- shared rule (identical to the materials/world ports) ----------------- */
static inline int can_enter(uint8_t mover, uint8_t target) {
    if (target == WALL) return 0;
    switch (mover) {
        case SAND:  return target == EMPTY || target == WATER || target == GAS;
        case WATER: return target == EMPTY || target == GAS;
        case GAS:   return target == EMPTY;
        default:    return 0;
    }
}
static inline uint32_t hash_coord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}
static inline uint8_t gen_cell(int gx, int gy, int wcells, int hcells) {
    if (gx == 0 || gy == 0 || gx == wcells - 1 || gy == hcells - 1) return WALL;
    if (gy % 40 == 39 && (gx % 11 != 0)) return WALL;
    uint32_t r = hash_coord(gx, gy) % 100u;
    switch ((gy / 40) % 3) {
        case 0:  return (r < 35u) ? SAND  : EMPTY;
        case 1:  return (r < 30u) ? WATER : EMPTY;
        default: return (r < 18u) ? GAS   : EMPTY;
    }
}

typedef struct {
    uint8_t cells[CHUNK * CHUNK];
    uint8_t moved[CHUNK * CHUNK];
    int dminx, dminy, dmaxx, dmaxy;     /* this frame's dirty rect */
    int nminx, nminy, nmaxx, nmaxy;     /* next frame's accumulator */
} Chunk;

typedef struct {
    int wchunks, hchunks, wcells, hcells;
    char dir[256];
    Chunk **chunks;                     /* wchunks*hchunks pointers, NULL = unresident */
    uint32_t frame;
    int resident_max;
    long long n_writes, n_reads, n_generated;
} World;

static void chunk_full_dirty(Chunk *c) { c->dminx = 0; c->dminy = 0; c->dmaxx = CHUNK - 1; c->dmaxy = CHUNK - 1; }
static void chunk_clear_next(Chunk *c) { c->nminx = CHUNK; c->nminy = CHUNK; c->nmaxx = -1; c->nmaxy = -1; }
static int  chunk_awake(const Chunk *c) { return c->dminx <= c->dmaxx && c->dminy <= c->dmaxy; }
static void chunk_commit_next(Chunk *c) {
    c->dminx = c->nminx; c->dminy = c->nminy; c->dmaxx = c->nmaxx; c->dmaxy = c->nmaxy;
    chunk_clear_next(c);
}

static Chunk *resident_at(World *w, int cx, int cy) {
    if (cx < 0 || cy < 0 || cx >= w->wchunks || cy >= w->hchunks) return NULL;
    return w->chunks[cy * w->wchunks + cx];
}

static void chunk_path(World *w, int cx, int cy, char *out, size_t n) {
    snprintf(out, n, "%s/c_%d_%d.bin", w->dir, cx, cy);
}
static void write_chunk(World *w, int cx, int cy, const Chunk *c) {
    char p[320]; chunk_path(w, cx, cy, p, sizeof(p));
    FILE *f = fopen(p, "wb");
    if (f) { fwrite(c->cells, 1, CHUNK * CHUNK, f); fclose(f); w->n_writes++; }
}
static int read_chunk_raw(World *w, int cx, int cy, uint8_t *cells) {
    char p[320]; chunk_path(w, cx, cy, p, sizeof(p));
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    size_t got = fread(cells, 1, CHUNK * CHUNK, f);
    fclose(f);
    return got == CHUNK * CHUNK;
}

static World *world_new(int wch, int hch, const char *dir) {
    World *w = calloc(1, sizeof(World));
    w->wchunks = wch; w->hchunks = hch; w->wcells = wch * CHUNK; w->hcells = hch * CHUNK;
    snprintf(w->dir, sizeof(w->dir), "%s", dir);
    mkdir(w->dir, 0777);
    w->chunks = calloc((size_t)wch * hch, sizeof(Chunk *));
    return w;
}

static uint8_t world_get(World *w, int gx, int gy) {
    if (gx < 0 || gy < 0 || gx >= w->wcells || gy >= w->hcells) return WALL;
    Chunk *c = resident_at(w, gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT);
    if (!c) return WALL;
    return c->cells[(gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK)];
}

static void generate_all_to_disk(World *w) {
    Chunk *ch = malloc(sizeof(Chunk));
    for (int cy = 0; cy < w->hchunks; ++cy)
        for (int cx = 0; cx < w->wchunks; ++cx) {
            for (int ly = 0; ly < CHUNK; ++ly)
                for (int lx = 0; lx < CHUNK; ++lx)
                    ch->cells[ly * CHUNK + lx] = gen_cell(cx * CHUNK + lx, cy * CHUNK + ly, w->wcells, w->hcells);
            write_chunk(w, cx, cy, ch);
        }
    free(ch);
}

static void load_or_generate(World *w, int cx, int cy) {
    Chunk *ch = malloc(sizeof(Chunk));
    memset(ch->moved, 0, CHUNK * CHUNK);
    if (read_chunk_raw(w, cx, cy, ch->cells)) w->n_reads++;
    else {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                ch->cells[ly * CHUNK + lx] = gen_cell(cx * CHUNK + lx, cy * CHUNK + ly, w->wcells, w->hcells);
        w->n_generated++;
    }
    chunk_full_dirty(ch);
    chunk_clear_next(ch);
    w->chunks[cy * w->wchunks + cx] = ch;
}

static void stream_around(World *w, int cam_cx, int cam_cy, int radius) {
    for (int cy = 0; cy < w->hchunks; ++cy)
        for (int cx = 0; cx < w->wchunks; ++cx) {
            Chunk *c = w->chunks[cy * w->wchunks + cx];
            if (!c) continue;
            if (abs(cx - cam_cx) > radius || abs(cy - cam_cy) > radius) {
                write_chunk(w, cx, cy, c);
                free(c);
                w->chunks[cy * w->wchunks + cx] = NULL;
            }
        }
    for (int cy = cam_cy - radius; cy <= cam_cy + radius; ++cy)
        for (int cx = cam_cx - radius; cx <= cam_cx + radius; ++cx) {
            if (cx < 0 || cy < 0 || cx >= w->wchunks || cy >= w->hchunks) continue;
            if (!resident_at(w, cx, cy)) load_or_generate(w, cx, cy);
        }
    int res = 0;
    for (int i = 0; i < w->wchunks * w->hchunks; ++i) if (w->chunks[i]) res++;
    if (res > w->resident_max) w->resident_max = res;
}

static void wake(World *w, int gx, int gy) {
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = gx + dx, ny = gy + dy;
            Chunk *c = resident_at(w, nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT);
            if (!c) continue;
            int lx = nx & CHUNK_MASK, ly = ny & CHUNK_MASK;
            if (lx < c->nminx) c->nminx = lx;
            if (ly < c->nminy) c->nminy = ly;
            if (lx > c->nmaxx) c->nmaxx = lx;
            if (ly > c->nmaxy) c->nmaxy = ly;
        }
}

static int try_move(World *w, int gx, int gy, int nx, int ny) {
    if (nx < 0 || ny < 0 || nx >= w->wcells || ny >= w->hcells) return 0;
    Chunk *tc = resident_at(w, nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT);
    if (!tc) return 0;
    int ti = (ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK);
    if (tc->moved[ti]) return 0;
    uint8_t target = tc->cells[ti];
    Chunk *sc = resident_at(w, gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT);
    int si = (gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK);
    uint8_t self = sc->cells[si];
    if (!can_enter(self, target)) return 0;
    tc->cells[ti] = self;
    sc->cells[si] = target;
    tc->moved[ti] = 1;
    sc->moved[si] = 1;
    wake(w, gx, gy);
    wake(w, nx, ny);
    return 1;
}

static void world_step(World *w) {
    for (int i = 0; i < w->wchunks * w->hchunks; ++i)
        if (w->chunks[i]) memset(w->chunks[i]->moved, 0, CHUNK * CHUNK);

    /* bottom chunks first (cy high->low), then left-to-right */
    for (int cy = w->hchunks - 1; cy >= 0; --cy)
        for (int cx = 0; cx < w->wchunks; ++cx) {
            Chunk *c = w->chunks[cy * w->wchunks + cx];
            if (!c || !chunk_awake(c)) continue;
            int baseX = cx * CHUNK, baseY = cy * CHUNK;
            for (int ly = c->dmaxy; ly >= c->dminy; --ly)
                for (int lx = c->dminx; lx <= c->dmaxx; ++lx) {
                    if (c->moved[ly * CHUNK + lx]) continue;
                    int gx = baseX + lx, gy = baseY + ly;
                    uint8_t m = c->cells[ly * CHUNK + lx];
                    if (m == EMPTY || m == WALL) continue;
                    int left = (((uint32_t)gx + (uint32_t)gy + w->frame) & 1u) == 0u;
                    int d1 = left ? -1 : 1, d2 = -d1;
                    if (m == SAND || m == WATER) {
                        if (try_move(w, gx, gy, gx, gy + 1)) continue;
                        if (try_move(w, gx, gy, gx + d1, gy + 1)) continue;
                        if (try_move(w, gx, gy, gx + d2, gy + 1)) continue;
                        if (m == WATER) {
                            if (try_move(w, gx, gy, gx + d1, gy)) continue;
                            if (try_move(w, gx, gy, gx + d2, gy)) continue;
                        }
                    } else { /* GAS */
                        if (try_move(w, gx, gy, gx, gy - 1)) continue;
                        if (try_move(w, gx, gy, gx + d1, gy - 1)) continue;
                        if (try_move(w, gx, gy, gx + d2, gy - 1)) continue;
                        if (try_move(w, gx, gy, gx + d1, gy)) continue;
                        if (try_move(w, gx, gy, gx + d2, gy)) continue;
                    }
                }
        }

    for (int i = 0; i < w->wchunks * w->hchunks; ++i)
        if (w->chunks[i]) chunk_commit_next(w->chunks[i]);
    w->frame++;
}

/* The world border (generated as WALL) is the indestructible solid shell,
   including the bottom floor: painting never modifies it. */
static int indestructible(World *w, int gx, int gy) {
    return gx == 0 || gy == 0 || gx == w->wcells - 1 || gy == w->hcells - 1;
}

static void world_paint(World *w, int gx, int gy, uint8_t material, int radius) {
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = gx + dx, ny = gy + dy;
            if (dx * dx + dy * dy > radius * radius) continue;
            if (indestructible(w, nx, ny)) continue;
            Chunk *c = resident_at(w, nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT);
            if (!c) continue;
            c->cells[(ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK)] = material;
            wake(w, nx, ny);
        }
}

static void world_summary(World *w, uint64_t *checksum, uint64_t counts[MATERIAL_COUNT]) {
    for (int i = 0; i < MATERIAL_COUNT; ++i) counts[i] = 0;
    uint64_t c = 14695981039346656037ull;
    uint8_t buf[CHUNK * CHUNK];
    for (int cy = 0; cy < w->hchunks; ++cy)
        for (int cx = 0; cx < w->wchunks; ++cx) {
            const uint8_t *cells;
            Chunk *res = resident_at(w, cx, cy);
            if (res) cells = res->cells;
            else { read_chunk_raw(w, cx, cy, buf); cells = buf; }
            for (int i = 0; i < CHUNK * CHUNK; ++i) {
                uint8_t v = cells[i];
                counts[v]++;
                c = (c ^ (uint64_t)v) * 1099511628211ull;
            }
        }
    *checksum = c;
}

/* --- headless benchmark --------------------------------------------------- */
static int run_bench(int steps, int wch, int hch) {
    char dir[256];
    snprintf(dir, sizeof(dir), "/tmp/sandsim_world_c_%d_%dx%d", steps, wch, hch);
    World *w = world_new(wch, hch, dir);
    generate_all_to_disk(w);

    uint64_t start_ck, start_cnt[MATERIAL_COUNT];
    world_summary(w, &start_ck, start_cnt);

    int cells = wch * hch;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * cells / steps);
        if (visit >= cells) visit = cells - 1;
        int row = visit / wch, col = visit % wch;
        int cam_cx = (row % 2 == 0) ? col : (wch - 1 - col);
        int cam_cy = row;
        stream_around(w, cam_cx, cam_cy, 1);
        world_step(w);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    uint64_t ck, cnt[MATERIAL_COUNT];
    world_summary(w, &ck, cnt);
    int conserved = 1;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != start_cnt[i]) conserved = 0;

    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1.0e6;
    printf("RESULT impl=c_world rule=world wchunks=%d hchunks=%d chunk=%d steps=%d "
           "elapsed_ms=%.3f checksum=%016llx empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           wch, hch, CHUNK, steps, elapsed_ms, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           w->resident_max, w->n_writes, w->n_reads, conserved ? "yes" : "no");
    return conserved ? 0 : 2;
}

/* --- interactive ---------------------------------------------------------- */
static int run_interactive(void) {
    const int VIEW_W = 320, VIEW_H = 240, WCH = 64, HCH = 64;
    int rw = VIEW_W * PIXEL_SIZE, rh = VIEW_H * PIXEL_SIZE;
    World *w = world_new(WCH, HCH, "/tmp/sandsim_world_c_interactive");
    int camX = WCH * CHUNK / 2 - VIEW_W / 2, camY = HCH * CHUNK / 2 - VIEW_H / 2;
    uint8_t current = SAND;
    uint32_t *pixels = calloc((size_t)rw * rh, sizeof(uint32_t));

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow(
        "Streamed World (C) - WASD/arrows pan  [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, rw, rh, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    /* Map rendering and the cursor through a fixed logical size, so painting
       lands under the pointer even when a tiling compositor (e.g. niri) resizes
       the window away from the requested size. */
    SDL_RenderSetLogicalSize(renderer, rw, rh);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING, rw, rh);
    int quit = 0, mouse_down = 0, mx = 0, my = 0;
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
                }
            }
        }
        /* Arrow keys or WASD pan the camera while held (smooth, continuous). */
        const Uint8 *keys = SDL_GetKeyboardState(NULL);
        const int pan = 6;
        if (keys[SDL_SCANCODE_LEFT]  || keys[SDL_SCANCODE_A]) camX -= pan;
        if (keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D]) camX += pan;
        if (keys[SDL_SCANCODE_UP]    || keys[SDL_SCANCODE_W]) camY -= pan;
        if (keys[SDL_SCANCODE_DOWN]  || keys[SDL_SCANCODE_S]) camY += pan;

        stream_around(w, (camX + VIEW_W / 2) >> CHUNK_SHIFT, (camY + VIEW_H / 2) >> CHUNK_SHIFT, 3);
        if (mouse_down) {
            float lx, ly;
            SDL_RenderWindowToLogical(renderer, mx, my, &lx, &ly);
            world_paint(w, camX + (int)lx / PIXEL_SIZE, camY + (int)ly / PIXEL_SIZE, current, 4);
        }
        world_step(w);
        for (int vy = 0; vy < VIEW_H; ++vy)
            for (int vx = 0; vx < VIEW_W; ++vx) {
                uint32_t color = COLORS[world_get(w, camX + vx, camY + vy)];
                for (int dy = 0; dy < PIXEL_SIZE; ++dy)
                    for (int dx = 0; dx < PIXEL_SIZE; ++dx)
                        pixels[(size_t)(vy * PIXEL_SIZE + dy) * rw + (vx * PIXEL_SIZE + dx)] = color;
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
    free(pixels);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        int steps = (argc > 2) ? atoi(argv[2]) : 600;
        int wch   = (argc > 3) ? atoi(argv[3]) : 6;
        int hch   = (argc > 4) ? atoi(argv[4]) : 6;
        return run_bench(steps, wch, hch);
    }
    return run_interactive();
}
