// sandsim - HIP implementation
//
// GPU falling-sand using a HIP compute kernel, following the same model as the
// Vulkan/OpenGL versions: a device buffer holds the grid twice (a "src" half
// and a "dst" half). Each step the host copies src->dst, then the kernel
// atomically claims destination cells (atomicCAS 0->1) and clears the source
// position, so unmoved particles stay and no two particles share a cell. This
// is the "gpu" rule group; like the other GPU versions its checksum may vary
// run to run (atomic contention is resolved by scheduling order) while the
// sand count is conserved and printed as a deterministic check.
//
// Modes:
//   (default)                 SDL2 window, 400x300 grid rendered 2x (800x600).
//   --bench [steps] [w] [h]   headless: time N steps on the GPU, print RESULT.
//
// Builds with hipcc and runs on ROCm (AMD) or, via HIP's CUDA backend, NVIDIA.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

#include <SDL2/SDL.h>

static constexpr int PIXEL_SIZE = 2;

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _e = (call);                                               \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Shared host helpers (seed + checksum identical to every implementation).
// ---------------------------------------------------------------------------
static inline uint32_t seedCell(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (h % 100u) < 30u ? 1u : 0u;
}
static uint64_t checksum(const std::vector<uint32_t>& grid) {
    uint64_t c = 14695981039346656037ull;
    for (uint32_t cell : grid) c = (c ^ (uint64_t)(cell & 1u)) * 1099511628211ull;
    return c;
}

// ---------------------------------------------------------------------------
// Device kernel.
// ---------------------------------------------------------------------------
__device__ inline uint32_t readCell(const uint32_t* cells, int width, int height,
                                    int x, int y, int half) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 1u; // OOB = solid
    return cells[(size_t)half * width * height + (size_t)y * width + x];
}
__device__ inline bool claimDst(uint32_t* cells, int width, int height,
                                int dst, int x, int y) {
    return atomicCAS(&cells[(size_t)dst * width * height + (size_t)y * width + x],
                     0u, 1u) == 0u;
}

__global__ void sandStep(uint32_t* cells, int width, int height, int src, int dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (readCell(cells, width, height, x, y, src) != 1u) return;

    size_t self = (size_t)dst * width * height + (size_t)y * width + x;
    if (readCell(cells, width, height, x, y + 1, src) == 0u &&
        claimDst(cells, width, height, dst, x, y + 1)) { cells[self] = 0u; return; }

    bool preferLeft = (((x + y) & 1) == 0);
    int dx1 = preferLeft ? -1 : 1;
    int dx2 = -dx1;
    if (readCell(cells, width, height, x + dx1, y + 1, src) == 0u &&
        claimDst(cells, width, height, dst, x + dx1, y + 1)) { cells[self] = 0u; return; }
    if (readCell(cells, width, height, x + dx2, y + 1, src) == 0u &&
        claimDst(cells, width, height, dst, x + dx2, y + 1)) { cells[self] = 0u; return; }
}

// One step: copy cur->other on the device, run the kernel src=cur dst=other.
static void step(uint32_t* d_cells, int width, int height, int cur) {
    int other = 1 - cur;
    size_t halfElems = (size_t)width * height;
    HIP_CHECK(hipMemcpy(d_cells + (size_t)other * halfElems,
                        d_cells + (size_t)cur * halfElems,
                        halfElems * sizeof(uint32_t), hipMemcpyDeviceToDevice));
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    sandStep<<<grid, block, 0, 0>>>(d_cells, width, height, cur, other);
    HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Headless benchmark.
// ---------------------------------------------------------------------------
static int runBench(int steps, int width, int height) {
    size_t halfElems = (size_t)width * height;
    std::vector<uint32_t> host(halfElems * 2, 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            host[(size_t)y * width + x] = seedCell(x, y); // half 0 = src

    uint32_t* d_cells = nullptr;
    HIP_CHECK(hipMalloc(&d_cells, host.size() * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_cells, host.data(), host.size() * sizeof(uint32_t), hipMemcpyHostToDevice));

    int cur = 0;
    HIP_CHECK(hipDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) { step(d_cells, width, height, cur); cur = 1 - cur; }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    std::vector<uint32_t> out(halfElems);
    HIP_CHECK(hipMemcpy(out.data(), d_cells + (size_t)cur * halfElems,
                        halfElems * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_cells));

    uint64_t sand = 0;
    for (uint32_t v : out) sand += (v & 1u);
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    double cells = (double)width * height * steps;
    double mcells = (elapsedMs > 0.0) ? cells / (elapsedMs / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=hip rule=gpu width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx sand=%llu\n",
           width, height, steps, elapsedMs, mcells,
           (unsigned long long)checksum(out), (unsigned long long)sand);
    return 0;
}

// ---------------------------------------------------------------------------
// Interactive SDL2 mode (device->host copy each frame for display).
// ---------------------------------------------------------------------------
static int runInteractive(int width, int height) {
    int renderW = width * PIXEL_SIZE, renderH = height * PIXEL_SIZE;
    size_t halfElems = (size_t)width * height;
    std::vector<uint32_t> host(halfElems, 0);
    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);

    uint32_t* d_cells = nullptr;
    HIP_CHECK(hipMalloc(&d_cells, halfElems * 2 * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_cells, 0, halfElems * 2 * sizeof(uint32_t)));
    int cur = 0;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }
    SDL_Window* window = SDL_CreateWindow("HIP Sand Simulation",
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          renderW, renderH, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
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
                if (e.key.keysym.sym == SDLK_c) std::fill(host.begin(), host.end(), 0u);
                else if (e.key.keysym.sym == SDLK_r)
                    for (auto& v : host) v = ((double)rand() / RAND_MAX < 0.3) ? 1u : 0u;
            }
        }
        if (mouseDown) {
            int cx = mouseX / PIXEL_SIZE, cy = mouseY / PIXEL_SIZE, r = 5;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && dx*dx + dy*dy <= r*r)
                        host[(size_t)ny * width + nx] = 1u;
                }
        }

        // Upload edits to the current half, simulate one step, read result back.
        HIP_CHECK(hipMemcpy(d_cells + (size_t)cur * halfElems, host.data(),
                            halfElems * sizeof(uint32_t), hipMemcpyHostToDevice));
        step(d_cells, width, height, cur);
        cur = 1 - cur;
        HIP_CHECK(hipMemcpy(host.data(), d_cells + (size_t)cur * halfElems,
                            halfElems * sizeof(uint32_t), hipMemcpyDeviceToHost));

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                uint32_t color = (host[(size_t)y * width + x] & 1u) ? 0xFFFFFF00u : 0xFF000000u;
                for (int dy = 0; dy < PIXEL_SIZE; ++dy)
                    for (int dx = 0; dx < PIXEL_SIZE; ++dx)
                        pixels[(size_t)(y * PIXEL_SIZE + dy) * renderW + (x * PIXEL_SIZE + dx)] = color;
            }
        SDL_UpdateTexture(texture, nullptr, pixels.data(), renderW * (int)sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    HIP_CHECK(hipFree(d_cells));
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        int steps  = (argc > 2) ? atoi(argv[2]) : 1000;
        int width  = (argc > 3) ? atoi(argv[3]) : 400;
        int height = (argc > 4) ? atoi(argv[4]) : 300;
        return runBench(steps, width, height);
    }
    return runInteractive(400, 300);
}
