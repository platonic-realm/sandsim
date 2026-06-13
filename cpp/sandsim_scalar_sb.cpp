#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <SDL2/SDL.h>

class ScalarSandSimulation {
private:
    static constexpr int PIXEL_SIZE = 2;
    int width;
    int height;
    int renderWidth;
    int renderHeight;
    std::vector<uint8_t> buffer;
    std::mt19937 rng;

    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    std::vector<uint32_t> pixelBuffer;

    const uint8_t sand_value = 1;
    const uint8_t empty_value = 0;

public:
    ScalarSandSimulation(int w, int h) : width(w), height(h),
                                         renderWidth(w * PIXEL_SIZE),
                                         renderHeight(h * PIXEL_SIZE),
                                         rng(std::random_device{}()) {
        buffer.resize(width * height, 0);
        pixelBuffer.resize(renderWidth * renderHeight, 0);

        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("Scalar Sand Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderWidth, renderHeight, 0);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, renderWidth, renderHeight);
    }

    ~ScalarSandSimulation() {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void update() {
        for (int y = height - 2; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                if (buffer[y * width + x] == sand_value) {
                    if (buffer[(y + 1) * width + x] == empty_value) {
                        buffer[(y + 1) * width + x] = sand_value;
                        buffer[y * width + x] = empty_value;
                    } else if (x > 0 && buffer[(y + 1) * width + (x - 1)] == empty_value) {
                        buffer[(y + 1) * width + (x - 1)] = sand_value;
                        buffer[y * width + x] = empty_value;
                    } else if (x < width - 1 && buffer[(y + 1) * width + (x + 1)] == empty_value) {
                        buffer[(y + 1) * width + (x + 1)] = sand_value;
                        buffer[y * width + x] = empty_value;
                    }
                }
            }
        }
    }

    void addSand(int x, int y, int radius = 5) {
        x /= PIXEL_SIZE;
        y /= PIXEL_SIZE;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (dx*dx + dy*dy <= radius*radius) {
                        buffer[ny * width + nx] = 1;
                    }
                }
            }
        }
    }

    void clear() {
        std::fill(buffer.begin(), buffer.end(), 0);
    }

    void randomize(float density = 0.3f) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < buffer.size(); ++i) {
            buffer[i] = (dist(rng) < density) ? 1 : 0;
        }
    }

    void render() {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t sandValue = buffer[y * width + x];
                uint32_t color = sandValue ? 0xFFFFFF00 : 0xFF000000; // Yellow sand, black background

                for (int dy = 0; dy < PIXEL_SIZE; ++dy) {
                    for (int dx = 0; dx < PIXEL_SIZE; ++dx) {
                        int renderX = x * PIXEL_SIZE + dx;
                        int renderY = y * PIXEL_SIZE + dy;
                        pixelBuffer[renderY * renderWidth + renderX] = color;
                    }
                }
            }
        }

        SDL_UpdateTexture(texture, NULL, pixelBuffer.data(), renderWidth * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    void run() {
        bool quit = false;
        SDL_Event event;
        bool mouseDown = false;
        int mouseX, mouseY;

        while (!quit) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    quit = true;
                } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                    mouseDown = true;
                } else if (event.type == SDL_MOUSEBUTTONUP) {
                    mouseDown = false;
                } else if (event.type == SDL_MOUSEMOTION) {
                    SDL_GetMouseState(&mouseX, &mouseY);
                } else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym == SDLK_c) {
                        clear();
                    } else if (event.key.keysym.sym == SDLK_r) {
                        randomize();
                    }
                }
            }

            if (mouseDown) {
                addSand(mouseX, mouseY, 5);
            }

            update();
            render();

            SDL_Delay(16); // Cap at roughly 60 FPS
        }
    }
};

// ---------------------------------------------------------------------------
// Headless benchmark mode. This is the golden reference for the "scalar" rule
// group: every CPU port (C, Rust, Zig) must reproduce these exact numbers
// from the same deterministic seed. The update rule below is identical to
// ScalarSandSimulation::update() above. Nothing here touches SDL.
// ---------------------------------------------------------------------------

// Deterministic per-cell seed (~30% sand). Pure u32 wraparound arithmetic so
// every language produces bit-identical grids.
static inline uint8_t seedCell(int x, int y) {
    uint32_t h = static_cast<uint32_t>(x) * 374761393u +
                 static_cast<uint32_t>(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (h % 100u) < 30u ? 1 : 0;
}

// FNV-1a over the grid in row-major order, u64 wraparound.
static uint64_t checksum(const std::vector<uint8_t>& grid) {
    uint64_t c = 14695981039346656037ull;
    for (uint8_t cell : grid) {
        c = (c ^ static_cast<uint64_t>(cell)) * 1099511628211ull;
    }
    return c;
}

static int runBench(int steps, int width, int height) {
    std::vector<uint8_t> buffer(static_cast<size_t>(width) * height, 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            buffer[y * width + x] = seedCell(x, y);

    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        for (int y = height - 2; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                if (buffer[y * width + x] == 1) {
                    if (buffer[(y + 1) * width + x] == 0) {
                        buffer[(y + 1) * width + x] = 1;
                        buffer[y * width + x] = 0;
                    } else if (x > 0 && buffer[(y + 1) * width + (x - 1)] == 0) {
                        buffer[(y + 1) * width + (x - 1)] = 1;
                        buffer[y * width + x] = 0;
                    } else if (x < width - 1 && buffer[(y + 1) * width + (x + 1)] == 0) {
                        buffer[(y + 1) * width + (x + 1)] = 1;
                        buffer[y * width + x] = 0;
                    }
                }
            }
        }
    }
    auto end = std::chrono::steady_clock::now();

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    double cells = static_cast<double>(width) * height * steps;
    double mcellsPerSec = (elapsedMs > 0.0) ? cells / (elapsedMs / 1000.0) / 1e6 : 0.0;

    printf("RESULT impl=cpp_scalar_sb rule=scalar width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx\n",
           width, height, steps, elapsedMs, mcellsPerSec,
           static_cast<unsigned long long>(checksum(buffer)));
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        int steps  = (argc > 2) ? std::atoi(argv[2]) : 1000;
        int width  = (argc > 3) ? std::atoi(argv[3]) : 400;
        int height = (argc > 4) ? std::atoi(argv[4]) : 300;
        return runBench(steps, width, height);
    }

    ScalarSandSimulation sim(400, 300);
    sim.run();
    return 0;
}
