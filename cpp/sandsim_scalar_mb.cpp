#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <SDL2/SDL.h>
#include <iostream>

class ScalarSandSimulation {
private:
    static constexpr int NUM_BUFFERS = 16;  // Match the number of buffers in SIMD versions
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
        buffer.resize(NUM_BUFFERS * width * height, 0);
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
                for (int b = 0; b < NUM_BUFFERS; ++b) {
                    if (buffer[getIndex(x, y, b)] == sand_value) {
                        if (buffer[getIndex(x, y + 1, b)] == empty_value) {
                            buffer[getIndex(x, y + 1, b)] = sand_value;
                            buffer[getIndex(x, y, b)] = empty_value;
                        } else if (x > 0 && buffer[getIndex(x - 1, y + 1, b)] == empty_value) {
                            buffer[getIndex(x - 1, y + 1, b)] = sand_value;
                            buffer[getIndex(x, y, b)] = empty_value;
                        } else if (x < width - 1 && buffer[getIndex(x + 1, y + 1, b)] == empty_value) {
                            buffer[getIndex(x + 1, y + 1, b)] = sand_value;
                            buffer[getIndex(x, y, b)] = empty_value;
                        }
                    }
                }
            }
        }
    }

    void addSand(int bufferIndex, int x, int y, int radius = 5) {
        if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS) return;
        x /= PIXEL_SIZE;
        y /= PIXEL_SIZE;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (dx*dx + dy*dy <= radius*radius) {
                        buffer[getIndex(nx, ny, bufferIndex)] = sand_value;
                    }
                }
            }
        }
    }

    uint8_t getSand(int bufferIndex, int x, int y) const {
        if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS || x < 0 || x >= width || y < 0 || y >= height) {
            return 0;
        }
        return buffer[getIndex(x, y, bufferIndex)];
    }

    void clear() {
        std::fill(buffer.begin(), buffer.end(), empty_value);
    }

    void randomize(float density = 0.3f) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < buffer.size(); ++i) {
            buffer[i] = (dist(rng) < density) ? sand_value : empty_value;
        }
    }

    void render(int activeBuffer) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t sandValue = getSand(activeBuffer, x, y);
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
        int activeBuffer = 0;
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
                    if (event.key.keysym.sym == SDLK_SPACE) {
                        activeBuffer = (activeBuffer + 1) % NUM_BUFFERS;
                        std::cout << "Switched to buffer: " << activeBuffer << std::endl;
                    } else if (event.key.keysym.sym == SDLK_c) {
                        clear();
                    } else if (event.key.keysym.sym == SDLK_r) {
                        randomize();
                    } else if (event.key.keysym.sym >= SDLK_0 && event.key.keysym.sym <= SDLK_9) {
                        int bufferIndex = event.key.keysym.sym - SDLK_0;
                        if (bufferIndex < NUM_BUFFERS) {
                            activeBuffer = bufferIndex;
                            std::cout << "Switched to buffer: " << activeBuffer << std::endl;
                        }
                    }
                }
            }

            if (mouseDown) {
                addSand(activeBuffer, mouseX, mouseY, 5);
            }

            update();
            render(activeBuffer);

            SDL_Delay(16); // Cap at roughly 60 FPS
        }
    }

private:
    size_t getIndex(int x, int y, int bufferIndex) const {
        return (y * width + x) * NUM_BUFFERS + bufferIndex;
    }
};

int main(int argc, char* argv[]) {
    ScalarSandSimulation sim(400, 300);
    sim.run();
    return 0;
}
