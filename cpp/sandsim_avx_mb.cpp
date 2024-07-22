#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <SDL2/SDL.h>
#include <iostream>

class AVXSandSimulation {
private:
    static constexpr int NUM_BUFFERS = 32;  // AVX operates on 256 bits (32 bytes)
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

    const __m256i is_sand_const = _mm256_set1_epi8(1);
    const __m256i is_empty_const = _mm256_setzero_si256();

public:
    AVXSandSimulation(int w, int h) : width(w), height(h),
                                       renderWidth(w * PIXEL_SIZE),
                                       renderHeight(h * PIXEL_SIZE),
                                       rng(std::random_device{}()) {
        buffer.resize(NUM_BUFFERS * width * height, 0);
        pixelBuffer.resize(renderWidth * renderHeight, 0);

        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("AVX Sand Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderWidth, renderHeight, 0);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, renderWidth, renderHeight);
    }

    ~AVXSandSimulation() {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void update() {
        for (int y = height - 2; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                __m256i current = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[getIndex(x, y)]));
                __m256i below = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[getIndex(x, y + 1)]));
                __m256i left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[getIndex((x - 1 + width) % width, y + 1)]));
                __m256i right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[getIndex((x + 1) % width, y + 1)]));

                /////////////////////////////////////

                // Check which cells contain sand and which cells below are empty
                __m256i is_sand = _mm256_cmpeq_epi8(current, is_sand_const);
                __m256i is_below_empty = _mm256_cmpeq_epi8(below, is_empty_const);

                // Determine which sand particles can fall straight down
                __m256i can_fall = _mm256_and_si256(is_sand, is_below_empty);

                // Update row below (add sand that has fallen)
                below = _mm256_or_si256(below, _mm256_and_si256(can_fall, is_sand_const));

                // Update current row (remove sand that will fall)
                current = _mm256_andnot_si256(can_fall, current);

                /////////////////////////////////////

                // Check which cells still contain sand
                is_sand = _mm256_cmpeq_epi8(current, is_sand_const);

                // Check which cells below-left are empty
                __m256i is_left_empty = _mm256_cmpeq_epi8(left, is_empty_const);

                // Determine which sand particles can fall to the left
                can_fall = _mm256_and_si256(is_sand, is_left_empty);

                // Update row below-left (add sand that has fallen)
                left = _mm256_or_si256(left, _mm256_and_si256(can_fall, is_sand_const));

                // Update current row (remove sand that will fall)
                current = _mm256_andnot_si256(can_fall, current);

                /////////////////////////////////////

                // Check which cells still contain sand
                is_sand = _mm256_cmpeq_epi8(current, is_sand_const);

                // Check which cells below-right are empty
                __m256i is_right_empty = _mm256_cmpeq_epi8(right, is_empty_const);

                // Determine which sand particles can fall to the right
                can_fall = _mm256_and_si256(is_sand, is_right_empty);

                // Update row below-right (add sand that has fallen)
                right = _mm256_or_si256(right, _mm256_and_si256(can_fall, is_sand_const));

                // Update current row (remove sand that will fall)
                current = _mm256_andnot_si256(can_fall, current);

                /////////////////////////////////////

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[getIndex(x, y)]), current);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[getIndex(x, y + 1)]), below);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[getIndex((x - 1 + width) % width, y + 1)]), left);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[getIndex((x + 1) % width, y + 1)]), right);
            }
        }
    }

    void addSand(int bufferIndex, int x, int y, int radius = 5) {
        if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS) return;
        // Convert screen coordinates to simulation coordinates
        x /= PIXEL_SIZE;
        y /= PIXEL_SIZE;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (dx*dx + dy*dy <= radius*radius) {
                        buffer[getIndex(nx, ny) + bufferIndex] = 1;
                    }
                }
            }
        }
    }

    uint8_t getSand(int bufferIndex, int x, int y) const {
        if (bufferIndex < 0 || bufferIndex >= NUM_BUFFERS || x < 0 || x >= width || y < 0 || y >= height) {
            return 0;
        }
        return buffer[getIndex(x, y) + bufferIndex];
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

    void render(int activeBuffer) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t sandValue = getSand(activeBuffer, x, y);
                uint32_t color = sandValue ? 0xFFFFFF00 : 0xFF000000; // Yellow sand, black background

                // Fill a PIXEL_SIZE x PIXEL_SIZE square in the pixelBuffer
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
    size_t getIndex(int x, int y) const {
        return (y * width + x) * NUM_BUFFERS;
    }
};

int main(int argc, char* argv[]) {
    AVXSandSimulation sim(400, 300);  // 400x300 simulation size
    sim.run();
    return 0;
}
