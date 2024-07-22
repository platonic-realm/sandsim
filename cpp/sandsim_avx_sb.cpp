#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <SDL2/SDL.h>
#include <iostream>

class AVXSandSimulation {
private:
    static constexpr int PIXEL_SIZE = 2;
    int width;
    int height;
    int renderWidth;
    int renderHeight;
    int gridWidth;  // Rounded up to multiple of 32 for AVX
    std::vector<uint8_t> buffer;
    std::mt19937 rng;

    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    std::vector<uint32_t> pixelBuffer;

    const __m256i sand_value = _mm256_set1_epi8(1);
    const __m256i empty_value = _mm256_setzero_si256();

public:
    AVXSandSimulation(int w, int h) : width(w), height(h),
                                       renderWidth(w * PIXEL_SIZE),
                                       renderHeight(h * PIXEL_SIZE),
                                       gridWidth((w + 31) & ~31),  // Round up to multiple of 32 for AVX
                                       rng(std::random_device{}()) {
        buffer.resize(gridWidth * height, 0);
        pixelBuffer.resize(renderWidth * renderHeight, 0);

        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("AVX Sand Simulation (Single Buffer)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderWidth, renderHeight, 0);
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
            for (int x = 0; x < gridWidth; x += 32) {
                __m256i current = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[y * gridWidth + x]));
                __m256i is_sand = _mm256_cmpeq_epi8(current, sand_value);

                // Update below
                __m256i below = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[(y + 1) * gridWidth + x]));
                __m256i is_empty_below = _mm256_cmpeq_epi8(below, empty_value);
                __m256i can_fall = _mm256_and_si256(is_sand, is_empty_below);
                current = _mm256_andnot_si256(can_fall, current);
                below = _mm256_or_si256(below, _mm256_and_si256(can_fall, sand_value));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[(y + 1) * gridWidth + x]), below);

                // Update below-left
                __m256i left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[(y + 1) * gridWidth + x - 1]));
                __m256i is_empty_left = _mm256_cmpeq_epi8(left, empty_value);
                is_sand = _mm256_cmpeq_epi8(current, sand_value);
                can_fall = _mm256_and_si256(is_sand, is_empty_left);
                current = _mm256_andnot_si256(can_fall, current);
                left = _mm256_or_si256(left, _mm256_and_si256(can_fall, sand_value));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[(y + 1) * gridWidth + x - 1]), left);

                // Update below-right
                __m256i right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&buffer[(y + 1) * gridWidth + x + 1]));
                __m256i is_empty_right = _mm256_cmpeq_epi8(right, empty_value);
                is_sand = _mm256_cmpeq_epi8(current, sand_value);
                can_fall = _mm256_and_si256(is_sand, is_empty_right);
                current = _mm256_andnot_si256(can_fall, current);
                right = _mm256_or_si256(right, _mm256_and_si256(can_fall, sand_value));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[(y + 1) * gridWidth + x + 1]), right);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buffer[y * gridWidth + x]), current);
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
                        buffer[ny * gridWidth + nx] = 1;
                    }
                }
            }
        }
    }

    uint8_t getSand(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 0;
        }
        return buffer[y * gridWidth + x];
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
                uint8_t sandValue = getSand(x, y);
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

int main(int argc, char* argv[]) {
    AVXSandSimulation sim(400, 300);  // 400x300 simulation size
    sim.run();
    return 0;
}
