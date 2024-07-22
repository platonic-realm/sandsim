#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <SDL2/SDL.h>
#include <iostream>

class SSESandSimulation {
private:
    static constexpr int PIXEL_SIZE = 2;
    int width;
    int height;
    int renderWidth;
    int renderHeight;
    int gridWidth;  // Rounded up to multiple of 32 for SSE
    std::vector<uint8_t> buffer;
    std::mt19937 rng;

    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    std::vector<uint32_t> pixelBuffer;

    const __m128i sand_value = _mm_set1_epi8(1);
    const __m128i empty_value = _mm_setzero_si128();

public:
    SSESandSimulation(int w, int h) : width(w), height(h),
                                       renderWidth(w * PIXEL_SIZE),
                                       renderHeight(h * PIXEL_SIZE),
                                       gridWidth((w + 31) & ~31),  // Round up to multiple of 32 for SSE
                                       rng(std::random_device{}()) {
        buffer.resize(gridWidth * height, 0);
        pixelBuffer.resize(renderWidth * renderHeight, 0);

        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("SSE Sand Simulation (Single Buffer)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderWidth, renderHeight, 0);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, renderWidth, renderHeight);
    }

    ~SSESandSimulation() {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void update() {
        for (int y = height - 2; y >= 0; --y) {
            for (int x = 0; x < gridWidth; x += 16) {
                __m128i current = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&buffer[y * gridWidth + x]));
                __m128i is_sand = _mm_cmpeq_epi8(current, sand_value);

                // Update below
                __m128i below = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&buffer[(y + 1) * gridWidth + x]));
                __m128i is_empty_below = _mm_cmpeq_epi8(below, empty_value);
                __m128i can_fall = _mm_and_si128(is_sand, is_empty_below);
                current = _mm_andnot_si128(can_fall, current);
                below = _mm_or_si128(below, _mm_and_si128(can_fall, sand_value));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&buffer[(y + 1) * gridWidth + x]), below);

                // Update below-left
                __m128i left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&buffer[(y + 1) * gridWidth + x - 1]));
                __m128i is_empty_left = _mm_cmpeq_epi8(left, empty_value);
                is_sand = _mm_cmpeq_epi8(current, sand_value);
                can_fall = _mm_and_si128(is_sand, is_empty_left);
                current = _mm_andnot_si128(can_fall, current);
                left = _mm_or_si128(left, _mm_and_si128(can_fall, sand_value));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&buffer[(y + 1) * gridWidth + x - 1]), left);

                // Update below-right
                __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&buffer[(y + 1) * gridWidth + x + 1]));
                __m128i is_empty_right = _mm_cmpeq_epi8(right, empty_value);
                is_sand = _mm_cmpeq_epi8(current, sand_value);
                can_fall = _mm_and_si128(is_sand, is_empty_right);
                current = _mm_andnot_si128(can_fall, current);
                right = _mm_or_si128(right, _mm_and_si128(can_fall, sand_value));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&buffer[(y + 1) * gridWidth + x + 1]), right);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(&buffer[y * gridWidth + x]), current);
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
    SSESandSimulation sim(400, 300);  // 400x300 simulation size
    sim.run();
    return 0;
}
