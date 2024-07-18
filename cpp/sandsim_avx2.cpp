#include <SDL2/SDL.h>
#include <vector>
#include <random>
#include <iostream>
#include <immintrin.h>


const int SCREEN_WIDTH = 1600;
const int SCREEN_HEIGHT = 900;
const int GRID_WIDTH = (SCREEN_WIDTH + 15) & ~15;  // Round up to multiple of 16
const int GRID_HEIGHT = SCREEN_HEIGHT;

std::vector<uint8_t> grid(GRID_WIDTH * SCREEN_HEIGHT, 0);
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

void updateSandSIMD() {
    const __m128i sand_value = _mm_set1_epi8(1);  // Sand is represented by 1
    const __m128i empty_value = _mm_setzero_si128();  // Empty is represented by 0

    for (int y = GRID_HEIGHT - 2; y >= 0; --y) {
        for (int x = 0; x < GRID_WIDTH; x += 16) {

            // Load current row
            __m128i current = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&grid[y * GRID_WIDTH + x]));

            // Check which cells contain sand
            __m128i is_sand = _mm_cmpeq_epi8(current, sand_value);

            /////////////////////////////////////

            // Load the row below
            __m128i below = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&grid[(y + 1) * GRID_WIDTH + x]));

            // Check which cells below are empty
            __m128i is_empty_below = _mm_cmpeq_epi8(below, empty_value);

            // Determine which sand particles can fall
            __m128i can_fall = _mm_and_si128(is_sand, is_empty_below);

            // Update current row (remove sand that will fall)
            current = _mm_andnot_si128(can_fall, current);

            // Update row below (add sand that has fallen)
            below = _mm_or_si128(below, _mm_and_si128(can_fall, sand_value));

            // Store updated rows back to grid
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&grid[(y + 1) * GRID_WIDTH + x]), below);

            /////////////////////////////////////

            // Load the row below-left
            __m128i left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&grid[(y + 1) * GRID_WIDTH + x - 1]));

            // Check which cells below-left are empty
            __m128i is_empty_left = _mm_cmpeq_epi8(left, empty_value);

            // Check which cells contain sand
            is_sand = _mm_cmpeq_epi8(current, sand_value);

            // Determine which sand particles can fall
            can_fall = _mm_and_si128(is_sand, is_empty_left);

            // Update current row (remove sand that will fall)
            current = _mm_andnot_si128(can_fall, current);

            // Update row below-left (add sand that has fallen)
            left = _mm_or_si128(left, _mm_and_si128(can_fall, sand_value));

            // Store updated rows back to grid
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&grid[(y + 1) * GRID_WIDTH + x - 1]), left);

            /////////////////////////////////////

            // Load the row below-right
            __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&grid[(y + 1) * GRID_WIDTH + x + 1]));

            // Check which cells below-right are empty
            __m128i is_empty_right = _mm_cmpeq_epi8(right, empty_value);

            // Check which cells contain sand
            is_sand = _mm_cmpeq_epi8(current, sand_value);

            // Determine which sand particles can fall
            can_fall = _mm_and_si128(is_sand, is_empty_right);

            // Update current row (remove sand that will fall)
            current = _mm_andnot_si128(can_fall, current);

            // Update row below-right (add sand that has fallen)
            right = _mm_or_si128(right, _mm_and_si128(can_fall, sand_value));

            // Store updated rows back to grid
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&grid[(y + 1) * GRID_WIDTH + x + 1]), right);

            /////////////////////////////////////

            // At the end
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&grid[y * GRID_WIDTH + x]), current);


        }
    }
}


void updateSandAVX2() {
    const __m256i sand_value = _mm256_set1_epi8(1);  // Sand is represented by 1
    const __m256i empty_value = _mm256_setzero_si256();  // Empty is represented by 0

    for (int y = GRID_HEIGHT - 2; y >= 0; --y) {
        for (int x = 0; x < GRID_WIDTH; x += 32) {
            // Load current row
            __m256i current = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&grid[y * GRID_WIDTH + x]));

            // Check which cells contain sand
            __m256i is_sand = _mm256_cmpeq_epi8(current, sand_value);


            /////////////////////////////////////

            // Load current row
            __m256i below = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&grid[(y + 1) * GRID_WIDTH + x]));

            // Check which cells below are empty
            __m256i is_empty_below = _mm256_cmpeq_epi8(below, empty_value);

            // Determine which sand particles can fall
            __m256i can_fall = _mm256_and_si256(is_sand, is_empty_below);

            // Update current row (remove sand that will fall)
            current = _mm256_andnot_si256(can_fall, current);

            // Update row below (add sand that has fallen)
            below = _mm256_or_si256(below, _mm256_and_si256(can_fall, sand_value));

            // Store updated rows back to grid
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&grid[(y + 1) * GRID_WIDTH + x]), below);

            /////////////////////////////////////

            // Load the row below-left
            __m256i left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&grid[(y + 1) * GRID_WIDTH + x - 1]));

            // Check which cells below-left are empty
            __m256i is_empty_left = _mm256_cmpeq_epi8(left, empty_value);

            // Check which cells contain sand
            is_sand = _mm256_cmpeq_epi8(current, sand_value);

            // Determine which sand particles can fall
            can_fall = _mm256_and_si256(is_sand, is_empty_left);

            // Update current row (remove sand that will fall)
            current = _mm256_andnot_si256(can_fall, current);

            // Update row below-left (add sand that has fallen)
            left = _mm256_or_si256(left, _mm256_and_si256(can_fall, sand_value));

            // Store updated rows back to grid
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&grid[(y + 1) * GRID_WIDTH + x - 1]), left);

            /////////////////////////////////////

            // Load the row below-right
            __m256i right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&grid[(y + 1) * GRID_WIDTH + x + 1]));

            // Check which cells below-right are empty
            __m256i is_empty_right = _mm256_cmpeq_epi8(right, empty_value);

            // Check which cells contain sand
            is_sand = _mm256_cmpeq_epi8(current, sand_value);

            // Determine which sand particles can fall
            can_fall = _mm256_and_si256(is_sand, is_empty_right);

            // Update current row (remove sand that will fall)
            current = _mm256_andnot_si256(can_fall, current);

            // Update row below-right (add sand that has fallen)
            right = _mm256_or_si256(right, _mm256_and_si256(can_fall, sand_value));
            // Store updated rows back to grid

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&grid[(y + 1) * GRID_WIDTH + x + 1]), right);

            /////////////////////////////////////

            // At the end
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&grid[y * GRID_WIDTH + x]), current);
        }
    }
}

void addSand(int x, int y) {
    for (int dx = -9; dx <= 10; ++dx) {
        for (int dy = -9; dy <= 10; ++dy) {
            if (x + dx >= 0 && x + dx < SCREEN_WIDTH && y + dy >= 0 && y + dy < SCREEN_HEIGHT) {
                if (dis(gen) < 0.7) {
                    grid[(y + dy) * GRID_WIDTH + (x + dx)] = 1;
                }
            }
        }
    }
}

int countSand() {
    int count = 0;
    for (int y = 0; y < SCREEN_HEIGHT; ++y) {
        for (int x = 0; x < SCREEN_WIDTH; ++x) {
            if (grid[y * GRID_WIDTH + x] == 1) count++;
        }
    }
    return count;
}

int main(int argc, char* args[]) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Sand Simulation (SIMD)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    bool quit = false;
    SDL_Event e;
    bool mousePressed = false;

    // Initialize some sand
    for (int i = 0; i < 5000; ++i) {
        int x = static_cast<int>(dis(gen) * SCREEN_WIDTH);
        int y = static_cast<int>(dis(gen) * (SCREEN_HEIGHT / 2));
        addSand(x, y);
    }

    int frameCount = 0;
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_MOUSEBUTTONDOWN) {
                mousePressed = true;
            } else if (e.type == SDL_MOUSEBUTTONUP) {
                mousePressed = false;
            }
        }

        if (mousePressed) {
            int x, y;
            SDL_GetMouseState(&x, &y);
            addSand(x, y);
        }

        updateSandAVX2();

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 194, 178, 128, 255);
        for (int y = 0; y < SCREEN_HEIGHT; ++y) {
            for (int x = 0; x < SCREEN_WIDTH; ++x) {
                if (grid[y * GRID_WIDTH + x] == 1) {
                    SDL_RenderDrawPoint(renderer, x, y);
                }
            }
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(5);  // Aim for about 60 FPS

        // Debug output
        if (frameCount % 60 == 0) {  // Print every 60 frames
            std::cout << "Frame: " << frameCount << ", Sand count: " << countSand() << std::endl;
        }
        frameCount++;
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
