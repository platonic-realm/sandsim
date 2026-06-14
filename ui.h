// Shared on-screen HUD: a row of clickable material swatches. Pure geometry +
// drawing into an ARGB8888 pixel buffer, with no backend dependencies, so the
// CPU/SDL and Vulkan/SDL viewers share it; the OpenGL viewer reproduces the same
// layout in its present shader. Click a swatch to pick that material.
#pragma once
#include <cstdint>
#include <cstddef>

namespace ui {

// Animated brightness flicker for "hot" materials (fire/lava), render-only -- it
// modulates the drawn colour by ~±15% per cell per tick so flame and molten rock
// shimmer instead of sitting flat. The OpenGL present shader uses the same hash.
inline uint32_t flicker(uint32_t argb, int lx, int ly, int tick) {
    uint32_t h = (uint32_t)lx * 374761393u + (uint32_t)ly * 668265263u + (uint32_t)tick * 2654435761u;
    h = (h ^ (h >> 13)) * 1274126177u;
    float f = 0.80f + (float)(h & 0xFFu) / 255.0f * 0.32f;   // 0.80 .. 1.12
    int r = (int)(((argb >> 16) & 0xFF) * f); if (r > 255) r = 255;
    int g = (int)(((argb >> 8)  & 0xFF) * f); if (g > 255) g = 255;
    int b = (int)((argb         & 0xFF) * f); if (b > 255) b = 255;
    return 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

// Palette geometry in render-pixel coordinates (top-left origin). The swatches wrap
// into a grid of `cols` columns so the whole palette always fits the window, however
// many materials there are.
struct Palette { int x0, y0, sw, gap, n, cols; };

inline Palette palette(int renderW, int n) {
    int sw = renderW / 15;                // aim for a comfortably clickable swatch
    if (sw < 20) sw = 20;
    if (sw > 44) sw = 44;
    int gap = sw / 6 + 2;
    int m = sw / 2;                       // outer margin
    int avail = renderW - 2 * m;          // width for the swatch grid
    int cols = (avail + gap) / (sw + gap);
    if (cols < 1) cols = 1;
    if (cols > n) cols = n;
    return Palette{ m, m, sw, gap, n, cols };
}

// Swatch index under (mx,my) in render-pixel coords, or -1 if not on the palette.
inline int hit(const Palette& p, int mx, int my) {
    int relx = mx - p.x0, rely = my - p.y0;
    if (relx < 0 || rely < 0) return -1;
    int stride = p.sw + p.gap;
    int col = relx / stride, row = rely / stride;
    if (col >= p.cols) return -1;                 // in the right-hand margin
    if (relx - col * stride >= p.sw) return -1;   // in a horizontal gap
    if (rely - row * stride >= p.sw) return -1;   // in a vertical gap
    int idx = row * p.cols + col;
    return (idx >= 0 && idx < p.n) ? idx : -1;
}

inline void fillRect(uint32_t* px, int W, int H, int x, int y, int w, int h, uint32_t c) {
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > W) w = W - x;
    if (y + h > H) h = H - y;
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            px[(size_t)(y + j) * W + (x + i)] = c;
}
inline void outline(uint32_t* px, int W, int H, int x, int y, int w, int h, int t, uint32_t c) {
    fillRect(px, W, H, x, y, w, t, c);
    fillRect(px, W, H, x, y + h - t, w, t, c);
    fillRect(px, W, H, x, y, t, h, c);
    fillRect(px, W, H, x + w - t, y, t, h, c);
}

// Draw the palette: swatchColors[i] is the ARGB fill for swatch i, `selected`
// gets a white highlight ring.
inline void draw(uint32_t* px, int W, int H, const Palette& p,
                 const uint32_t* swatchColors, int selected) {
    int stride = p.sw + p.gap;
    int rows = (p.n + p.cols - 1) / p.cols;
    int panelW = p.cols * stride + p.gap;
    int panelH = rows * stride + p.gap;
    fillRect(px, W, H, p.x0 - p.gap, p.y0 - p.gap, panelW, panelH, 0xFF1A1A1Au);
    for (int i = 0; i < p.n; ++i) {
        int x = p.x0 + (i % p.cols) * stride;
        int y = p.y0 + (i / p.cols) * stride;
        fillRect(px, W, H, x, y, p.sw, p.sw, swatchColors[i] | 0xFF000000u);
        outline(px, W, H, x, y, p.sw, p.sw, 1, 0xFF101010u);
        if (i == selected) outline(px, W, H, x - 2, y - 2, p.sw + 4, p.sw + 4, 2, 0xFFFFFFFFu);
    }
}

// ---- tiny embedded 5x7 bitmap font (no deps) so every viewer can draw labels ----
// Each glyph is 7 rows; the low 5 bits of each row are the pixels, bit 4 = leftmost.
inline const uint8_t* glyph5x7(char c) {
    if (c >= 'a' && c <= 'z') c -= 32;                 // fold to uppercase
    static const uint8_t SP[7] = {0,0,0,0,0,0,0};
    switch (c) {
        case 'A': { static const uint8_t g[7]={0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001}; return g; }
        case 'B': { static const uint8_t g[7]={0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110}; return g; }
        case 'C': { static const uint8_t g[7]={0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110}; return g; }
        case 'D': { static const uint8_t g[7]={0b11110,0b10001,0b10001,0b10001,0b10001,0b10001,0b11110}; return g; }
        case 'E': { static const uint8_t g[7]={0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111}; return g; }
        case 'F': { static const uint8_t g[7]={0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000}; return g; }
        case 'G': { static const uint8_t g[7]={0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01111}; return g; }
        case 'H': { static const uint8_t g[7]={0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001}; return g; }
        case 'I': { static const uint8_t g[7]={0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110}; return g; }
        case 'J': { static const uint8_t g[7]={0b00111,0b00010,0b00010,0b00010,0b00010,0b10010,0b01100}; return g; }
        case 'K': { static const uint8_t g[7]={0b10001,0b10010,0b10100,0b11000,0b10100,0b10010,0b10001}; return g; }
        case 'L': { static const uint8_t g[7]={0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111}; return g; }
        case 'M': { static const uint8_t g[7]={0b10001,0b11011,0b10101,0b10101,0b10001,0b10001,0b10001}; return g; }
        case 'N': { static const uint8_t g[7]={0b10001,0b10001,0b11001,0b10101,0b10011,0b10001,0b10001}; return g; }
        case 'O': { static const uint8_t g[7]={0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}; return g; }
        case 'P': { static const uint8_t g[7]={0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000}; return g; }
        case 'Q': { static const uint8_t g[7]={0b01110,0b10001,0b10001,0b10001,0b10101,0b10010,0b01101}; return g; }
        case 'R': { static const uint8_t g[7]={0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001}; return g; }
        case 'S': { static const uint8_t g[7]={0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110}; return g; }
        case 'T': { static const uint8_t g[7]={0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100}; return g; }
        case 'U': { static const uint8_t g[7]={0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}; return g; }
        case 'V': { static const uint8_t g[7]={0b10001,0b10001,0b10001,0b10001,0b10001,0b01010,0b00100}; return g; }
        case 'W': { static const uint8_t g[7]={0b10001,0b10001,0b10001,0b10101,0b10101,0b11011,0b10001}; return g; }
        case 'X': { static const uint8_t g[7]={0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001}; return g; }
        case 'Y': { static const uint8_t g[7]={0b10001,0b10001,0b01010,0b00100,0b00100,0b00100,0b00100}; return g; }
        case 'Z': { static const uint8_t g[7]={0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111}; return g; }
        case '0': { static const uint8_t g[7]={0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110}; return g; }
        case '1': { static const uint8_t g[7]={0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110}; return g; }
        case '2': { static const uint8_t g[7]={0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111}; return g; }
        case '3': { static const uint8_t g[7]={0b11111,0b00010,0b00100,0b00010,0b00001,0b10001,0b01110}; return g; }
        case '4': { static const uint8_t g[7]={0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010}; return g; }
        case '5': { static const uint8_t g[7]={0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110}; return g; }
        case '6': { static const uint8_t g[7]={0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110}; return g; }
        case '7': { static const uint8_t g[7]={0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000}; return g; }
        case '8': { static const uint8_t g[7]={0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110}; return g; }
        case '9': { static const uint8_t g[7]={0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100}; return g; }
        case ':': { static const uint8_t g[7]={0b00000,0b01100,0b01100,0b00000,0b01100,0b01100,0b00000}; return g; }
        case '.': { static const uint8_t g[7]={0b00000,0b00000,0b00000,0b00000,0b00000,0b01100,0b01100}; return g; }
        case ',': { static const uint8_t g[7]={0b00000,0b00000,0b00000,0b00000,0b01100,0b00100,0b01000}; return g; }
        case '-': { static const uint8_t g[7]={0b00000,0b00000,0b00000,0b11111,0b00000,0b00000,0b00000}; return g; }
        case '+': { static const uint8_t g[7]={0b00000,0b00100,0b00100,0b11111,0b00100,0b00100,0b00000}; return g; }
        case '/': { static const uint8_t g[7]={0b00001,0b00010,0b00010,0b00100,0b01000,0b01000,0b10000}; return g; }
        case '[': { static const uint8_t g[7]={0b01110,0b01000,0b01000,0b01000,0b01000,0b01000,0b01110}; return g; }
        case ']': { static const uint8_t g[7]={0b01110,0b00010,0b00010,0b00010,0b00010,0b00010,0b01110}; return g; }
        case '(': { static const uint8_t g[7]={0b00010,0b00100,0b01000,0b01000,0b01000,0b00100,0b00010}; return g; }
        case ')': { static const uint8_t g[7]={0b01000,0b00100,0b00010,0b00010,0b00010,0b00100,0b01000}; return g; }
        case '!': { static const uint8_t g[7]={0b00100,0b00100,0b00100,0b00100,0b00100,0b00000,0b00100}; return g; }
        case '%': { static const uint8_t g[7]={0b11001,0b11010,0b00100,0b01011,0b10011,0b00000,0b00000}; return g; }
        case '*': { static const uint8_t g[7]={0b00000,0b01010,0b00100,0b11111,0b00100,0b01010,0b00000}; return g; }
        default: return SP;
    }
}

// Width in render pixels of a string drawn with text() at the given scale.
inline int textWidth(const char* s, int scale) {
    int n = 0; for (const char* p = s; *p; ++p) ++n;
    return n > 0 ? (n * 6 - 1) * scale : 0;          // 5px glyph + 1px gap, minus trailing gap
}

// Draw a string of the 5x7 font at (x,y) in render pixels, `scale`x magnified.
inline void text(uint32_t* px, int W, int H, int x, int y, const char* s, int scale, uint32_t color) {
    int cx = x;
    for (; *s; ++s) {
        const uint8_t* g = glyph5x7(*s);
        for (int row = 0; row < 7; ++row)
            for (int col = 0; col < 5; ++col)
                if (g[row] & (1u << (4 - col)))
                    fillRect(px, W, H, cx + col * scale, y + row * scale, scale, scale, color);
        cx += 6 * scale;                              // advance one glyph + gap
    }
}

// Draw text with a soft dark backing box for legibility over any background.
inline void label(uint32_t* px, int W, int H, int x, int y, const char* s, int scale, uint32_t color) {
    int w = textWidth(s, scale), h = 7 * scale, pad = 2 * scale;
    fillRect(px, W, H, x - pad, y - pad, w + 2 * pad, h + 2 * pad, 0xE0101014u);
    outline(px, W, H, x - pad, y - pad, w + 2 * pad, h + 2 * pad, 1, 0xFF3A3A44u);
    text(px, W, H, x, y, s, scale, color);
}

}  // namespace ui
