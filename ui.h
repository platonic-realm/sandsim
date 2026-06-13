// Shared on-screen HUD: a row of clickable material swatches. Pure geometry +
// drawing into an ARGB8888 pixel buffer, with no backend dependencies, so the
// CPU/SDL and Vulkan/SDL viewers share it; the OpenGL viewer reproduces the same
// layout in its present shader. Click a swatch to pick that material.
#pragma once
#include <cstdint>
#include <cstddef>

namespace ui {

// Palette geometry in render-pixel coordinates (top-left origin).
struct Palette { int x0, y0, sw, gap, n; };

inline Palette palette(int renderW, int n) {
    int sw = renderW / 22;
    if (sw < 22) sw = 22;
    if (sw > 52) sw = 52;
    int m = sw / 2;                       // outer margin
    return Palette{ m, m, sw, sw / 6 + 2, n };
}

// Swatch index under (mx,my) in render-pixel coords, or -1 if not on the palette.
inline int hit(const Palette& p, int mx, int my) {
    if (my < p.y0 || my >= p.y0 + p.sw) return -1;
    for (int i = 0; i < p.n; ++i) {
        int x = p.x0 + i * (p.sw + p.gap);
        if (mx >= x && mx < x + p.sw) return i;
    }
    return -1;
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
    int panelW = p.n * (p.sw + p.gap) + p.gap;
    fillRect(px, W, H, p.x0 - p.gap, p.y0 - p.gap, panelW, p.sw + 2 * p.gap, 0xFF1A1A1Au);
    for (int i = 0; i < p.n; ++i) {
        int x = p.x0 + i * (p.sw + p.gap);
        fillRect(px, W, H, x, p.y0, p.sw, p.sw, swatchColors[i] | 0xFF000000u);
        outline(px, W, H, x, p.y0, p.sw, p.sw, 1, 0xFF101010u);
        if (i == selected) outline(px, W, H, x - 2, p.y0 - 2, p.sw + 4, p.sw + 4, 2, 0xFFFFFFFFu);
    }
}

}  // namespace ui
