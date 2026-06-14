// Shared interactive HUD + canvas compositing for every viewer. The CPU/SDL and
// Vulkan/SDL viewers render the whole frame on the CPU and call both renderCanvas()
// (cells -> pixels, with flame flicker + emissive bloom) and drawHud() (the categorised
// palette, info bar, hover tooltip, brush cursor and pause banner). The OpenGL viewer
// renders the cells on the GPU and calls only drawHud(), into a transparent RGBA overlay
// that it blends on top. Because the drawing lives here, every viewer stays in sync.
//
// Include AFTER the Material enum and hud_meta.h (it uses the bare enum names + kNames
// etc). Cell lookups are passed as a callable cell(worldX, worldY) -> material id, so a
// GPU viewer can supply a single-cell readback while the SDL viewers pass world.viewCell.
#pragma once
#include "ui.h"
#include "hud_meta.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace hud {

struct View {
    int renderW, renderH, LWv, LHv, PIXEL, viewX, viewY;
    const uint32_t* kColors;
    int tick;
};

struct State {
    const ui::Palette* pal;
    const uint32_t* orderedColors;   // swatch colours in category order (size MATERIAL_COUNT)
    const int* slotOf;               // material id -> swatch slot
    int current, brushRadius;
    bool paused;
    int fps;
    int mouseLX, mouseLY;            // logical/render-space mouse position
    // Challenge mode (chalIdx < 0 = free sandbox, no banner).
    int chalIdx = -1, chalCount = 0;
    const char* chalName = nullptr;
    const char* chalGoal = nullptr;
    bool chalSolved = false;
    int chalSecs = 0;
};

// Precompute the radial bloom falloff kernel for a given radius into kern (size (2r+1)^2).
inline void buildGlowKernel(float* kern, int GR) {
    for (int dy = -GR; dy <= GR; ++dy)
        for (int dx = -GR; dx <= GR; ++dx) {
            float w = 1.0f - std::sqrt((float)(dx * dx + dy * dy)) / (GR + 1);
            kern[(dy + GR) * (2 * GR + 1) + (dx + GR)] = (w > 0.0f) ? w * w : 0.0f;
        }
}

// Cells -> pixels, with flame flicker and an additive emissive bloom (fire/lava/lasers
// glow). glowR/G/B are scratch buffers sized LWv*LHv. CellFn is cell(wx,wy)->id.
template <class CellFn>
inline void renderCanvas(uint32_t* px, const View& v, CellFn cell,
                         std::vector<float>& gR, std::vector<float>& gG, std::vector<float>& gB,
                         const float* kern, int GR, float GLOW) {
    const int LWv = v.LWv, LHv = v.LHv, PIXEL = v.PIXEL, W = v.renderW;
    std::fill(gR.begin(), gR.end(), 0.0f);
    std::fill(gG.begin(), gG.end(), 0.0f);
    std::fill(gB.begin(), gB.end(), 0.0f);
    for (int y = 0; y < LHv; ++y)
        for (int x = 0; x < LWv; ++x) {
            uint8_t m = cell(v.viewX + x, v.viewY + y);
            float s = emissionStrength(m);
            if (s <= 0.0f) continue;
            uint32_t c = v.kColors[m];
            float cr = (c >> 16) & 0xFF, cg = (c >> 8) & 0xFF, cb = c & 0xFF;
            for (int dy = -GR; dy <= GR; ++dy) {
                int ny = y + dy; if (ny < 0 || ny >= LHv) continue;
                for (int dx = -GR; dx <= GR; ++dx) {
                    int nx = x + dx; if (nx < 0 || nx >= LWv) continue;
                    float w = kern[(dy + GR) * (2 * GR + 1) + (dx + GR)] * s;
                    if (w <= 0.0f) continue;
                    size_t ni = (size_t)ny * LWv + nx;
                    gR[ni] += cr * w; gG[ni] += cg * w; gB[ni] += cb * w;
                }
            }
        }
    for (int y = 0; y < LHv; ++y)
        for (int x = 0; x < LWv; ++x) {
            int wxc = v.viewX + x, wyc = v.viewY + y;
            uint8_t m = cell(wxc, wyc);
            uint32_t color = v.kColors[m];
            if (m == FIRE || m == LAVA) color = ui::flicker(color, wxc, wyc, v.tick);
            size_t gi = (size_t)y * LWv + x;
            if (gR[gi] + gG[gi] + gB[gi] > 0.0f) {
                int rr = (int)((color >> 16) & 0xFF) + (int)(gR[gi] * GLOW);
                int gg = (int)((color >> 8)  & 0xFF) + (int)(gG[gi] * GLOW);
                int bb = (int)( color        & 0xFF) + (int)(gB[gi] * GLOW);
                if (rr > 255) rr = 255;
                if (gg > 255) gg = 255;
                if (bb > 255) bb = 255;
                color = 0xFF000000u | ((uint32_t)rr << 16) | ((uint32_t)gg << 8) | (uint32_t)bb;
            }
            for (int dy = 0; dy < PIXEL; ++dy)
                for (int dx = 0; dx < PIXEL; ++dx)
                    px[(size_t)(y * PIXEL + dy) * W + (x * PIXEL + dx)] = color;
        }
}

// The categorised palette + accents + hover tooltip + brush-outline cursor + bottom info
// bar + pause banner. Draws onto px (the frame, or a transparent overlay for the GPU
// viewer). cell(wx,wy)->id is used only to name the cell under the cursor.
template <class CellFn>
inline void drawHud(uint32_t* px, const View& v, const State& s, CellFn cell) {
    const ui::Palette& pal = *s.pal;
    const int W = v.renderW, H = v.renderH;
    ui::draw(px, W, H, pal, s.orderedColors, s.slotOf[s.current]);
    int stride = pal.sw + pal.gap;                  // thin per-category accent under each swatch
    for (int i = 0; i < pal.n; ++i) {
        int sx = pal.x0 + (i % pal.cols) * stride, sy = pal.y0 + (i / pal.cols) * stride;
        ui::fillRect(px, W, H, sx, sy + pal.sw + 1, pal.sw, 2, kCatAccent[kSlotCat[i]]);
    }

    int hlx = s.mouseLX, hly = s.mouseLY;
    int hov = ui::hit(pal, hlx, hly);
    bool onCanvas = (hlx >= 0 && hly >= 0 && hlx < W && hly < H);
    if (hov < 0 && onCanvas) {                       // circular brush-outline cursor
        int ccx = hlx / v.PIXEL, ccy = hly / v.PIXEL, r = s.brushRadius;
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx) {
                if (dx * dx + dy * dy > r * r) continue;
                bool edge = (dx-1)*(dx-1)+dy*dy > r*r || (dx+1)*(dx+1)+dy*dy > r*r
                          || dx*dx+(dy-1)*(dy-1) > r*r || dx*dx+(dy+1)*(dy+1) > r*r;
                if (!edge) continue;
                int cx = ccx + dx, cy = ccy + dy;
                if (cx >= 0 && cy >= 0 && cx < v.LWv && cy < v.LHv)
                    ui::fillRect(px, W, H, cx * v.PIXEL, cy * v.PIXEL, v.PIXEL, v.PIXEL, 0xFFFFFFFFu);
            }
    }
    { // tooltip "NAME  CATEGORY" for the swatch or cell under the cursor
        int tipMat = -1;
        if (hov >= 0) tipMat = kPaletteOrder[hov];
        else if (onCanvas) tipMat = cell(v.viewX + hlx / v.PIXEL, v.viewY + hly / v.PIXEL);
        if (tipMat >= 0 && kNames[tipMat] && *kNames[tipMat]) {
            char tip[64];
            std::snprintf(tip, sizeof tip, "%s  %s", kNames[tipMat], kCatNames[kSlotCat[s.slotOf[tipMat]]]);
            int tx = hlx + 16, ty = hly + 16, tw = ui::textWidth(tip, 2);
            if (tx + tw + 6 > W) tx = W - tw - 6;
            if (ty + 22 > H - 26) ty = hly - 24;
            if (tx < 4) tx = 4;
            if (ty < 4) ty = 4;
            ui::label(px, W, H, tx, ty, tip, 2, 0xFFFFFFFFu);
        }
    }
    { // bottom info bar: selected swatch + name + category + brush + fps
        int barH = 24, by = H - barH;
        ui::fillRect(px, W, H, 0, by, W, barH, 0xFF121218u);
        ui::fillRect(px, W, H, 0, by, W, 1, 0xFF3A3A44u);
        ui::fillRect(px, W, H, 6, by + 5, 14, 14, v.kColors[s.current] | 0xFF000000u);
        ui::outline(px, W, H, 6, by + 5, 14, 14, 1, 0xFF000000u);
        char buf[128];
        std::snprintf(buf, sizeof buf, "%s  %s   BRUSH %d   %d FPS",
                      kNames[s.current], kCatNames[kSlotCat[s.slotOf[s.current]]], s.brushRadius, s.fps);
        ui::text(px, W, H, 26, by + 6, buf, 2, 0xFFFFFFFFu);
        const char* help = "LMB PAINT  RMB ERASE  MMB PICK  WHEEL BRUSH  SPACE PAUSE  DEL CLEAR  ENTER CHALLENGE";
        int hw = ui::textWidth(help, 1);
        ui::text(px, W, H, W - hw - 6, by + 9, help, 1, 0xFFB0B0BEu);
    }
    if (s.paused) {
        const char* pw = "PAUSED  -  SPACE RESUME  TAB STEP";
        int sc = 3, w = ui::textWidth(pw, sc);
        ui::label(px, W, H, (W - w) / 2, H / 2 - 10, pw, sc, 0xFFFFE060u);
    }
    if (s.chalIdx >= 0) {                          // challenge banner above the info bar
        char cb[160];
        std::snprintf(cb, sizeof cb, "CHALLENGE %d/%d   %s:  %s",
                      s.chalIdx + 1, s.chalCount, s.chalName ? s.chalName : "", s.chalGoal ? s.chalGoal : "");
        ui::label(px, W, H, 6, H - 24 - 22, cb, 1, s.chalSolved ? 0xFF70FF90u : 0xFFFFE060u);
        if (s.chalSolved) {
            char sv[64];
            std::snprintf(sv, sizeof sv, "SOLVED IN %ds!   ENTER = NEXT", s.chalSecs);
            int sc = 3, w = ui::textWidth(sv, sc);
            ui::label(px, W, H, (W - w) / 2, H / 2 + 24, sv, sc, 0xFF70FF90u);
        }
    }
}

} // namespace hud
