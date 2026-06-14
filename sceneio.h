// Save / load a viewport of cells to a small binary file, so you can keep a creation (or a
// challenge solution) and reload it later. Format: 4-byte magic "SSV1", int32 W, int32 H,
// then W*H raw material bytes. Backend-agnostic -- the host reads the live viewport into a
// flat buffer to save, and stamps a loaded buffer back via world.loadView.
#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>

namespace sio {

inline bool save(const char* path, const uint8_t* cells, int W, int H) {
    if (W <= 0 || H <= 0) return false;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const char magic[4] = { 'S', 'S', 'V', '1' };
    bool ok = std::fwrite(magic, 1, 4, f) == 4
           && std::fwrite(&W, sizeof W, 1, f) == 1
           && std::fwrite(&H, sizeof H, 1, f) == 1
           && std::fwrite(cells, 1, (size_t)W * H, f) == (size_t)W * H;
    std::fclose(f);
    return ok;
}

inline bool load(const char* path, std::vector<uint8_t>& cells, int& W, int& H) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[4]; int w = 0, h = 0; bool ok = false;
    if (std::fread(magic, 1, 4, f) == 4
        && magic[0] == 'S' && magic[1] == 'S' && magic[2] == 'V' && magic[3] == '1'
        && std::fread(&w, sizeof w, 1, f) == 1 && std::fread(&h, sizeof h, 1, f) == 1
        && w > 0 && h > 0 && w < 100000 && h < 100000) {
        cells.assign((size_t)w * h, 0);
        ok = std::fread(cells.data(), 1, (size_t)w * h, f) == (size_t)w * h;
        W = w; H = h;
    }
    std::fclose(f);
    return ok;
}

} // namespace sio
