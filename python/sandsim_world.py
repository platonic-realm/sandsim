"""sandsim - Python chunked streaming world (Noita-style).

Port of cpp/sandsim_world.cpp: a world larger than memory, with only a few
"live boxes" (chunks) resident around a camera and the rest streamed to disk.
See WORLD.md. Resident chunks live in a dense list and are processed
bottom-chunk-first, reproducing the C++ reference checksum.

Modes:
    (default)                 pygame window; WASD/arrows pan, number keys paint.
    --bench [steps] [wch] [hch]   headless deterministic streaming benchmark
                              (pure Python, no pygame needed).
"""

import os
import sys
import time

EMPTY, WALL, SAND, WATER, GAS = 0, 1, 2, 3, 4
MATERIAL_COUNT = 5
CHUNK = 64
CHUNK_MASK = 63
CHUNK_SHIFT = 6
U32 = 0xFFFFFFFF
U64 = 0xFFFFFFFFFFFFFFFF

COLORS = [(0, 0, 0), (128, 128, 128), (226, 200, 120), (68, 136, 255), (176, 196, 222)]
NAMES = ["Eraser", "Wall", "Sand", "Water", "Gas"]


def can_enter(mover, target):
    if target == WALL:
        return False
    if mover == SAND:
        return target == EMPTY or target == WATER or target == GAS
    if mover == WATER:
        return target == EMPTY or target == GAS
    if mover == GAS:
        return target == EMPTY
    return False


def hash_coord(gx, gy):
    h = (gx * 374761393 + gy * 668265263) & U32
    return ((h ^ (h >> 13)) * 1274126177) & U32


def gen_cell(gx, gy, wcells, hcells):
    if gx == 0 or gy == 0 or gx == wcells - 1 or gy == hcells - 1:
        return WALL
    if gy % 40 == 39 and (gx % 11 != 0):
        return WALL
    r = hash_coord(gx, gy) % 100
    band = (gy // 40) % 3
    if band == 0:
        return SAND if r < 35 else EMPTY
    if band == 1:
        return WATER if r < 30 else EMPTY
    return GAS if r < 18 else EMPTY


class Chunk:
    __slots__ = ("cells", "moved", "dminx", "dminy", "dmaxx", "dmaxy",
                 "nminx", "nminy", "nmaxx", "nmaxy")

    def __init__(self):
        self.cells = bytearray(CHUNK * CHUNK)
        self.moved = bytearray(CHUNK * CHUNK)
        self.full_dirty()
        self.clear_next()

    def full_dirty(self):
        self.dminx, self.dminy, self.dmaxx, self.dmaxy = 0, 0, CHUNK - 1, CHUNK - 1

    def clear_next(self):
        self.nminx, self.nminy, self.nmaxx, self.nmaxy = CHUNK, CHUNK, -1, -1

    def awake(self):
        return self.dminx <= self.dmaxx and self.dminy <= self.dmaxy

    def commit_next(self):
        self.dminx, self.dminy, self.dmaxx, self.dmaxy = self.nminx, self.nminy, self.nmaxx, self.nmaxy
        self.clear_next()


class World:
    def __init__(self, wch, hch, directory):
        self.wch, self.hch = wch, hch
        self.wcells, self.hcells = wch * CHUNK, hch * CHUNK
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        self.chunks = [None] * (wch * hch)
        self.frame = 0
        self.resident_max = 0
        self.n_writes = self.n_reads = self.n_generated = 0

    def resident_at(self, cx, cy):
        if cx < 0 or cy < 0 or cx >= self.wch or cy >= self.hch:
            return None
        return self.chunks[cy * self.wch + cx]

    def _path(self, cx, cy):
        return os.path.join(self.dir, f"c_{cx}_{cy}.bin")

    def _write_chunk(self, cx, cy, ch):
        with open(self._path(cx, cy), "wb") as f:
            f.write(ch.cells)
        self.n_writes += 1

    def _read_chunk_raw(self, cx, cy):
        try:
            with open(self._path(cx, cy), "rb") as f:
                return bytearray(f.read())
        except FileNotFoundError:
            return None

    def get(self, gx, gy):
        if gx < 0 or gy < 0 or gx >= self.wcells or gy >= self.hcells:
            return WALL
        c = self.resident_at(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT)
        if c is None:
            return WALL
        return c.cells[(gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK)]

    def generate_all_to_disk(self):
        for cy in range(self.hch):
            for cx in range(self.wch):
                ch = Chunk()
                for ly in range(CHUNK):
                    base = ly * CHUNK
                    gy = cy * CHUNK + ly
                    for lx in range(CHUNK):
                        ch.cells[base + lx] = gen_cell(cx * CHUNK + lx, gy, self.wcells, self.hcells)
                self._write_chunk(cx, cy, ch)

    def _load_or_generate(self, cx, cy):
        ch = Chunk()
        raw = self._read_chunk_raw(cx, cy)
        if raw is not None:
            ch.cells = raw
            self.n_reads += 1
        else:
            for ly in range(CHUNK):
                base = ly * CHUNK
                gy = cy * CHUNK + ly
                for lx in range(CHUNK):
                    ch.cells[base + lx] = gen_cell(cx * CHUNK + lx, gy, self.wcells, self.hcells)
            self.n_generated += 1
        ch.full_dirty()
        ch.clear_next()
        self.chunks[cy * self.wch + cx] = ch

    def stream_around(self, cam_cx, cam_cy, radius):
        for cy in range(self.hch):
            for cx in range(self.wch):
                c = self.chunks[cy * self.wch + cx]
                if c is None:
                    continue
                if abs(cx - cam_cx) > radius or abs(cy - cam_cy) > radius:
                    self._write_chunk(cx, cy, c)
                    self.chunks[cy * self.wch + cx] = None
        for cy in range(cam_cy - radius, cam_cy + radius + 1):
            for cx in range(cam_cx - radius, cam_cx + radius + 1):
                if 0 <= cx < self.wch and 0 <= cy < self.hch and self.resident_at(cx, cy) is None:
                    self._load_or_generate(cx, cy)
        res = sum(1 for c in self.chunks if c is not None)
        if res > self.resident_max:
            self.resident_max = res

    def _wake(self, gx, gy):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = gx + dx, gy + dy
                c = self.resident_at(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT)
                if c is None:
                    continue
                lx, ly = nx & CHUNK_MASK, ny & CHUNK_MASK
                if lx < c.nminx: c.nminx = lx
                if ly < c.nminy: c.nminy = ly
                if lx > c.nmaxx: c.nmaxx = lx
                if ly > c.nmaxy: c.nmaxy = ly

    def _try_move(self, gx, gy, nx, ny):
        if nx < 0 or ny < 0 or nx >= self.wcells or ny >= self.hcells:
            return False
        tc = self.resident_at(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT)
        if tc is None:
            return False
        ti = (ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK)
        if tc.moved[ti]:
            return False
        target = tc.cells[ti]
        sc = self.resident_at(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT)
        si = (gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK)
        self_mat = sc.cells[si]
        if not can_enter(self_mat, target):
            return False
        tc.cells[ti] = self_mat
        sc.cells[si] = target
        tc.moved[ti] = 1
        sc.moved[si] = 1
        self._wake(gx, gy)
        self._wake(nx, ny)
        return True

    def step(self):
        for c in self.chunks:
            if c is not None:
                c.moved = bytearray(CHUNK * CHUNK)
        for cy in range(self.hch - 1, -1, -1):       # bottom chunks first
            for cx in range(self.wch):
                c = self.chunks[cy * self.wch + cx]
                if c is None or not c.awake():
                    continue
                baseX, baseY = cx * CHUNK, cy * CHUNK
                for ly in range(c.dmaxy, c.dminy - 1, -1):
                    row = ly * CHUNK
                    gy = baseY + ly
                    for lx in range(c.dminx, c.dmaxx + 1):
                        if c.moved[row + lx]:
                            continue
                        m = c.cells[row + lx]
                        if m == EMPTY or m == WALL:
                            continue
                        gx = baseX + lx
                        left = ((gx + gy + self.frame) & 1) == 0
                        d1 = -1 if left else 1
                        d2 = -d1
                        if m == SAND or m == WATER:
                            if self._try_move(gx, gy, gx, gy + 1): continue
                            if self._try_move(gx, gy, gx + d1, gy + 1): continue
                            if self._try_move(gx, gy, gx + d2, gy + 1): continue
                            if m == WATER:
                                if self._try_move(gx, gy, gx + d1, gy): continue
                                if self._try_move(gx, gy, gx + d2, gy): continue
                        else:  # GAS
                            if self._try_move(gx, gy, gx, gy - 1): continue
                            if self._try_move(gx, gy, gx + d1, gy - 1): continue
                            if self._try_move(gx, gy, gx + d2, gy - 1): continue
                            if self._try_move(gx, gy, gx + d1, gy): continue
                            if self._try_move(gx, gy, gx + d2, gy): continue
        for c in self.chunks:
            if c is not None:
                c.commit_next()
        self.frame += 1

    def paint(self, gx, gy, material, radius):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if dx * dx + dy * dy > radius * radius:
                    continue
                c = self.resident_at(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT)
                if c is None:
                    continue
                c.cells[(ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK)] = material
                self._wake(nx, ny)

    def summary(self):
        counts = [0] * MATERIAL_COUNT
        c = 14695981039346656037
        for cy in range(self.hch):
            for cx in range(self.wch):
                res = self.resident_at(cx, cy)
                cells = res.cells if res is not None else self._read_chunk_raw(cx, cy)
                for v in cells:
                    counts[v] += 1
                    c = ((c ^ v) * 1099511628211) & U64
        return c, counts


def run_bench(steps, wch, hch):
    directory = f"/tmp/sandsim_world_py_{steps}_{wch}x{hch}"
    world = World(wch, hch, directory)
    world.generate_all_to_disk()
    _, start_cnt = world.summary()

    cells = wch * hch
    start = time.perf_counter()
    for s in range(steps):
        visit = min(cells - 1, (s * cells) // steps)
        row, col = visit // wch, visit % wch
        cam_cx = col if row % 2 == 0 else (wch - 1 - col)
        cam_cy = row
        world.stream_around(cam_cx, cam_cy, 1)
        world.step()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    ck, cnt = world.summary()
    conserved = all(cnt[i] == start_cnt[i] for i in range(WALL, GAS + 1))
    print(
        "RESULT impl=python_world rule=world "
        f"wchunks={wch} hchunks={hch} chunk={CHUNK} steps={steps} "
        f"elapsed_ms={elapsed_ms:.3f} checksum={ck:016x} "
        f"empty={cnt[EMPTY]} wall={cnt[WALL]} sand={cnt[SAND]} water={cnt[WATER]} gas={cnt[GAS]} "
        f"resident_max={world.resident_max} disk_writes={world.n_writes} "
        f"disk_reads={world.n_reads} conserved={'yes' if conserved else 'no'}"
    )
    return 0 if conserved else 2


def run_interactive():
    import pygame  # lazy: --bench needs no pygame

    PIXEL = 2
    VIEW_W, VIEW_H, WCH, HCH = 320, 240, 64, 64
    pygame.init()
    screen = pygame.display.set_mode((VIEW_W * PIXEL, VIEW_H * PIXEL))
    pygame.display.set_caption("Streamed World (Python)")
    clock = pygame.time.Clock()
    surface = pygame.Surface((VIEW_W, VIEW_H))

    world = World(WCH, HCH, "/tmp/sandsim_world_py_interactive")
    camX = WCH * CHUNK // 2 - VIEW_W // 2
    camY = HCH * CHUNK // 2 - VIEW_H // 2
    current = SAND
    keymap = {pygame.K_0: EMPTY, pygame.K_1: WALL, pygame.K_2: SAND, pygame.K_3: WATER, pygame.K_4: GAS}

    running, mouse_down = True, False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
            elif event.type == pygame.KEYDOWN:
                if event.key in keymap:
                    current = keymap[event.key]
                    pygame.display.set_caption(f"Streamed World - painting {NAMES[current]}")
                elif event.key in (pygame.K_w, pygame.K_UP): camY -= 16
                elif event.key in (pygame.K_s, pygame.K_DOWN): camY += 16
                elif event.key in (pygame.K_a, pygame.K_LEFT): camX -= 16
                elif event.key in (pygame.K_d, pygame.K_RIGHT): camX += 16

        world.stream_around((camX + VIEW_W // 2) >> CHUNK_SHIFT, (camY + VIEW_H // 2) >> CHUNK_SHIFT, 3)
        if mouse_down:
            mx, my = pygame.mouse.get_pos()
            world.paint(camX + mx // PIXEL, camY + my // PIXEL, current, 4)
        world.step()

        for vy in range(VIEW_H):
            for vx in range(VIEW_W):
                surface.set_at((vx, vy), COLORS[world.get(camX + vx, camY + vy)])
        pygame.transform.scale(surface, (VIEW_W * PIXEL, VIEW_H * PIXEL), screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        wch = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        hch = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        sys.exit(run_bench(steps, wch, hch))
    run_interactive()


if __name__ == "__main__":
    main()
