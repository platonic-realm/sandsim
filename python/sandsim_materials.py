"""sandsim - Python multi-material engine (Noita-style).

A readable companion to cpp/sandsim_materials.cpp implementing the exact same
rule, so its --bench checksum matches the C++ reference. Materials:

    EMPTY  air / nothing
    WALL   solid, never moves
    SAND   powder: falls, then piles diagonally
    WATER  liquid: falls, then spreads horizontally
    GAS    gas: rises, then spreads horizontally

Movement is a swap, so every material is conserved (only EMPTY shuffles). A
mover may swap into a target only if the target is strictly lighter in the
relevant direction:

    SAND  -> EMPTY, WATER, GAS   (sinks through water and gas)
    WATER -> EMPTY, GAS          (sinks through gas)
    GAS   -> EMPTY               (rises; never re-enters water, so no oscillation)

Modes:
    (default)                 pygame window; number keys pick a material, the
                              left mouse button paints it.
    --bench [steps] [w] [h]   headless, pure-Python: deterministic scene, time
                              the update loop, print one RESULT line with a
                              checksum and conserved per-material counts.
"""

import sys
import time

EMPTY, WALL, SAND, WATER, GAS = 0, 1, 2, 3, 4
MATERIAL_COUNT = 5

# ARGB colors (used by the interactive renderer).
COLORS = [
    (0, 0, 0),        # EMPTY
    (128, 128, 128),  # WALL
    (226, 200, 120),  # SAND
    (68, 136, 255),   # WATER
    (176, 196, 222),  # GAS
]
NAMES = ["Eraser", "Wall", "Sand", "Water", "Gas"]

U32 = 0xFFFFFFFF
U64 = 0xFFFFFFFFFFFFFFFF


def can_enter(mover, target):
    """May `mover` swap into a cell holding `target`?"""
    if target == WALL:
        return False
    if mover == SAND:
        return target == EMPTY or target == WATER or target == GAS
    if mover == WATER:
        return target == EMPTY or target == GAS
    if mover == GAS:
        return target == EMPTY
    return False


def seed_material(x, y, w, h):
    """Deterministic per-cell material for --bench (matches the C++ port)."""
    if x == 0 or x == w - 1 or y == h - 1:
        return WALL
    if y == h // 2 and (x % 20 != 0):
        return WALL
    hsh = (x * 374761393 + y * 668265263) & U32
    hsh = ((hsh ^ (hsh >> 13)) * 1274126177) & U32
    r = hsh % 100
    if y < h // 3:
        return SAND if r < 40 else EMPTY
    elif y < 2 * h // 3:
        return WATER if r < 35 else EMPTY
    return GAS if r < 20 else EMPTY


def checksum(grid):
    c = 14695981039346656037
    for cell in grid:
        c = ((c ^ cell) * 1099511628211) & U64
    return c


class MaterialSim:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.grid = bytearray(w * h)        # all EMPTY
        self.moved = bytearray(w * h)

    def seed_bench_scene(self):
        w, h = self.w, self.h
        g = self.grid
        for y in range(h):
            base = y * w
            for x in range(w):
                g[base + x] = seed_material(x, y, w, h)

    def _try_move(self, x, y, nx, ny):
        w, h = self.w, self.h
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            return False
        ni = ny * w + nx
        if self.moved[ni]:
            return False
        g = self.grid
        i = y * w + x
        target = g[ni]
        if not can_enter(g[i], target):
            return False
        g[ni] = g[i]
        g[i] = target                 # swap conserves all materials
        self.moved[ni] = 1
        self.moved[i] = 1
        return True

    def update(self, frame):
        w, h = self.w, self.h
        g = self.grid
        self.moved = bytearray(w * h)
        moved = self.moved
        for y in range(h - 1, -1, -1):        # bottom-to-top
            base = y * w
            for x in range(w):
                if moved[base + x]:
                    continue
                m = g[base + x]
                if m == EMPTY or m == WALL:
                    continue
                left = ((x + y + frame) & 1) == 0
                d1 = -1 if left else 1
                d2 = -d1
                if m == SAND or m == WATER:
                    if self._try_move(x, y, x, y + 1):
                        continue
                    if self._try_move(x, y, x + d1, y + 1):
                        continue
                    if self._try_move(x, y, x + d2, y + 1):
                        continue
                    if m == WATER:
                        if self._try_move(x, y, x + d1, y):
                            continue
                        if self._try_move(x, y, x + d2, y):
                            continue
                else:  # GAS
                    if self._try_move(x, y, x, y - 1):
                        continue
                    if self._try_move(x, y, x + d1, y - 1):
                        continue
                    if self._try_move(x, y, x + d2, y - 1):
                        continue
                    if self._try_move(x, y, x + d1, y):
                        continue
                    if self._try_move(x, y, x + d2, y):
                        continue

    def paint(self, cx, cy, material, radius):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.w and 0 <= ny < self.h and dx * dx + dy * dy <= radius * radius:
                    self.grid[ny * self.w + nx] = material

    def clear(self):
        self.grid = bytearray(self.w * self.h)


def run_bench(steps, width, height):
    sim = MaterialSim(width, height)
    sim.seed_bench_scene()

    start = time.perf_counter()
    for s in range(steps):
        sim.update(s)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    counts = [0] * MATERIAL_COUNT
    for c in sim.grid:
        counts[c] += 1
    cells = width * height * steps
    mcells = cells / (elapsed_ms / 1000.0) / 1e6 if elapsed_ms > 0 else 0.0
    print(
        "RESULT impl=python_materials rule=materials "
        f"width={width} height={height} steps={steps} "
        f"elapsed_ms={elapsed_ms:.3f} mcells_per_s={mcells:.2f} "
        f"checksum={checksum(sim.grid):016x} "
        f"empty={counts[EMPTY]} wall={counts[WALL]} sand={counts[SAND]} "
        f"water={counts[WATER]} gas={counts[GAS]}"
    )


def run_interactive(width, height):
    import pygame  # imported lazily so --bench needs no pygame

    PIXEL = 3
    pygame.init()
    screen = pygame.display.set_mode((width * PIXEL, height * PIXEL))
    pygame.display.set_caption("Materials Sand Simulation")
    clock = pygame.time.Clock()
    surface = pygame.Surface((width, height))

    sim = MaterialSim(width, height)
    current = SAND
    keymap = {pygame.K_0: EMPTY, pygame.K_1: WALL, pygame.K_2: SAND,
              pygame.K_3: WATER, pygame.K_4: GAS}

    running = True
    mouse_down = False
    frame = 0
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
                    pygame.display.set_caption(f"Materials Sand Simulation - current: {NAMES[current]}")
                elif event.key == pygame.K_c:
                    sim.clear()

        if mouse_down:
            mx, my = pygame.mouse.get_pos()
            # Map the cursor from the actual window size (a tiling compositor
            # such as niri may resize it) to the logical surface.
            ww, wh = pygame.display.get_window_size()
            lw, lh = width * PIXEL, height * PIXEL
            lx = mx * lw // ww if ww else mx
            ly = my * lh // wh if wh else my
            sim.paint(lx // PIXEL, ly // PIXEL, current, 4)

        sim.update(frame)
        frame += 1

        # Draw at grid resolution, then scale up to the window.
        for y in range(height):
            base = y * width
            for x in range(width):
                surface.set_at((x, y), COLORS[sim.grid[base + x]])
        pygame.transform.scale(surface, (width * PIXEL, height * PIXEL), screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        width = int(sys.argv[3]) if len(sys.argv) > 3 else 400
        height = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        run_bench(steps, width, height)
    else:
        run_interactive(200, 150)


if __name__ == "__main__":
    main()
