// sandsim - Zig multi-material engine (Noita-style)
//
// Port of cpp/sandsim_materials.cpp. Materials: EMPTY, WALL, SAND (powder),
// WATER (liquid), GAS. Movement is a swap (every material is conserved); a
// mover may swap into a target only if the target is strictly lighter in the
// relevant direction (SAND sinks through WATER and GAS; WATER sinks through
// GAS; GAS rises into EMPTY). A per-frame "moved" flag keeps the in-place
// update deterministic for both up- and down-movers.
//
// Modes:
//   (default)                 SDL2 window; number keys pick a material, the
//                             left mouse button paints it.
//   --bench [steps] [w] [h]   headless: deterministic scene, print a RESULT
//                             line whose checksum matches the other ports.

const std = @import("std");

const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("time.h");
    @cInclude("SDL2/SDL.h");
});

const EMPTY: u8 = 0;
const WALL: u8 = 1;
const SAND: u8 = 2;
const WATER: u8 = 3;
const GAS: u8 = 4;
const PIXEL_SIZE: usize = 2;

const COLORS = [_]u32{ 0xFF000000, 0xFF808080, 0xFFE2C878, 0xFF4488FF, 0xFFB0C4DE };

const SDL_WINDOWPOS_UNDEFINED: c_int = 0x1FFF0000;
const SDL_PIXELFORMAT_ARGB8888: u32 = 0x16362004;

// --- shared rule (identical across all material ports) ---------------------

fn canEnter(mover: u8, target: u8) bool {
    if (target == WALL) return false;
    return switch (mover) {
        SAND => target == EMPTY or target == WATER or target == GAS,
        WATER => target == EMPTY or target == GAS,
        GAS => target == EMPTY,
        else => false,
    };
}

fn seedMaterial(x: usize, y: usize, w: usize, h: usize) u8 {
    if (x == 0 or x == w - 1 or y == h - 1) return WALL;
    if (y == h / 2 and (x % 20 != 0)) return WALL;
    var hsh: u32 = @as(u32, @intCast(x)) *% 374761393 +% @as(u32, @intCast(y)) *% 668265263;
    hsh = (hsh ^ (hsh >> 13)) *% 1274126177;
    const r = hsh % 100;
    if (y < h / 3) return if (r < 40) SAND else EMPTY;
    if (y < 2 * h / 3) return if (r < 35) WATER else EMPTY;
    return if (r < 20) GAS else EMPTY;
}

fn checksum(grid: []const u8) u64 {
    var acc: u64 = 14695981039346656037;
    for (grid) |cell| acc = (acc ^ @as(u64, cell)) *% 1099511628211;
    return acc;
}

const Sim = struct {
    w: usize,
    h: usize,
    grid: []u8,
    moved: []u8,

    fn tryMove(self: *Sim, x: usize, y: usize, nx: i64, ny: i64) bool {
        if (nx < 0 or nx >= @as(i64, @intCast(self.w)) or ny < 0 or ny >= @as(i64, @intCast(self.h)))
            return false;
        const unx: usize = @intCast(nx);
        const uny: usize = @intCast(ny);
        const ni = uny * self.w + unx;
        if (self.moved[ni] != 0) return false;
        const i = y * self.w + x;
        const target = self.grid[ni];
        if (!canEnter(self.grid[i], target)) return false;
        self.grid[ni] = self.grid[i];
        self.grid[i] = target; // swap conserves all materials
        self.moved[ni] = 1;
        self.moved[i] = 1;
        return true;
    }

    fn update(self: *Sim, frame: u32) void {
        @memset(self.moved, 0);
        var y: usize = self.h;
        while (y > 0) {
            y -= 1;
            var x: usize = 0;
            while (x < self.w) : (x += 1) {
                if (self.moved[y * self.w + x] != 0) continue;
                const m = self.grid[y * self.w + x];
                if (m == EMPTY or m == WALL) continue;
                const left = ((x + y + @as(usize, frame)) & 1) == 0;
                const d1: i64 = if (left) -1 else 1;
                const d2: i64 = -d1;
                const xi: i64 = @intCast(x);
                const yi: i64 = @intCast(y);
                if (m == SAND or m == WATER) {
                    if (self.tryMove(x, y, xi, yi + 1)) continue;
                    if (self.tryMove(x, y, xi + d1, yi + 1)) continue;
                    if (self.tryMove(x, y, xi + d2, yi + 1)) continue;
                    if (m == WATER) {
                        if (self.tryMove(x, y, xi + d1, yi)) continue;
                        if (self.tryMove(x, y, xi + d2, yi)) continue;
                    }
                } else { // GAS
                    if (self.tryMove(x, y, xi, yi - 1)) continue;
                    if (self.tryMove(x, y, xi + d1, yi - 1)) continue;
                    if (self.tryMove(x, y, xi + d2, yi - 1)) continue;
                    if (self.tryMove(x, y, xi + d1, yi)) continue;
                    if (self.tryMove(x, y, xi + d2, yi)) continue;
                }
            }
        }
    }

    fn paint(self: *Sim, px: c_int, py: c_int, material: u8, radius: c_int) void {
        const cx = @divTrunc(px, @as(c_int, PIXEL_SIZE));
        const cy = @divTrunc(py, @as(c_int, PIXEL_SIZE));
        var dy: c_int = -radius;
        while (dy <= radius) : (dy += 1) {
            var dx: c_int = -radius;
            while (dx <= radius) : (dx += 1) {
                const nx = cx + dx;
                const ny = cy + dy;
                if (nx >= 0 and nx < @as(c_int, @intCast(self.w)) and
                    ny >= 0 and ny < @as(c_int, @intCast(self.h)) and
                    dx * dx + dy * dy <= radius * radius)
                {
                    self.grid[@as(usize, @intCast(ny)) * self.w + @as(usize, @intCast(nx))] = material;
                }
            }
        }
    }
};

// --- headless benchmark ----------------------------------------------------

fn runBench(allocator: std.mem.Allocator, steps: usize, width: usize, height: usize) !void {
    const grid = try allocator.alloc(u8, width * height);
    defer allocator.free(grid);
    const moved = try allocator.alloc(u8, width * height);
    defer allocator.free(moved);
    var sim = Sim{ .w = width, .h = height, .grid = grid, .moved = moved };

    var y: usize = 0;
    while (y < height) : (y += 1) {
        var x: usize = 0;
        while (x < width) : (x += 1) grid[y * width + x] = seedMaterial(x, y, width, height);
    }

    var t0: c.struct_timespec = undefined;
    var t1: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &t0);
    var step: usize = 0;
    while (step < steps) : (step += 1) sim.update(@intCast(step));
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &t1);
    const elapsed_ms = @as(f64, @floatFromInt(t1.tv_sec - t0.tv_sec)) * 1000.0 +
        @as(f64, @floatFromInt(t1.tv_nsec - t0.tv_nsec)) / 1.0e6;

    var counts = [_]u64{0} ** 5;
    for (grid) |cell| counts[cell] += 1;
    const cells = @as(f64, @floatFromInt(width * height)) * @as(f64, @floatFromInt(steps));
    const mcells = if (elapsed_ms > 0.0) cells / (elapsed_ms / 1000.0) / 1e6 else 0.0;

    _ = c.printf(
        "RESULT impl=zig_materials rule=materials width=%d height=%d steps=%d elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx empty=%llu wall=%llu sand=%llu water=%llu gas=%llu\n",
        @as(c_int, @intCast(width)),
        @as(c_int, @intCast(height)),
        @as(c_int, @intCast(steps)),
        elapsed_ms,
        mcells,
        @as(c_ulonglong, checksum(grid)),
        @as(c_ulonglong, counts[0]),
        @as(c_ulonglong, counts[1]),
        @as(c_ulonglong, counts[2]),
        @as(c_ulonglong, counts[3]),
        @as(c_ulonglong, counts[4]),
    );
}

// --- interactive mode ------------------------------------------------------

fn runInteractive(allocator: std.mem.Allocator, width: usize, height: usize) !void {
    const render_w = width * PIXEL_SIZE;
    const render_h = height * PIXEL_SIZE;
    const grid = try allocator.alloc(u8, width * height);
    defer allocator.free(grid);
    @memset(grid, EMPTY);
    const moved = try allocator.alloc(u8, width * height);
    defer allocator.free(moved);
    const pixels = try allocator.alloc(u32, render_w * render_h);
    defer allocator.free(pixels);
    var sim = Sim{ .w = width, .h = height, .grid = grid, .moved = moved };
    var current: u8 = SAND;

    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        _ = c.printf("SDL_Init failed: %s\n", c.SDL_GetError());
        return;
    }
    defer c.SDL_Quit();

    const window = c.SDL_CreateWindow(
        "Materials Sand Simulation - [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        @as(c_int, @intCast(render_w)),
        @as(c_int, @intCast(render_h)),
        0,
    );
    defer c.SDL_DestroyWindow(window);
    const renderer = c.SDL_CreateRenderer(window, -1, c.SDL_RENDERER_ACCELERATED);
    defer c.SDL_DestroyRenderer(renderer);
    const texture = c.SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, c.SDL_TEXTUREACCESS_STREAMING, @as(c_int, @intCast(render_w)), @as(c_int, @intCast(render_h)));
    defer c.SDL_DestroyTexture(texture);

    var quit = false;
    var mouse_down = false;
    var mx: c_int = 0;
    var my: c_int = 0;
    var frame: u32 = 0;
    var e: c.SDL_Event = undefined;

    while (!quit) {
        while (c.SDL_PollEvent(&e) != 0) {
            switch (e.type) {
                c.SDL_QUIT => quit = true,
                c.SDL_MOUSEBUTTONDOWN => mouse_down = true,
                c.SDL_MOUSEBUTTONUP => mouse_down = false,
                c.SDL_MOUSEMOTION => _ = c.SDL_GetMouseState(&mx, &my),
                c.SDL_KEYDOWN => switch (e.key.keysym.sym) {
                    '0' => current = EMPTY,
                    '1' => current = WALL,
                    '2' => current = SAND,
                    '3' => current = WATER,
                    '4' => current = GAS,
                    'c' => @memset(grid, EMPTY),
                    else => {},
                },
                else => {},
            }
        }
        if (mouse_down) sim.paint(mx, my, current, 4);
        sim.update(frame);
        frame +%= 1;

        var ry: usize = 0;
        while (ry < height) : (ry += 1) {
            var rx: usize = 0;
            while (rx < width) : (rx += 1) {
                const color = COLORS[grid[ry * width + rx]];
                var py: usize = 0;
                while (py < PIXEL_SIZE) : (py += 1) {
                    var px: usize = 0;
                    while (px < PIXEL_SIZE) : (px += 1) {
                        pixels[(ry * PIXEL_SIZE + py) * render_w + (rx * PIXEL_SIZE + px)] = color;
                    }
                }
            }
        }
        _ = c.SDL_UpdateTexture(texture, null, pixels.ptr, @as(c_int, @intCast(render_w * @sizeOf(u32))));
        _ = c.SDL_RenderClear(renderer);
        _ = c.SDL_RenderCopy(renderer, texture, null, null);
        c.SDL_RenderPresent(renderer);
        c.SDL_Delay(16);
    }
}

pub fn main(init: std.process.Init.Minimal) !void {
    const allocator = std.heap.page_allocator;
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.skip();

    if (it.next()) |first| {
        if (std.mem.eql(u8, first, "--bench")) {
            const steps = if (it.next()) |s| (std.fmt.parseInt(usize, s, 10) catch 1000) else 1000;
            const width = if (it.next()) |s| (std.fmt.parseInt(usize, s, 10) catch 400) else 400;
            const height = if (it.next()) |s| (std.fmt.parseInt(usize, s, 10) catch 300) else 300;
            try runBench(allocator, steps, width, height);
            return;
        }
    }
    try runInteractive(allocator, 400, 300);
}
