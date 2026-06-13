// sandsim - Zig implementation
//
// Canonical "scalar" falling-sand rule, a direct port of
// cpp/sandsim_scalar_sb.cpp. SDL2 is pulled in via @cImport and the one
// machine-readable RESULT line is printed with libc's printf, which keeps the
// program independent of std.io churn across Zig releases.
//
// Modes:
//   (default)                 SDL2 window, 400x300 grid rendered 2x (800x600).
//   --bench [steps] [w] [h]   headless: deterministic seed, time the update
//                             loop, print one RESULT line whose checksum matches
//                             every other scalar-rule implementation.

const std = @import("std");

const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("time.h");
    @cInclude("SDL2/SDL.h");
});

const PIXEL_SIZE: usize = 2;

// SDL constants that come from function-like macros are spelled out here so we
// do not depend on @cImport translating them.
const SDL_WINDOWPOS_UNDEFINED: c_int = 0x1FFF0000;
const SDL_PIXELFORMAT_ARGB8888: u32 = 0x16362004;

// ---------------------------------------------------------------------------
// Shared, language-independent helpers (must match the other implementations).
// ---------------------------------------------------------------------------

// Deterministic per-cell seed (~30% sand), u32 wraparound arithmetic.
fn seedCell(x: usize, y: usize) u8 {
    var h: u32 = @as(u32, @intCast(x)) *% 374761393 +% @as(u32, @intCast(y)) *% 668265263;
    h = (h ^ (h >> 13)) *% 1274126177;
    return if (h % 100 < 30) 1 else 0;
}

// FNV-1a over the grid, row-major, u64 wraparound.
fn checksum(grid: []const u8) u64 {
    var acc: u64 = 14695981039346656037;
    for (grid) |cell| {
        acc = (acc ^ @as(u64, cell)) *% 1099511628211;
    }
    return acc;
}

// One update step: sand falls down, else down-left, else down-right.
fn update(buf: []u8, width: usize, height: usize) void {
    var y: usize = height - 1;
    while (y > 0) {
        y -= 1;
        var x: usize = 0;
        while (x < width) : (x += 1) {
            const i = y * width + x;
            if (buf[i] == 1) {
                const below = (y + 1) * width + x;
                if (buf[below] == 0) {
                    buf[below] = 1;
                    buf[i] = 0;
                } else if (x > 0 and buf[below - 1] == 0) {
                    buf[below - 1] = 1;
                    buf[i] = 0;
                } else if (x < width - 1 and buf[below + 1] == 0) {
                    buf[below + 1] = 1;
                    buf[i] = 0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Headless benchmark.
// ---------------------------------------------------------------------------

fn runBench(allocator: std.mem.Allocator, steps: usize, width: usize, height: usize) !void {
    const buf = try allocator.alloc(u8, width * height);
    defer allocator.free(buf);

    var y: usize = 0;
    while (y < height) : (y += 1) {
        var x: usize = 0;
        while (x < width) : (x += 1) {
            buf[y * width + x] = seedCell(x, y);
        }
    }

    var t0: c.struct_timespec = undefined;
    var t1: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &t0);
    var step: usize = 0;
    while (step < steps) : (step += 1) {
        update(buf, width, height);
    }
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &t1);
    const elapsed_ms = @as(f64, @floatFromInt(t1.tv_sec - t0.tv_sec)) * 1000.0 +
        @as(f64, @floatFromInt(t1.tv_nsec - t0.tv_nsec)) / 1.0e6;

    const cells = @as(f64, @floatFromInt(width * height)) * @as(f64, @floatFromInt(steps));
    const mcells = if (elapsed_ms > 0.0) cells / (elapsed_ms / 1000.0) / 1e6 else 0.0;

    _ = c.printf(
        "RESULT impl=zig rule=scalar width=%d height=%d steps=%d elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx\n",
        @as(c_int, @intCast(width)),
        @as(c_int, @intCast(height)),
        @as(c_int, @intCast(steps)),
        elapsed_ms,
        mcells,
        @as(c_ulonglong, checksum(buf)),
    );
}

// ---------------------------------------------------------------------------
// Interactive SDL2 mode.
// ---------------------------------------------------------------------------

// Tiny LCG so the interactive `R` (randomize) needs no std.Random dependency.
fn lcgNext(state: *u64) f32 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    return @as(f32, @floatFromInt(state.* >> 40)) / @as(f32, @floatFromInt(@as(u32, 1) << 24));
}

fn addSand(buf: []u8, width: usize, height: usize, px: c_int, py: c_int, radius: c_int) void {
    const x = @divTrunc(px, @as(c_int, PIXEL_SIZE));
    const y = @divTrunc(py, @as(c_int, PIXEL_SIZE));
    var dy: c_int = -radius;
    while (dy <= radius) : (dy += 1) {
        var dx: c_int = -radius;
        while (dx <= radius) : (dx += 1) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 and nx < @as(c_int, @intCast(width)) and
                ny >= 0 and ny < @as(c_int, @intCast(height)) and
                dx * dx + dy * dy <= radius * radius)
            {
                buf[@as(usize, @intCast(ny)) * width + @as(usize, @intCast(nx))] = 1;
            }
        }
    }
}

fn runInteractive(allocator: std.mem.Allocator, width: usize, height: usize) !void {
    const render_w = width * PIXEL_SIZE;
    const render_h = height * PIXEL_SIZE;
    const buf = try allocator.alloc(u8, width * height);
    defer allocator.free(buf);
    @memset(buf, 0);
    const pixels = try allocator.alloc(u32, render_w * render_h);
    defer allocator.free(pixels);
    var rng: u64 = 0x9E3779B97F4A7C15;

    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        _ = c.printf("SDL_Init failed: %s\n", c.SDL_GetError());
        return;
    }
    defer c.SDL_Quit();

    const window = c.SDL_CreateWindow(
        "Zig Sand Simulation",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        @as(c_int, @intCast(render_w)),
        @as(c_int, @intCast(render_h)),
        0,
    );
    defer c.SDL_DestroyWindow(window);
    const renderer = c.SDL_CreateRenderer(window, -1, c.SDL_RENDERER_ACCELERATED);
    defer c.SDL_DestroyRenderer(renderer);
    const texture = c.SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        c.SDL_TEXTUREACCESS_STREAMING,
        @as(c_int, @intCast(render_w)),
        @as(c_int, @intCast(render_h)),
    );
    defer c.SDL_DestroyTexture(texture);

    var quit = false;
    var mouse_down = false;
    var mx: c_int = 0;
    var my: c_int = 0;
    var e: c.SDL_Event = undefined;

    while (!quit) {
        while (c.SDL_PollEvent(&e) != 0) {
            switch (e.type) {
                c.SDL_QUIT => quit = true,
                c.SDL_MOUSEBUTTONDOWN => mouse_down = true,
                c.SDL_MOUSEBUTTONUP => mouse_down = false,
                c.SDL_MOUSEMOTION => _ = c.SDL_GetMouseState(&mx, &my),
                c.SDL_KEYDOWN => {
                    if (e.key.keysym.sym == 'c') {
                        @memset(buf, 0);
                    } else if (e.key.keysym.sym == 'r') {
                        for (buf) |*cell| cell.* = if (lcgNext(&rng) < 0.3) 1 else 0;
                    }
                },
                else => {},
            }
        }

        if (mouse_down) addSand(buf, width, height, mx, my, 5);
        update(buf, width, height);

        // Render: 2x2 pixel blocks, yellow sand on black.
        var ry: usize = 0;
        while (ry < height) : (ry += 1) {
            var rx: usize = 0;
            while (rx < width) : (rx += 1) {
                const color: u32 = if (buf[ry * width + rx] != 0) 0xFFFFFF00 else 0xFF000000;
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
    _ = it.skip(); // skip argv[0]

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
