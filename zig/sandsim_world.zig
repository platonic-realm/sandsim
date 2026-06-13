// sandsim - Zig chunked streaming world (Noita-style)
//
// Port of cpp/sandsim_world.cpp: a world larger than memory, with only a few
// "live boxes" (chunks) resident around a camera and the rest streamed to disk.
// See WORLD.md. Resident chunks live in a dense array and are processed
// bottom-chunk-first, reproducing the C++ reference checksum.
//
// Modes:
//   (default)                 SDL2 window; WASD pan, number keys paint.
//   --bench [steps] [wch] [hch]   headless deterministic streaming benchmark.

const std = @import("std");

const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("time.h");
    @cInclude("sys/stat.h");
    @cInclude("SDL2/SDL.h");
});

const EMPTY: u8 = 0;
const WALL: u8 = 1;
const SAND: u8 = 2;
const WATER: u8 = 3;
const GAS: u8 = 4;
const CHUNK: i32 = 64;
const CHUNK_MASK: i32 = 63;
const CHUNK_SHIFT: u5 = 6;
const PIXEL_SIZE: usize = 2;

const COLORS = [_]u32{ 0xFF000000, 0xFF808080, 0xFFE2C878, 0xFF4488FF, 0xFFB0C4DE };
const SDL_WINDOWPOS_UNDEFINED: c_int = 0x1FFF0000;
const SDL_PIXELFORMAT_ARGB8888: u32 = 0x16362004;

fn canEnter(mover: u8, target: u8) bool {
    if (target == WALL) return false;
    return switch (mover) {
        SAND => target == EMPTY or target == WATER or target == GAS,
        WATER => target == EMPTY or target == GAS,
        GAS => target == EMPTY,
        else => false,
    };
}
fn hashCoord(gx: i32, gy: i32) u32 {
    const h = @as(u32, @intCast(gx)) *% 374761393 +% @as(u32, @intCast(gy)) *% 668265263;
    return (h ^ (h >> 13)) *% 1274126177;
}
fn genCell(gx: i32, gy: i32, wcells: i32, hcells: i32) u8 {
    if (gx == 0 or gy == 0 or gx == wcells - 1 or gy == hcells - 1) return WALL;
    if (@mod(gy, 40) == 39 and (@mod(gx, 11) != 0)) return WALL;
    const r = hashCoord(gx, gy) % 100;
    return switch (@mod(@divTrunc(gy, 40), 3)) {
        0 => if (r < 35) SAND else EMPTY,
        1 => if (r < 30) WATER else EMPTY,
        else => if (r < 18) GAS else EMPTY,
    };
}

const N = @as(usize, @intCast(CHUNK * CHUNK));
const Chunk = struct {
    cells: [N]u8,
    moved: [N]u8,
    dminx: i32, dminy: i32, dmaxx: i32, dmaxy: i32,
    nminx: i32, nminy: i32, nmaxx: i32, nmaxy: i32,

    fn fullDirty(self: *Chunk) void { self.dminx = 0; self.dminy = 0; self.dmaxx = CHUNK - 1; self.dmaxy = CHUNK - 1; }
    fn clearNext(self: *Chunk) void { self.nminx = CHUNK; self.nminy = CHUNK; self.nmaxx = -1; self.nmaxy = -1; }
    fn awake(self: *const Chunk) bool { return self.dminx <= self.dmaxx and self.dminy <= self.dmaxy; }
    fn commitNext(self: *Chunk) void {
        self.dminx = self.nminx; self.dminy = self.nminy; self.dmaxx = self.nmaxx; self.dmaxy = self.nmaxy;
        self.clearNext();
    }
};

const World = struct {
    alloc: std.mem.Allocator,
    wch: i32, hch: i32, wcells: i32, hcells: i32,
    dir: []const u8,
    chunks: []?*Chunk,
    frame: u32 = 0,
    resident_max: i32 = 0,
    n_writes: i64 = 0, n_reads: i64 = 0, n_generated: i64 = 0,

    fn init(alloc: std.mem.Allocator, wch: i32, hch: i32, dir: []const u8) !World {
        var buf: [320]u8 = undefined;
        const z = std.fmt.bufPrintZ(&buf, "{s}", .{dir}) catch unreachable;
        _ = c.mkdir(z.ptr, 0o777);
        const chunks = try alloc.alloc(?*Chunk, @intCast(wch * hch));
        @memset(chunks, null);
        return .{ .alloc = alloc, .wch = wch, .hch = hch, .wcells = wch * CHUNK, .hcells = hch * CHUNK, .dir = dir, .chunks = chunks };
    }

    fn residentAt(self: *World, cx: i32, cy: i32) ?*Chunk {
        if (cx < 0 or cy < 0 or cx >= self.wch or cy >= self.hch) return null;
        return self.chunks[@intCast(cy * self.wch + cx)];
    }

    // Disk I/O via libc (independent of the std.Io overhaul in recent Zig).
    fn pathZ(self: *World, buf: []u8, cx: i32, cy: i32) [:0]const u8 {
        return std.fmt.bufPrintZ(buf, "{s}/c_{d}_{d}.bin", .{ self.dir, cx, cy }) catch unreachable;
    }
    fn writeChunk(self: *World, cx: i32, cy: i32, cells: *const [N]u8) void {
        var buf: [320]u8 = undefined;
        const p = self.pathZ(&buf, cx, cy);
        const f = c.fopen(p.ptr, "wb") orelse return;
        _ = c.fwrite(cells, 1, N, f);
        _ = c.fclose(f);
        self.n_writes += 1;
    }
    fn readChunkRaw(self: *World, cx: i32, cy: i32, cells: *[N]u8) bool {
        var buf: [320]u8 = undefined;
        const p = self.pathZ(&buf, cx, cy);
        const f = c.fopen(p.ptr, "rb") orelse return false;
        const got = c.fread(cells, 1, N, f);
        _ = c.fclose(f);
        return got == N;
    }

    fn get(self: *World, gx: i32, gy: i32) u8 {
        if (gx < 0 or gy < 0 or gx >= self.wcells or gy >= self.hcells) return WALL;
        const ch = self.residentAt(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT) orelse return WALL;
        return ch.cells[@intCast((gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK))];
    }

    fn generateAllToDisk(self: *World) !void {
        const ch = try self.alloc.create(Chunk);
        defer self.alloc.destroy(ch);
        var cy: i32 = 0;
        while (cy < self.hch) : (cy += 1) {
            var cx: i32 = 0;
            while (cx < self.wch) : (cx += 1) {
                var ly: i32 = 0;
                while (ly < CHUNK) : (ly += 1) {
                    var lx: i32 = 0;
                    while (lx < CHUNK) : (lx += 1)
                        ch.cells[@intCast(ly * CHUNK + lx)] = genCell(cx * CHUNK + lx, cy * CHUNK + ly, self.wcells, self.hcells);
                }
                self.writeChunk(cx, cy, &ch.cells);
            }
        }
    }

    fn loadOrGenerate(self: *World, cx: i32, cy: i32) !void {
        const ch = try self.alloc.create(Chunk);
        @memset(&ch.moved, 0);
        if (self.readChunkRaw(cx, cy, &ch.cells)) {
            self.n_reads += 1;
        } else {
            var ly: i32 = 0;
            while (ly < CHUNK) : (ly += 1) {
                var lx: i32 = 0;
                while (lx < CHUNK) : (lx += 1)
                    ch.cells[@intCast(ly * CHUNK + lx)] = genCell(cx * CHUNK + lx, cy * CHUNK + ly, self.wcells, self.hcells);
            }
            self.n_generated += 1;
        }
        ch.fullDirty();
        ch.clearNext();
        self.chunks[@intCast(cy * self.wch + cx)] = ch;
    }

    fn streamAround(self: *World, cam_cx: i32, cam_cy: i32, radius: i32) !void {
        var cy: i32 = 0;
        while (cy < self.hch) : (cy += 1) {
            var cx: i32 = 0;
            while (cx < self.wch) : (cx += 1) {
                const slot: usize = @intCast(cy * self.wch + cx);
                if (self.chunks[slot]) |ch| {
                    if (@abs(cx - cam_cx) > radius or @abs(cy - cam_cy) > radius) {
                        self.writeChunk(cx, cy, &ch.cells);
                        self.alloc.destroy(ch);
                        self.chunks[slot] = null;
                    }
                }
            }
        }
        cy = cam_cy - radius;
        while (cy <= cam_cy + radius) : (cy += 1) {
            var cx: i32 = cam_cx - radius;
            while (cx <= cam_cx + radius) : (cx += 1) {
                if (cx >= 0 and cy >= 0 and cx < self.wch and cy < self.hch and self.residentAt(cx, cy) == null)
                    try self.loadOrGenerate(cx, cy);
            }
        }
        var res: i32 = 0;
        for (self.chunks) |ch| {
            if (ch != null) res += 1;
        }
        if (res > self.resident_max) self.resident_max = res;
    }

    fn wake(self: *World, gx: i32, gy: i32) void {
        var dy: i32 = -1;
        while (dy <= 1) : (dy += 1) {
            var dx: i32 = -1;
            while (dx <= 1) : (dx += 1) {
                const nx = gx + dx;
                const ny = gy + dy;
                if (self.residentAt(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT)) |ch| {
                    const lx = nx & CHUNK_MASK;
                    const ly = ny & CHUNK_MASK;
                    if (lx < ch.nminx) ch.nminx = lx;
                    if (ly < ch.nminy) ch.nminy = ly;
                    if (lx > ch.nmaxx) ch.nmaxx = lx;
                    if (ly > ch.nmaxy) ch.nmaxy = ly;
                }
            }
        }
    }

    fn tryMove(self: *World, gx: i32, gy: i32, nx: i32, ny: i32) bool {
        if (nx < 0 or ny < 0 or nx >= self.wcells or ny >= self.hcells) return false;
        const tc = self.residentAt(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT) orelse return false;
        const ti: usize = @intCast((ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK));
        if (tc.moved[ti] != 0) return false;
        const target = tc.cells[ti];
        const sc = self.residentAt(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT).?;
        const si: usize = @intCast((gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK));
        const self_mat = sc.cells[si];
        if (!canEnter(self_mat, target)) return false;
        tc.cells[ti] = self_mat;
        sc.cells[si] = target;
        tc.moved[ti] = 1;
        sc.moved[si] = 1;
        self.wake(gx, gy);
        self.wake(nx, ny);
        return true;
    }

    fn step(self: *World) void {
        for (self.chunks) |maybe| {
            if (maybe) |ch| @memset(&ch.moved, 0);
        }
        var cy: i32 = self.hch - 1;
        while (cy >= 0) : (cy -= 1) {
            var cx: i32 = 0;
            while (cx < self.wch) : (cx += 1) {
                const ch = self.residentAt(cx, cy) orelse continue;
                if (!ch.awake()) continue;
                const baseX = cx * CHUNK;
                const baseY = cy * CHUNK;
                var ly: i32 = ch.dmaxy;
                while (ly >= ch.dminy) : (ly -= 1) {
                    var lx: i32 = ch.dminx;
                    while (lx <= ch.dmaxx) : (lx += 1) {
                        const idx: usize = @intCast(ly * CHUNK + lx);
                        if (ch.moved[idx] != 0) continue;
                        const m = ch.cells[idx];
                        if (m == EMPTY or m == WALL) continue;
                        const gx = baseX + lx;
                        const gy = baseY + ly;
                        const left = ((@as(u32, @intCast(gx)) +% @as(u32, @intCast(gy)) +% self.frame) & 1) == 0;
                        const d1: i32 = if (left) -1 else 1;
                        const d2: i32 = -d1;
                        if (m == SAND or m == WATER) {
                            if (self.tryMove(gx, gy, gx, gy + 1)) continue;
                            if (self.tryMove(gx, gy, gx + d1, gy + 1)) continue;
                            if (self.tryMove(gx, gy, gx + d2, gy + 1)) continue;
                            if (m == WATER) {
                                if (self.tryMove(gx, gy, gx + d1, gy)) continue;
                                if (self.tryMove(gx, gy, gx + d2, gy)) continue;
                            }
                        } else {
                            if (self.tryMove(gx, gy, gx, gy - 1)) continue;
                            if (self.tryMove(gx, gy, gx + d1, gy - 1)) continue;
                            if (self.tryMove(gx, gy, gx + d2, gy - 1)) continue;
                            if (self.tryMove(gx, gy, gx + d1, gy)) continue;
                            if (self.tryMove(gx, gy, gx + d2, gy)) continue;
                        }
                    }
                }
            }
        }
        for (self.chunks) |maybe| {
            if (maybe) |ch| ch.commitNext();
        }
        self.frame += 1;
    }

    fn paint(self: *World, gx: i32, gy: i32, material: u8, radius: i32) void {
        var dy: i32 = -radius;
        while (dy <= radius) : (dy += 1) {
            var dx: i32 = -radius;
            while (dx <= radius) : (dx += 1) {
                const nx = gx + dx;
                const ny = gy + dy;
                if (dx * dx + dy * dy > radius * radius) continue;
                if (self.residentAt(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT)) |ch| {
                    ch.cells[@intCast((ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK))] = material;
                    self.wake(nx, ny);
                }
            }
        }
    }

    fn summary(self: *World, counts: *[5]u64) u64 {
        @memset(counts, 0);
        var cksum: u64 = 14695981039346656037;
        var tmp: [N]u8 = undefined;
        var cy: i32 = 0;
        while (cy < self.hch) : (cy += 1) {
            var cx: i32 = 0;
            while (cx < self.wch) : (cx += 1) {
                var cells: *const [N]u8 = undefined;
                if (self.residentAt(cx, cy)) |ch| {
                    cells = &ch.cells;
                } else {
                    _ = self.readChunkRaw(cx, cy, &tmp);
                    cells = &tmp;
                }
                for (cells) |v| {
                    counts[v] += 1;
                    cksum = (cksum ^ @as(u64, v)) *% 1099511628211;
                }
            }
        }
        return cksum;
    }
};

fn runBench(alloc: std.mem.Allocator, steps: i64, wch: i32, hch: i32) !u8 {
    var dirbuf: [128]u8 = undefined;
    const dir = std.fmt.bufPrint(&dirbuf, "/tmp/sandsim_world_zig_{d}_{d}x{d}", .{ steps, wch, hch }) catch unreachable;
    var world = try World.init(alloc, wch, hch, dir);
    try world.generateAllToDisk();

    var startCnt: [5]u64 = undefined;
    _ = world.summary(&startCnt);

    const cells = wch * hch;
    var t0: c.struct_timespec = undefined;
    var t1: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &t0);
    var s: i64 = 0;
    while (s < steps) : (s += 1) {
        var visit: i32 = @intCast(@divTrunc(s * cells, steps));
        if (visit >= cells) visit = cells - 1;
        const row = @divTrunc(visit, wch);
        const col = @mod(visit, wch);
        const cam_cx = if (@mod(row, 2) == 0) col else (wch - 1 - col);
        try world.streamAround(cam_cx, row, 1);
        world.step();
    }
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &t1);
    const elapsed_ms = @as(f64, @floatFromInt(t1.tv_sec - t0.tv_sec)) * 1000.0 +
        @as(f64, @floatFromInt(t1.tv_nsec - t0.tv_nsec)) / 1.0e6;

    var cnt: [5]u64 = undefined;
    const ck = world.summary(&cnt);
    var conserved = true;
    var i: usize = WALL;
    while (i <= GAS) : (i += 1) {
        if (cnt[i] != startCnt[i]) conserved = false;
    }

    const cons: [*:0]const u8 = if (conserved) "yes" else "no";
    _ = c.printf("RESULT impl=zig_world rule=world wchunks=%d hchunks=%d chunk=%d steps=%d elapsed_ms=%.3f checksum=%016llx empty=%llu wall=%llu sand=%llu water=%llu gas=%llu resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n", @as(c_int, @intCast(wch)), @as(c_int, @intCast(hch)), @as(c_int, CHUNK), @as(c_int, @intCast(steps)), elapsed_ms, @as(c_ulonglong, ck), @as(c_ulonglong, cnt[0]), @as(c_ulonglong, cnt[1]), @as(c_ulonglong, cnt[2]), @as(c_ulonglong, cnt[3]), @as(c_ulonglong, cnt[4]), @as(c_int, world.resident_max), @as(c_longlong, world.n_writes), @as(c_longlong, world.n_reads), cons);
    return if (conserved) 0 else 2;
}

fn runInteractive(alloc: std.mem.Allocator) !void {
    const VIEW_W: i32 = 320;
    const VIEW_H: i32 = 240;
    const WCH: i32 = 64;
    const HCH: i32 = 64;
    const render_w: usize = @intCast(VIEW_W * @as(i32, PIXEL_SIZE));
    const render_h: usize = @intCast(VIEW_H * @as(i32, PIXEL_SIZE));

    var world = try World.init(alloc, WCH, HCH, "/tmp/sandsim_world_zig_interactive");
    var camX: i32 = @divTrunc(WCH * CHUNK, 2) - @divTrunc(VIEW_W, 2);
    var camY: i32 = @divTrunc(HCH * CHUNK, 2) - @divTrunc(VIEW_H, 2);
    var current: u8 = SAND;
    const pixels = try alloc.alloc(u32, render_w * render_h);
    defer alloc.free(pixels);

    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) return;
    defer c.SDL_Quit();
    const window = c.SDL_CreateWindow("Streamed World (Zig) - WASD pan  [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, @as(c_int, @intCast(render_w)), @as(c_int, @intCast(render_h)), 0);
    defer c.SDL_DestroyWindow(window);
    const renderer = c.SDL_CreateRenderer(window, -1, c.SDL_RENDERER_ACCELERATED);
    defer c.SDL_DestroyRenderer(renderer);
    const texture = c.SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, c.SDL_TEXTUREACCESS_STREAMING, @as(c_int, @intCast(render_w)), @as(c_int, @intCast(render_h)));
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
                c.SDL_KEYDOWN => switch (e.key.keysym.sym) {
                    '0' => current = EMPTY,
                    '1' => current = WALL,
                    '2' => current = SAND,
                    '3' => current = WATER,
                    '4' => current = GAS,
                    'w' => camY -= 16,
                    's' => camY += 16,
                    'a' => camX -= 16,
                    'd' => camX += 16,
                    else => {},
                },
                else => {},
            }
        }
        try world.streamAround((camX + @divTrunc(VIEW_W, 2)) >> CHUNK_SHIFT, (camY + @divTrunc(VIEW_H, 2)) >> CHUNK_SHIFT, 3);
        if (mouse_down) world.paint(camX + @divTrunc(mx, @as(c_int, PIXEL_SIZE)), camY + @divTrunc(my, @as(c_int, PIXEL_SIZE)), current, 4);
        world.step();
        var vy: i32 = 0;
        while (vy < VIEW_H) : (vy += 1) {
            var vx: i32 = 0;
            while (vx < VIEW_W) : (vx += 1) {
                const color = COLORS[world.get(camX + vx, camY + vy)];
                var dy: usize = 0;
                while (dy < PIXEL_SIZE) : (dy += 1) {
                    var dx: usize = 0;
                    while (dx < PIXEL_SIZE) : (dx += 1)
                        pixels[(@as(usize, @intCast(vy)) * PIXEL_SIZE + dy) * render_w + (@as(usize, @intCast(vx)) * PIXEL_SIZE + dx)] = color;
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
    const alloc = std.heap.page_allocator;
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.skip();
    if (it.next()) |first| {
        if (std.mem.eql(u8, first, "--bench")) {
            const steps = if (it.next()) |s| (std.fmt.parseInt(i64, s, 10) catch 600) else 600;
            const wch = if (it.next()) |s| (std.fmt.parseInt(i32, s, 10) catch 6) else 6;
            const hch = if (it.next()) |s| (std.fmt.parseInt(i32, s, 10) catch 6) else 6;
            _ = try runBench(alloc, steps, wch, hch);
            return;
        }
    }
    try runInteractive(alloc);
}
