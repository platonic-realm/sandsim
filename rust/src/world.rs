//! sandsim - Rust chunked streaming world (Noita-style).
//!
//! Port of cpp/sandsim_world.cpp: a world larger than memory, with only a few
//! "live boxes" (chunks) resident around a camera and the rest streamed to disk.
//! See WORLD.md. Resident chunks live in a dense Vec and are processed
//! bottom-chunk-first, reproducing the C++ reference checksum.
//!
//! Modes:
//!   (default)                 SDL2 window; WASD pan, number keys paint.
//!   --bench [steps] [wch] [hch]   headless deterministic streaming benchmark.

use std::time::Instant;

mod sdl;

const EMPTY: u8 = 0;
const WALL: u8 = 1;
const SAND: u8 = 2;
const WATER: u8 = 3;
const GAS: u8 = 4;
const CHUNK: i32 = 64;
const CHUNK_MASK: i32 = 63;
const CHUNK_SHIFT: u32 = 6;
const PIXEL: usize = 2;

const COLORS: [u32; 5] = [0xFF000000, 0xFF808080, 0xFFE2C878, 0xFF4488FF, 0xFFB0C4DE];

#[inline]
fn can_enter(mover: u8, target: u8) -> bool {
    if target == WALL {
        return false;
    }
    match mover {
        SAND => target == EMPTY || target == WATER || target == GAS,
        WATER => target == EMPTY || target == GAS,
        GAS => target == EMPTY,
        _ => false,
    }
}

#[inline]
fn hash_coord(gx: i32, gy: i32) -> u32 {
    let h = (gx as u32)
        .wrapping_mul(374761393)
        .wrapping_add((gy as u32).wrapping_mul(668265263));
    (h ^ (h >> 13)).wrapping_mul(1274126177)
}

fn gen_cell(gx: i32, gy: i32, wcells: i32, hcells: i32) -> u8 {
    if gx == 0 || gy == 0 || gx == wcells - 1 || gy == hcells - 1 {
        return WALL;
    }
    if gy % 40 == 39 && (gx % 11 != 0) {
        return WALL;
    }
    let r = hash_coord(gx, gy) % 100;
    match (gy / 40) % 3 {
        0 => if r < 35 { SAND } else { EMPTY },
        1 => if r < 30 { WATER } else { EMPTY },
        _ => if r < 18 { GAS } else { EMPTY },
    }
}

struct Chunk {
    cells: Vec<u8>,
    moved: Vec<u8>,
    dminx: i32, dminy: i32, dmaxx: i32, dmaxy: i32,
    nminx: i32, nminy: i32, nmaxx: i32, nmaxy: i32,
}
impl Chunk {
    fn new() -> Self {
        let n = (CHUNK * CHUNK) as usize;
        let mut c = Chunk {
            cells: vec![EMPTY; n], moved: vec![0; n],
            dminx: 0, dminy: 0, dmaxx: 0, dmaxy: 0,
            nminx: 0, nminy: 0, nmaxx: 0, nmaxy: 0,
        };
        c.full_dirty();
        c.clear_next();
        c
    }
    fn full_dirty(&mut self) { self.dminx = 0; self.dminy = 0; self.dmaxx = CHUNK - 1; self.dmaxy = CHUNK - 1; }
    fn clear_next(&mut self) { self.nminx = CHUNK; self.nminy = CHUNK; self.nmaxx = -1; self.nmaxy = -1; }
    fn awake(&self) -> bool { self.dminx <= self.dmaxx && self.dminy <= self.dmaxy }
    fn commit_next(&mut self) {
        self.dminx = self.nminx; self.dminy = self.nminy; self.dmaxx = self.nmaxx; self.dmaxy = self.nmaxy;
        self.clear_next();
    }
}

struct World {
    wch: i32, hch: i32, wcells: i32, hcells: i32,
    dir: String,
    chunks: Vec<Option<Chunk>>,
    frame: u32,
    resident_max: i32,
    n_writes: i64, n_reads: i64, n_generated: i64,
}

impl World {
    fn new(wch: i32, hch: i32, dir: String) -> Self {
        std::fs::create_dir_all(&dir).ok();
        let mut chunks = Vec::with_capacity((wch * hch) as usize);
        for _ in 0..wch * hch { chunks.push(None); }
        World {
            wch, hch, wcells: wch * CHUNK, hcells: hch * CHUNK, dir, chunks,
            frame: 0, resident_max: 0, n_writes: 0, n_reads: 0, n_generated: 0,
        }
    }

    #[inline]
    fn slot(&self, cx: i32, cy: i32) -> Option<usize> {
        if cx < 0 || cy < 0 || cx >= self.wch || cy >= self.hch { None }
        else { Some((cy * self.wch + cx) as usize) }
    }
    fn resident(&self, cx: i32, cy: i32) -> Option<&Chunk> {
        self.slot(cx, cy).and_then(|s| self.chunks[s].as_ref())
    }

    fn path(&self, cx: i32, cy: i32) -> String { format!("{}/c_{}_{}.bin", self.dir, cx, cy) }
    fn write_chunk(&mut self, cx: i32, cy: i32, cells: &[u8]) {
        std::fs::write(self.path(cx, cy), cells).ok();
        self.n_writes += 1;
    }
    fn read_chunk_raw(&self, cx: i32, cy: i32) -> Option<Vec<u8>> {
        std::fs::read(self.path(cx, cy)).ok()
    }

    fn get(&self, gx: i32, gy: i32) -> u8 {
        if gx < 0 || gy < 0 || gx >= self.wcells || gy >= self.hcells {
            return WALL;
        }
        match self.resident(gx >> CHUNK_SHIFT, gy >> CHUNK_SHIFT) {
            Some(c) => c.cells[((gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK)) as usize],
            None => WALL,
        }
    }

    fn generate_all_to_disk(&mut self) {
        for cy in 0..self.hch {
            for cx in 0..self.wch {
                let mut cells = vec![EMPTY; (CHUNK * CHUNK) as usize];
                for ly in 0..CHUNK {
                    for lx in 0..CHUNK {
                        cells[(ly * CHUNK + lx) as usize] =
                            gen_cell(cx * CHUNK + lx, cy * CHUNK + ly, self.wcells, self.hcells);
                    }
                }
                self.write_chunk(cx, cy, &cells);
            }
        }
    }

    fn load_or_generate(&mut self, cx: i32, cy: i32) {
        let mut ch = Chunk::new();
        if let Some(raw) = self.read_chunk_raw(cx, cy) {
            ch.cells = raw;
            self.n_reads += 1;
        } else {
            for ly in 0..CHUNK {
                for lx in 0..CHUNK {
                    ch.cells[(ly * CHUNK + lx) as usize] =
                        gen_cell(cx * CHUNK + lx, cy * CHUNK + ly, self.wcells, self.hcells);
                }
            }
            self.n_generated += 1;
        }
        ch.full_dirty();
        ch.clear_next();
        let s = (cy * self.wch + cx) as usize;
        self.chunks[s] = Some(ch);
    }

    fn stream_around(&mut self, cam_cx: i32, cam_cy: i32, radius: i32) {
        for cy in 0..self.hch {
            for cx in 0..self.wch {
                let s = (cy * self.wch + cx) as usize;
                if self.chunks[s].is_some()
                    && ((cx - cam_cx).abs() > radius || (cy - cam_cy).abs() > radius)
                {
                    let cells = std::mem::take(&mut self.chunks[s]).unwrap().cells;
                    self.write_chunk(cx, cy, &cells);
                }
            }
        }
        for cy in (cam_cy - radius)..=(cam_cy + radius) {
            for cx in (cam_cx - radius)..=(cam_cx + radius) {
                if cx >= 0 && cy >= 0 && cx < self.wch && cy < self.hch && self.resident(cx, cy).is_none() {
                    self.load_or_generate(cx, cy);
                }
            }
        }
        let res = self.chunks.iter().filter(|c| c.is_some()).count() as i32;
        if res > self.resident_max { self.resident_max = res; }
    }

    fn wake(&mut self, gx: i32, gy: i32) {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (gx + dx, gy + dy);
                if let Some(s) = self.slot(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT) {
                    if let Some(c) = self.chunks[s].as_mut() {
                        let (lx, ly) = (nx & CHUNK_MASK, ny & CHUNK_MASK);
                        if lx < c.nminx { c.nminx = lx; }
                        if ly < c.nminy { c.nminy = ly; }
                        if lx > c.nmaxx { c.nmaxx = lx; }
                        if ly > c.nmaxy { c.nmaxy = ly; }
                    }
                }
            }
        }
    }

    fn try_move(&mut self, gx: i32, gy: i32, nx: i32, ny: i32) -> bool {
        if nx < 0 || ny < 0 || nx >= self.wcells || ny >= self.hcells {
            return false;
        }
        let ts = match self.slot(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT) {
            Some(s) if self.chunks[s].is_some() => s,
            _ => return false,
        };
        let ti = ((ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK)) as usize;
        if self.chunks[ts].as_ref().unwrap().moved[ti] != 0 {
            return false;
        }
        let target = self.chunks[ts].as_ref().unwrap().cells[ti];
        let ss = (((gy >> CHUNK_SHIFT) * self.wch) + (gx >> CHUNK_SHIFT)) as usize;
        let si = ((gy & CHUNK_MASK) * CHUNK + (gx & CHUNK_MASK)) as usize;
        let self_mat = self.chunks[ss].as_ref().unwrap().cells[si];
        if !can_enter(self_mat, target) {
            return false;
        }
        // swap (ss and ts may be the same chunk)
        self.chunks[ts].as_mut().unwrap().cells[ti] = self_mat;
        self.chunks[ts].as_mut().unwrap().moved[ti] = 1;
        self.chunks[ss].as_mut().unwrap().cells[si] = target;
        self.chunks[ss].as_mut().unwrap().moved[si] = 1;
        self.wake(gx, gy);
        self.wake(nx, ny);
        true
    }

    fn step(&mut self) {
        for c in self.chunks.iter_mut().flatten() {
            for m in c.moved.iter_mut() { *m = 0; }
        }
        for cy in (0..self.hch).rev() {
            for cx in 0..self.wch {
                let s = (cy * self.wch + cx) as usize;
                let (awake, dminx, dminy, dmaxx, dmaxy) = match &self.chunks[s] {
                    Some(c) if c.awake() => (true, c.dminx, c.dminy, c.dmaxx, c.dmaxy),
                    _ => continue,
                };
                let _ = awake;
                let (base_x, base_y) = (cx * CHUNK, cy * CHUNK);
                let mut ly = dmaxy;
                while ly >= dminy {
                    for lx in dminx..=dmaxx {
                        if self.chunks[s].as_ref().unwrap().moved[(ly * CHUNK + lx) as usize] != 0 {
                            continue;
                        }
                        let m = self.chunks[s].as_ref().unwrap().cells[(ly * CHUNK + lx) as usize];
                        if m == EMPTY || m == WALL { continue; }
                        let (gx, gy) = (base_x + lx, base_y + ly);
                        let left = ((gx as u32 + gy as u32 + self.frame) & 1) == 0;
                        let (d1, d2) = if left { (-1, 1) } else { (1, -1) };
                        if m == SAND || m == WATER {
                            if self.try_move(gx, gy, gx, gy + 1) { continue; }
                            if self.try_move(gx, gy, gx + d1, gy + 1) { continue; }
                            if self.try_move(gx, gy, gx + d2, gy + 1) { continue; }
                            if m == WATER {
                                if self.try_move(gx, gy, gx + d1, gy) { continue; }
                                if self.try_move(gx, gy, gx + d2, gy) { continue; }
                            }
                        } else {
                            if self.try_move(gx, gy, gx, gy - 1) { continue; }
                            if self.try_move(gx, gy, gx + d1, gy - 1) { continue; }
                            if self.try_move(gx, gy, gx + d2, gy - 1) { continue; }
                            if self.try_move(gx, gy, gx + d1, gy) { continue; }
                            if self.try_move(gx, gy, gx + d2, gy) { continue; }
                        }
                    }
                    ly -= 1;
                }
            }
        }
        for c in self.chunks.iter_mut().flatten() { c.commit_next(); }
        self.frame += 1;
    }

    // The world border (generated as WALL) is the indestructible solid shell,
    // including the bottom floor: painting never modifies it.
    fn indestructible(&self, gx: i32, gy: i32) -> bool {
        gx == 0 || gy == 0 || gx == self.wcells - 1 || gy == self.hcells - 1
    }

    fn paint(&mut self, gx: i32, gy: i32, material: u8, radius: i32) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (nx, ny) = (gx + dx, gy + dy);
                if dx * dx + dy * dy > radius * radius { continue; }
                if self.indestructible(nx, ny) { continue; }
                if let Some(s) = self.slot(nx >> CHUNK_SHIFT, ny >> CHUNK_SHIFT) {
                    if self.chunks[s].is_some() {
                        let i = ((ny & CHUNK_MASK) * CHUNK + (nx & CHUNK_MASK)) as usize;
                        self.chunks[s].as_mut().unwrap().cells[i] = material;
                        self.wake(nx, ny);
                    }
                }
            }
        }
    }

    fn summary(&self) -> (u64, [u64; 5]) {
        let mut counts = [0u64; 5];
        let mut c: u64 = 14695981039346656037;
        for cy in 0..self.hch {
            for cx in 0..self.wch {
                let owned = if self.resident(cx, cy).is_some() { None } else { self.read_chunk_raw(cx, cy) };
                let cells: &[u8] = match self.resident(cx, cy) {
                    Some(ch) => &ch.cells,
                    None => owned.as_deref().unwrap_or(&[]),
                };
                for &v in cells {
                    counts[v as usize] += 1;
                    c = (c ^ v as u64).wrapping_mul(1099511628211);
                }
            }
        }
        (c, counts)
    }
}

fn run_bench(steps: i64, wch: i32, hch: i32) -> i32 {
    let dir = format!("/tmp/sandsim_world_rust_{}_{}x{}", steps, wch, hch);
    let mut world = World::new(wch, hch, dir);
    world.generate_all_to_disk();
    let (_, start_cnt) = world.summary();

    let cells = (wch * hch) as i64;
    let start = Instant::now();
    for s in 0..steps {
        let visit = (((s * cells) / steps) as i32).min(cells as i32 - 1);
        let (row, col) = (visit / wch, visit % wch);
        let cam_cx = if row % 2 == 0 { col } else { wch - 1 - col };
        world.stream_around(cam_cx, row, 1);
        world.step();
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (ck, cnt) = world.summary();
    let conserved = (WALL..=GAS).all(|i| cnt[i as usize] == start_cnt[i as usize]);
    println!(
        "RESULT impl=rust_world rule=world wchunks={} hchunks={} chunk={} steps={} \
         elapsed_ms={:.3} checksum={:016x} empty={} wall={} sand={} water={} gas={} \
         resident_max={} disk_writes={} disk_reads={} conserved={}",
        wch, hch, CHUNK, steps, elapsed_ms, ck,
        cnt[0], cnt[1], cnt[2], cnt[3], cnt[4],
        world.resident_max, world.n_writes, world.n_reads,
        if conserved { "yes" } else { "no" }
    );
    if conserved { 0 } else { 2 }
}

fn run_interactive() {
    const VIEW_W: i32 = 320;
    const VIEW_H: i32 = 240;
    const WCH: i32 = 64;
    const HCH: i32 = 64;
    let (render_w, render_h) = ((VIEW_W as usize) * PIXEL, (VIEW_H as usize) * PIXEL);
    let mut world = World::new(WCH, HCH, "/tmp/sandsim_world_rust_interactive".to_string());
    let mut cam_x = WCH * CHUNK / 2 - VIEW_W / 2;
    let mut cam_y = HCH * CHUNK / 2 - VIEW_H / 2;
    let mut current = SAND;
    let mut pixels = vec![0u32; render_w * render_h];

    unsafe {
        if sdl::SDL_Init(sdl::INIT_VIDEO) != 0 { return; }
        let title = b"Streamed World (Rust) - WASD pan  [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser\0".as_ptr() as *const _;
        let window = sdl::SDL_CreateWindow(title, sdl::WINDOWPOS_UNDEFINED, sdl::WINDOWPOS_UNDEFINED,
                                           render_w as i32, render_h as i32, 0);
        let renderer = sdl::SDL_CreateRenderer(window, -1, sdl::RENDERER_ACCELERATED);
        // Map rendering and the cursor through a fixed logical size, so painting
        // lands under the pointer even when a tiling compositor (e.g. niri)
        // resizes the window away from the requested size.
        sdl::SDL_RenderSetLogicalSize(renderer, render_w as i32, render_h as i32);
        let texture = sdl::SDL_CreateTexture(renderer, sdl::PIXELFORMAT_ARGB8888,
                                             sdl::TEXTUREACCESS_STREAMING, render_w as i32, render_h as i32);
        let mut quit = false;
        let mut mouse_down = false;
        let (mut mx, mut my) = (0i32, 0i32);
        let mut ev = sdl::Event::new();
        while !quit {
            while sdl::SDL_PollEvent(&mut ev) != 0 {
                match ev.kind() {
                    sdl::QUIT => quit = true,
                    sdl::MOUSEBUTTONDOWN => mouse_down = true,
                    sdl::MOUSEBUTTONUP => mouse_down = false,
                    sdl::MOUSEMOTION => { sdl::SDL_GetMouseState(&mut mx, &mut my); }
                    sdl::KEYDOWN => match ev.keysym() {
                        k if k == '0' as i32 => current = EMPTY,
                        k if k == '1' as i32 => current = WALL,
                        k if k == '2' as i32 => current = SAND,
                        k if k == '3' as i32 => current = WATER,
                        k if k == '4' as i32 => current = GAS,
                        _ => {}
                    },
                    _ => {}
                }
            }
            // Arrow keys or WASD pan the camera while held (smooth, continuous).
            let keys = sdl::SDL_GetKeyboardState(std::ptr::null_mut());
            let down = |sc: usize| *keys.add(sc) != 0;
            let pan = 6;
            if down(sdl::SCANCODE_LEFT)  || down(sdl::SCANCODE_A) { cam_x -= pan; }
            if down(sdl::SCANCODE_RIGHT) || down(sdl::SCANCODE_D) { cam_x += pan; }
            if down(sdl::SCANCODE_UP)    || down(sdl::SCANCODE_W) { cam_y -= pan; }
            if down(sdl::SCANCODE_DOWN)  || down(sdl::SCANCODE_S) { cam_y += pan; }

            world.stream_around((cam_x + VIEW_W / 2) >> CHUNK_SHIFT, (cam_y + VIEW_H / 2) >> CHUNK_SHIFT, 3);
            if mouse_down {
                let (mut lx, mut ly) = (0f32, 0f32);
                sdl::SDL_RenderWindowToLogical(renderer, mx, my, &mut lx, &mut ly);
                world.paint(cam_x + lx as i32 / PIXEL as i32, cam_y + ly as i32 / PIXEL as i32, current, 4);
            }
            world.step();
            for vy in 0..VIEW_H {
                for vx in 0..VIEW_W {
                    let color = COLORS[world.get(cam_x + vx, cam_y + vy) as usize];
                    for dy in 0..PIXEL {
                        for dx in 0..PIXEL {
                            pixels[(vy as usize * PIXEL + dy) * render_w + (vx as usize * PIXEL + dx)] = color;
                        }
                    }
                }
            }
            sdl::SDL_UpdateTexture(texture, std::ptr::null(), pixels.as_ptr() as *const _,
                                   (render_w * std::mem::size_of::<u32>()) as i32);
            sdl::SDL_RenderClear(renderer);
            sdl::SDL_RenderCopy(renderer, texture, std::ptr::null(), std::ptr::null());
            sdl::SDL_RenderPresent(renderer);
            sdl::SDL_Delay(16);
        }
        sdl::SDL_DestroyTexture(texture);
        sdl::SDL_DestroyRenderer(renderer);
        sdl::SDL_DestroyWindow(window);
        sdl::SDL_Quit();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--bench" {
        let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(600);
        let wch = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
        let hch = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
        std::process::exit(run_bench(steps, wch, hch));
    }
    run_interactive();
}
