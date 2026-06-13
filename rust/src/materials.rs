//! sandsim - Rust multi-material engine (Noita-style).
//!
//! Port of cpp/sandsim_materials.cpp. Materials: EMPTY, WALL, SAND (powder),
//! WATER (liquid), GAS. Movement is a swap (every material is conserved); a
//! mover may swap into a target only if the target is strictly lighter in the
//! relevant direction (SAND sinks through WATER and GAS; WATER sinks through
//! GAS; GAS rises into EMPTY). A per-frame "moved" flag keeps the in-place
//! update deterministic for both up- and down-movers.
//!
//! Modes:
//!   (default)                 SDL2 window; number keys pick a material, the
//!                             left mouse button paints it.
//!   --bench [steps] [w] [h]   headless: deterministic scene, print a RESULT
//!                             line whose checksum matches the other ports.

use std::time::Instant;

mod sdl;

const EMPTY: u8 = 0;
const WALL: u8 = 1;
const SAND: u8 = 2;
const WATER: u8 = 3;
const GAS: u8 = 4;
const PIXEL: usize = 2;

const COLORS: [u32; 5] = [0xFF000000, 0xFF808080, 0xFFE2C878, 0xFF4488FF, 0xFFB0C4DE];

// --- shared rule (identical across all material ports) ---------------------

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
fn seed_material(x: i32, y: i32, w: i32, h: i32) -> u8 {
    if x == 0 || x == w - 1 || y == h - 1 {
        return WALL;
    }
    if y == h / 2 && (x % 20 != 0) {
        return WALL;
    }
    let mut hsh = (x as u32)
        .wrapping_mul(374761393)
        .wrapping_add((y as u32).wrapping_mul(668265263));
    hsh = (hsh ^ (hsh >> 13)).wrapping_mul(1274126177);
    let r = hsh % 100;
    if y < h / 3 {
        if r < 40 { SAND } else { EMPTY }
    } else if y < 2 * h / 3 {
        if r < 35 { WATER } else { EMPTY }
    } else if r < 20 {
        GAS
    } else {
        EMPTY
    }
}

fn checksum(grid: &[u8]) -> u64 {
    let mut c: u64 = 14695981039346656037;
    for &cell in grid {
        c = (c ^ cell as u64).wrapping_mul(1099511628211);
    }
    c
}

struct Sim {
    w: usize,
    h: usize,
    grid: Vec<u8>,
    moved: Vec<u8>,
}

impl Sim {
    fn new(w: usize, h: usize) -> Self {
        Sim { w, h, grid: vec![EMPTY; w * h], moved: vec![0; w * h] }
    }

    fn seed_bench_scene(&mut self) {
        for y in 0..self.h {
            for x in 0..self.w {
                self.grid[y * self.w + x] = seed_material(x as i32, y as i32, self.w as i32, self.h as i32);
            }
        }
    }

    #[inline]
    fn try_move(&mut self, x: usize, y: usize, nx: i32, ny: i32) -> bool {
        if nx < 0 || nx >= self.w as i32 || ny < 0 || ny >= self.h as i32 {
            return false;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        let ni = ny * self.w + nx;
        if self.moved[ni] != 0 {
            return false;
        }
        let i = y * self.w + x;
        let target = self.grid[ni];
        if !can_enter(self.grid[i], target) {
            return false;
        }
        self.grid[ni] = self.grid[i];
        self.grid[i] = target; // swap conserves all materials
        self.moved[ni] = 1;
        self.moved[i] = 1;
        true
    }

    fn update(&mut self, frame: u32) {
        for m in self.moved.iter_mut() {
            *m = 0;
        }
        for y in (0..self.h).rev() {
            for x in 0..self.w {
                if self.moved[y * self.w + x] != 0 {
                    continue;
                }
                let m = self.grid[y * self.w + x];
                if m == EMPTY || m == WALL {
                    continue;
                }
                let left = ((x as u32 + y as u32 + frame) & 1) == 0;
                let (d1, d2) = if left { (-1i32, 1i32) } else { (1i32, -1i32) };
                let (xi, yi) = (x as i32, y as i32);
                if m == SAND || m == WATER {
                    if self.try_move(x, y, xi, yi + 1) { continue; }
                    if self.try_move(x, y, xi + d1, yi + 1) { continue; }
                    if self.try_move(x, y, xi + d2, yi + 1) { continue; }
                    if m == WATER {
                        if self.try_move(x, y, xi + d1, yi) { continue; }
                        if self.try_move(x, y, xi + d2, yi) { continue; }
                    }
                } else {
                    // GAS
                    if self.try_move(x, y, xi, yi - 1) { continue; }
                    if self.try_move(x, y, xi + d1, yi - 1) { continue; }
                    if self.try_move(x, y, xi + d2, yi - 1) { continue; }
                    if self.try_move(x, y, xi + d1, yi) { continue; }
                    if self.try_move(x, y, xi + d2, yi) { continue; }
                }
            }
        }
    }

    fn paint(&mut self, px: i32, py: i32, material: u8, radius: i32) {
        let cx = px / PIXEL as i32;
        let cy = py / PIXEL as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && nx < self.w as i32 && ny >= 0 && ny < self.h as i32
                    && dx * dx + dy * dy <= radius * radius
                {
                    self.grid[ny as usize * self.w + nx as usize] = material;
                }
            }
        }
    }
}

// --- headless benchmark ----------------------------------------------------

fn run_bench(steps: u64, width: usize, height: usize) {
    let mut sim = Sim::new(width, height);
    sim.seed_bench_scene();

    let start = Instant::now();
    for s in 0..steps {
        sim.update(s as u32);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut counts = [0u64; 5];
    for &c in &sim.grid {
        counts[c as usize] += 1;
    }
    let cells = width as f64 * height as f64 * steps as f64;
    let mcells = if elapsed_ms > 0.0 { cells / (elapsed_ms / 1000.0) / 1e6 } else { 0.0 };
    println!(
        "RESULT impl=rust_materials rule=materials width={} height={} steps={} \
         elapsed_ms={:.3} mcells_per_s={:.2} checksum={:016x} \
         empty={} wall={} sand={} water={} gas={}",
        width, height, steps, elapsed_ms, mcells, checksum(&sim.grid),
        counts[0], counts[1], counts[2], counts[3], counts[4]
    );
}

// --- interactive mode ------------------------------------------------------

fn run_interactive(width: usize, height: usize) {
    let render_w = width * PIXEL;
    let render_h = height * PIXEL;
    let mut sim = Sim::new(width, height);
    let mut pixels = vec![0u32; render_w * render_h];
    let mut current = SAND;

    unsafe {
        if sdl::SDL_Init(sdl::INIT_VIDEO) != 0 {
            let err = std::ffi::CStr::from_ptr(sdl::SDL_GetError());
            eprintln!("SDL_Init failed: {}", err.to_string_lossy());
            return;
        }
        let title = b"Materials Sand Simulation - [1]Wall [2]Sand [3]Water [4]Gas [0]Eraser\0".as_ptr() as *const _;
        let window = sdl::SDL_CreateWindow(
            title, sdl::WINDOWPOS_UNDEFINED, sdl::WINDOWPOS_UNDEFINED,
            render_w as i32, render_h as i32, 0,
        );
        let renderer = sdl::SDL_CreateRenderer(window, -1, sdl::RENDERER_ACCELERATED);
        // Map rendering and the cursor through a fixed logical size, so painting
        // lands under the pointer even when a tiling compositor (e.g. niri)
        // resizes the window away from the requested size.
        sdl::SDL_RenderSetLogicalSize(renderer, render_w as i32, render_h as i32);
        let texture = sdl::SDL_CreateTexture(
            renderer, sdl::PIXELFORMAT_ARGB8888, sdl::TEXTUREACCESS_STREAMING,
            render_w as i32, render_h as i32,
        );

        let mut quit = false;
        let mut mouse_down = false;
        let (mut mx, mut my) = (0i32, 0i32);
        let mut ev = sdl::Event::new();
        let mut frame: u32 = 0;

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
                        k if k == 'c' as i32 => sim.grid.iter_mut().for_each(|c| *c = EMPTY),
                        _ => {}
                    },
                    _ => {}
                }
            }
            if mouse_down {
                let (mut lx, mut ly) = (0f32, 0f32);
                sdl::SDL_RenderWindowToLogical(renderer, mx, my, &mut lx, &mut ly);
                sim.paint(lx as i32, ly as i32, current, 4);
            }
            sim.update(frame);
            frame = frame.wrapping_add(1);

            for y in 0..height {
                for x in 0..width {
                    let color = COLORS[sim.grid[y * width + x] as usize];
                    for dy in 0..PIXEL {
                        for dx in 0..PIXEL {
                            pixels[(y * PIXEL + dy) * render_w + (x * PIXEL + dx)] = color;
                        }
                    }
                }
            }
            sdl::SDL_UpdateTexture(
                texture, std::ptr::null(), pixels.as_ptr() as *const _,
                (render_w * std::mem::size_of::<u32>()) as i32,
            );
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
        let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
        let width = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
        let height = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(300);
        run_bench(steps, width, height);
    } else {
        run_interactive(400, 300);
    }
}
