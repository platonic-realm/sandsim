//! sandsim - Rust implementation
//!
//! Canonical "scalar" falling-sand rule, a direct port of
//! cpp/sandsim_scalar_sb.cpp. Standard-library only: the interactive window
//! talks to system SDL2 through a tiny hand-written FFI shim (see `sdl`), and
//! `--bench` is pure safe Rust.
//!
//! Modes:
//!   (default)                 SDL2 window, 400x300 grid rendered 2x (800x600).
//!   --bench [steps] [w] [h]   headless: deterministic seed, time the update
//!                             loop, print one RESULT line. Its checksum matches
//!                             every other scalar-rule implementation.

use std::time::Instant;

const PIXEL_SIZE: usize = 2;

// ---------------------------------------------------------------------------
// Shared, language-independent helpers (must match the other implementations).
// ---------------------------------------------------------------------------

/// Deterministic per-cell seed (~30% sand), u32 wraparound arithmetic.
#[inline]
fn seed_cell(x: i32, y: i32) -> u8 {
    let mut h = (x as u32)
        .wrapping_mul(374761393)
        .wrapping_add((y as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    if h % 100 < 30 { 1 } else { 0 }
}

/// FNV-1a over the grid, row-major, u64 wraparound.
fn checksum(grid: &[u8]) -> u64 {
    let mut c: u64 = 14695981039346656037;
    for &cell in grid {
        c = (c ^ cell as u64).wrapping_mul(1099511628211);
    }
    c
}

/// One update step: sand falls down, else down-left, else down-right.
fn update(buf: &mut [u8], width: usize, height: usize) {
    for y in (0..height - 1).rev() {
        for x in 0..width {
            let i = y * width + x;
            if buf[i] == 1 {
                let below = (y + 1) * width + x;
                if buf[below] == 0 {
                    buf[below] = 1;
                    buf[i] = 0;
                } else if x > 0 && buf[below - 1] == 0 {
                    buf[below - 1] = 1;
                    buf[i] = 0;
                } else if x < width - 1 && buf[below + 1] == 0 {
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

fn run_bench(steps: u64, width: usize, height: usize) {
    let mut buf = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            buf[y * width + x] = seed_cell(x as i32, y as i32);
        }
    }

    let start = Instant::now();
    for _ in 0..steps {
        update(&mut buf, width, height);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let cells = width as f64 * height as f64 * steps as f64;
    let mcells = if elapsed_ms > 0.0 {
        cells / (elapsed_ms / 1000.0) / 1e6
    } else {
        0.0
    };

    println!(
        "RESULT impl=rust rule=scalar width={} height={} steps={} \
         elapsed_ms={:.3} mcells_per_s={:.2} checksum={:016x}",
        width,
        height,
        steps,
        elapsed_ms,
        mcells,
        checksum(&buf)
    );
}

// ---------------------------------------------------------------------------
// Minimal SDL2 FFI (only what the interactive window needs).
// ---------------------------------------------------------------------------

mod sdl {
    use std::os::raw::{c_char, c_int, c_void};

    pub const INIT_VIDEO: u32 = 0x0000_0020;
    pub const WINDOWPOS_UNDEFINED: c_int = 0x1FFF_0000u32 as c_int;
    pub const RENDERER_ACCELERATED: u32 = 0x0000_0002;
    pub const PIXELFORMAT_ARGB8888: u32 = 0x1636_2004;
    pub const TEXTUREACCESS_STREAMING: c_int = 1;

    pub const QUIT: u32 = 0x100;
    pub const KEYDOWN: u32 = 0x300;
    pub const MOUSEMOTION: u32 = 0x400;
    pub const MOUSEBUTTONDOWN: u32 = 0x401;
    pub const MOUSEBUTTONUP: u32 = 0x402;

    pub const KEY_C: i32 = 'c' as i32;
    pub const KEY_R: i32 = 'r' as i32;

    pub enum Window {}
    pub enum Renderer {}
    pub enum Texture {}

    /// SDL_Event is a 56-byte union; we treat it as raw bytes and decode the
    /// few fields we care about (the 4-byte type, and keysym.sym at offset 20).
    #[repr(C)]
    pub struct Event {
        pub bytes: [u8; 56],
    }
    impl Event {
        pub fn new() -> Self {
            Event { bytes: [0u8; 56] }
        }
        pub fn kind(&self) -> u32 {
            u32::from_ne_bytes([self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]])
        }
        pub fn keysym(&self) -> i32 {
            i32::from_ne_bytes([self.bytes[20], self.bytes[21], self.bytes[22], self.bytes[23]])
        }
    }

    #[link(name = "SDL2")]
    extern "C" {
        pub fn SDL_Init(flags: u32) -> c_int;
        pub fn SDL_Quit();
        pub fn SDL_GetError() -> *const c_char;
        pub fn SDL_CreateWindow(
            title: *const c_char,
            x: c_int,
            y: c_int,
            w: c_int,
            h: c_int,
            flags: u32,
        ) -> *mut Window;
        pub fn SDL_CreateRenderer(w: *mut Window, index: c_int, flags: u32) -> *mut Renderer;
        pub fn SDL_CreateTexture(
            r: *mut Renderer,
            format: u32,
            access: c_int,
            w: c_int,
            h: c_int,
        ) -> *mut Texture;
        pub fn SDL_UpdateTexture(
            t: *mut Texture,
            rect: *const c_void,
            pixels: *const c_void,
            pitch: c_int,
        ) -> c_int;
        pub fn SDL_RenderClear(r: *mut Renderer) -> c_int;
        pub fn SDL_RenderCopy(
            r: *mut Renderer,
            t: *mut Texture,
            src: *const c_void,
            dst: *const c_void,
        ) -> c_int;
        pub fn SDL_RenderPresent(r: *mut Renderer);
        pub fn SDL_PollEvent(e: *mut Event) -> c_int;
        pub fn SDL_GetMouseState(x: *mut c_int, y: *mut c_int) -> u32;
        pub fn SDL_Delay(ms: u32);
        pub fn SDL_DestroyTexture(t: *mut Texture);
        pub fn SDL_DestroyRenderer(r: *mut Renderer);
        pub fn SDL_DestroyWindow(w: *mut Window);
    }
}

/// Tiny LCG so the interactive `R` (randomize) needs no external rng crate.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

fn add_sand(buf: &mut [u8], width: usize, height: usize, px: i32, py: i32, radius: i32) {
    let x = px / PIXEL_SIZE as i32;
    let y = py / PIXEL_SIZE as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0
                && nx < width as i32
                && ny >= 0
                && ny < height as i32
                && dx * dx + dy * dy <= radius * radius
            {
                buf[ny as usize * width + nx as usize] = 1;
            }
        }
    }
}

fn run_interactive(width: usize, height: usize) {
    let render_w = width * PIXEL_SIZE;
    let render_h = height * PIXEL_SIZE;
    let mut buf = vec![0u8; width * height];
    let mut pixels = vec![0u32; render_w * render_h];
    let mut rng = Lcg(0x9E3779B97F4A7C15);

    unsafe {
        if sdl::SDL_Init(sdl::INIT_VIDEO) != 0 {
            let err = std::ffi::CStr::from_ptr(sdl::SDL_GetError());
            eprintln!("SDL_Init failed: {}", err.to_string_lossy());
            return;
        }
        let title = b"Rust Sand Simulation\0".as_ptr() as *const _;
        let window = sdl::SDL_CreateWindow(
            title,
            sdl::WINDOWPOS_UNDEFINED,
            sdl::WINDOWPOS_UNDEFINED,
            render_w as i32,
            render_h as i32,
            0,
        );
        let renderer = sdl::SDL_CreateRenderer(window, -1, sdl::RENDERER_ACCELERATED);
        let texture = sdl::SDL_CreateTexture(
            renderer,
            sdl::PIXELFORMAT_ARGB8888,
            sdl::TEXTUREACCESS_STREAMING,
            render_w as i32,
            render_h as i32,
        );

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
                    sdl::MOUSEMOTION => {
                        sdl::SDL_GetMouseState(&mut mx, &mut my);
                    }
                    sdl::KEYDOWN => match ev.keysym() {
                        sdl::KEY_C => buf.iter_mut().for_each(|c| *c = 0),
                        sdl::KEY_R => {
                            for c in buf.iter_mut() {
                                *c = if rng.next_f32() < 0.3 { 1 } else { 0 };
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            if mouse_down {
                add_sand(&mut buf, width, height, mx, my, 5);
            }
            update(&mut buf, width, height);

            // Render: 2x2 pixel blocks, yellow sand on black.
            for y in 0..height {
                for x in 0..width {
                    let color = if buf[y * width + x] != 0 { 0xFFFFFF00u32 } else { 0xFF000000u32 };
                    for dy in 0..PIXEL_SIZE {
                        for dx in 0..PIXEL_SIZE {
                            let rx = x * PIXEL_SIZE + dx;
                            let ry = y * PIXEL_SIZE + dy;
                            pixels[ry * render_w + rx] = color;
                        }
                    }
                }
            }
            sdl::SDL_UpdateTexture(
                texture,
                std::ptr::null(),
                pixels.as_ptr() as *const _,
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
