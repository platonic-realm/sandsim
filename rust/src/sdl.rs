//! Minimal SDL2 FFI shim shared by the sand and materials binaries.
//! Standard-library only; links the system SDL2 (see build.rs).
#![allow(dead_code)] // each binary uses a different subset

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

// SDL scancodes (indices into SDL_GetKeyboardState) for camera panning.
pub const SCANCODE_A: usize = 4;
pub const SCANCODE_D: usize = 7;
pub const SCANCODE_S: usize = 22;
pub const SCANCODE_W: usize = 26;
pub const SCANCODE_RIGHT: usize = 79;
pub const SCANCODE_LEFT: usize = 80;
pub const SCANCODE_DOWN: usize = 81;
pub const SCANCODE_UP: usize = 82;

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
    pub fn SDL_RenderSetLogicalSize(r: *mut Renderer, w: c_int, h: c_int) -> c_int;
    pub fn SDL_RenderWindowToLogical(r: *mut Renderer, window_x: c_int, window_y: c_int,
                                     logical_x: *mut f32, logical_y: *mut f32);
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
    pub fn SDL_GetKeyboardState(numkeys: *mut c_int) -> *const u8;
    pub fn SDL_Delay(ms: u32);
    pub fn SDL_DestroyTexture(t: *mut Texture);
    pub fn SDL_DestroyRenderer(r: *mut Renderer);
    pub fn SDL_DestroyWindow(w: *mut Window);
}
