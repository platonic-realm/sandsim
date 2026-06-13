// Link against the system SDL2. Kept dependency-free on purpose: the crate has
// no third-party crates so it builds fully offline with just `cargo`/`rustc`.
// Only the interactive window needs SDL2; `--bench` is pure Rust.
fn main() {
    // Prefer pkg-config's link path when available, but never fail the build on it.
    if let Ok(out) = std::process::Command::new("pkg-config")
        .args(["--libs-only-L", "sdl2"])
        .output()
    {
        if let Ok(s) = String::from_utf8(out.stdout) {
            for tok in s.split_whitespace() {
                if let Some(path) = tok.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={path}");
                }
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=SDL2");
}
