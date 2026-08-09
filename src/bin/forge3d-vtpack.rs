#[cfg(not(target_arch = "wasm32"))]
include!("../bin_support/forge3d-vtpack_native.rs");

#[cfg(target_arch = "wasm32")]
fn main() {}
