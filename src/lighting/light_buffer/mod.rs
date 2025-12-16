// src/lighting/light_buffer/mod.rs
// P1: Light buffer management with triple-buffering for multi-light support
// SSBO storage buffer layout (std430) for efficient GPU upload

mod creation;
mod frame;
mod r2;
mod types;
mod update;

#[cfg(test)]
mod tests;

pub use types::*;
