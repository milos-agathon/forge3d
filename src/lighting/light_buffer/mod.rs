// src/lighting/light_buffer/mod.rs
// P1: Light buffer management with triple-buffering for multi-light support
// SSBO storage buffer layout (std430) for efficient GPU upload

mod r2;
mod types;
mod creation;
mod update;
mod frame;

#[cfg(test)]
mod tests;

pub use types::*;
