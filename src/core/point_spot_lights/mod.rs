// B13: Point & Spot Lights (Realtime) - Core Rust implementation
// Provides point and spot light management with shadows and penumbra shaping

mod creation;
mod draw;
mod management;
mod presets;
mod structs;
mod types;

pub use presets::*;
pub use structs::*;
pub use types::*;
