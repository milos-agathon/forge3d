//! Q1: Post-processing compute pipeline
//!
//! Provides a flexible effect chain manager for post-processing operations including
//! temporal effects, ping-pong resource management, and GPU compute-based filters.

use wgpu::*;

mod chain;
mod config;
mod effect;
mod resources;

pub use chain::PostFxChain;
pub use config::{PostFxConfig, PostFxResourceDesc};
pub use effect::PostFxEffect;
pub use resources::PostFxResourcePool;

/// No-op application hook for integration points in the render path.
pub fn postfx_apply_noop(_device: &Device, _encoder: &mut CommandEncoder, _input: &TextureView) {}
