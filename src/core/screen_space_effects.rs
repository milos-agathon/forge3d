//! P5: Screen-space effects system (SSAO/GTAO, SSGI, SSR)
//!
//! Provides GPU-accelerated screen-space techniques for ambient occlusion,
//! global illumination, and reflections.

use super::error::{RenderError, RenderResult};
use crate::core::gbuffer::{GBuffer, GBufferConfig};
use crate::core::gpu_timing::GpuTimingManager;
use futures_intrusive::channel::shared::oneshot_channel;
use pollster::block_on;
use std::mem::size_of;
use std::time::Instant;
use wgpu::util::DeviceExt;
use wgpu::*;

const SSAO_SHADER_SRC: &str = concat!(
    include_str!("../shaders/ssao/common.wgsl"),
    "\n",
    include_str!("../shaders/ssao/ssao.wgsl")
);
const GTAO_SHADER_SRC: &str = concat!(
    include_str!("../shaders/ssao/common.wgsl"),
    "\n",
    include_str!("../shaders/ssao/gtao.wgsl")
);
const SSAO_COMPOSITE_SHADER_SRC: &str = include_str!("../shaders/ssao/composite.wgsl");

/// Screen-space effect type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenSpaceEffect {
    /// Screen-Space Ambient Occlusion / Ground-Truth Ambient Occlusion
    SSAO,
    /// Screen-Space Global Illumination
    SSGI,
    /// Screen-Space Reflections
    SSR,
}

mod hzb;
mod manager;
mod settings;
mod ssao;
mod ssgi;
mod ssr;

pub use hzb::HzbPyramid;
pub use manager::ScreenSpaceEffectsManager;
pub use settings::{CameraParams, SsaoSettings, SsgiSettings, SsrSettings, SsrStats};
pub use ssao::SsaoRenderer;
pub use ssgi::SsgiRenderer;
pub use ssr::SsrRenderer;
