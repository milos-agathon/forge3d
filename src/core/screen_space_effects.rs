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
use wgpu::*;

const SSAO_SHADER_SRC: &str = concat!(
    include_str!("../shaders/ssao/common.wgsl"),
    "\n",
    include_str!("../shaders/ssao/ssao.wgsl")
);
pub(crate) const GTAO_SHADER_SRC: &str = concat!(
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

pub(crate) mod hzb;
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

pub(crate) use manager::PreparedScreenSpaceTargets;

/// Shared-frame reconstruction supports both perspective and orthographic depth.
/// Called only by CONCORDIA factories; authoritative WGSL and legacy pipelines stay unchanged.
pub(crate) fn concordia_reconstruction(
    source: &str,
    camera: &str,
    function: &str,
) -> Result<String, crate::core::error::RenderError> {
    let mut source = source.replace("\r\n", "\n");
    let start = source.find(&format!("fn {function}(")).ok_or_else(|| {
        crate::core::error::RenderError::Render("shared reconstruction anchor missing".into())
    })?;
    let end = source[start..]
        .find("\n}")
        .map(|end| start + end + 2)
        .ok_or_else(|| {
            crate::core::error::RenderError::Render("shared reconstruction end missing".into())
        })?;
    source.replace_range(
        start..end,
        &format!(
            r#"fn {function}(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {{
        let ndc = uv*vec2<f32>(2.0,-2.0)+vec2<f32>(-1.0,1.0);
        let a_h = {camera}.inv_proj_matrix*vec4<f32>(ndc,0.0,1.0);
        let b_h = {camera}.inv_proj_matrix*vec4<f32>(ndc,1.0,1.0);
        let a = a_h.xyz/a_h.w; let b = b_h.xyz/b_h.w;
        return mix(a,b,(linear_depth+a.z)/(a.z-b.z));
    }}"#
        ),
    );
    Ok(source)
}
