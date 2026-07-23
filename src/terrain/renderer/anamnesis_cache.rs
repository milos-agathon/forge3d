//! Conservative ANAMNESIS boundary for the native terrain renderer.
//!
//! Native terminal-frame caching is intentionally disabled until the
//! framegraph-owned scheduler can restore every declared intermediate
//! resource. The former implementation keyed a pickle of Python parameters
//! and a subset of leaf resources, which omitted live virtual-texture,
//! streaming, scatter, overlay-uniform, and pipeline state. Serving a hit from
//! that incomplete key was unsound. A supplied cache path therefore causes a
//! conservative recompute and an empty report; `cache=None` remains bit-for-bit
//! identical.

use super::*;
use crate::core::anamnesis::CacheReport;
use crate::terrain::render_params;
use pyo3::types::PyAnyMethods;

pub(super) struct PreparedFrameCache;

impl PreparedFrameCache {
    pub(super) fn execute<F>(
        &self,
        _device: std::sync::Arc<wgpu::Device>,
        _queue: std::sync::Arc<wgpu::Queue>,
        encode: F,
    ) -> PyResult<(crate::Frame, CacheReport)>
    where
        F: FnOnce() -> PyResult<crate::Frame>,
    {
        Ok((encode()?, CacheReport::default()))
    }
}

fn certificate_enabled(certificate: Option<&Bound<'_, PyAny>>) -> bool {
    let Some(value) = certificate else {
        return false;
    };
    if value.is_none() {
        return false;
    }
    !matches!(value.extract::<bool>(), Ok(false))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_frame_cache(
    _py: Python<'_>,
    _scene: &TerrainScene,
    cache: Option<&Bound<'_, PyAny>>,
    certificate: Option<&Bound<'_, PyAny>>,
    _material_set: &crate::render::material_set::MaterialSet,
    _env_maps: &crate::lighting::ibl_wrapper::IBL,
    _params: &render_params::TerrainRenderParams,
    _heightmap: &PyReadonlyArray2<'_, f32>,
    _water_mask: Option<&PyReadonlyArray2<'_, f32>>,
    _time_seconds: f32,
) -> PyResult<Option<PreparedFrameCache>> {
    if cache.is_none() || certificate_enabled(certificate) {
        return Ok(None);
    }
    // Soundness boundary: a false miss is acceptable; a stale hit is not.
    Ok(None)
}
