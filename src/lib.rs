// src/lib.rs
// Rust crate root for forge3d - GPU rendering library with Python bindings
// Provides SDF primitives, CSG operations, hybrid traversal, and path tracing
// RELEVANT FILES:src/sdf/mod.rs,src/path_tracing/mod.rs,python/forge3d/__init__.py

#[cfg(feature = "extension-module")]
use once_cell::sync::Lazy;
#[cfg(feature = "extension-module")]
use std::sync::Mutex;

#[cfg(feature = "extension-module")]
use shadows::state::{CpuCsmConfig, CpuCsmState};

#[cfg(feature = "extension-module")]
use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

#[cfg(feature = "extension-module")]
static GLOBAL_CSM_STATE: Lazy<Mutex<CpuCsmState>> =
    Lazy::new(|| Mutex::new(CpuCsmState::default()));

// Core modules
pub mod math {
    /// Orthonormalize a tangent `t` against normal `n` and return (tangent, bitangent).
    ///
    /// Uses simple Gram-Schmidt then computes bitangent as cross(n, t_ortho).
    pub fn orthonormalize_tangent(n: [f32; 3], t: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }
        fn norm(v: [f32; 3]) -> f32 {
            dot(v, v).sqrt()
        }
        fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        }
        fn mul(v: [f32; 3], s: f32) -> [f32; 3] {
            [v[0] * s, v[1] * s, v[2] * s]
        }
        fn normalize(v: [f32; 3]) -> [f32; 3] {
            let l = norm(v);
            if l > 0.0 {
                [v[0] / l, v[1] / l, v[2] / l]
            } else {
                v
            }
        }
        fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }

        let n_n = normalize(n);
        let t_ortho = normalize(sub(t, mul(n_n, dot(n_n, t))));
        let b = cross(n_n, t_ortho);
        (t_ortho, b)
    }
}

// Rendering modules
pub mod accel;
pub mod camera;
pub mod colormap;
pub mod context;
pub mod core;
pub mod device_caps;
pub mod error;
pub mod external_image;
pub mod formats;
pub mod gpu;
pub mod grid;
pub mod loaders;
pub mod mesh;
pub mod path_tracing;
pub mod pipeline;
pub mod renderer;
pub mod scene;
pub mod sdf; // New SDF module
pub mod shadows; // Shadow mapping implementations
pub mod terrain;
pub mod terrain_stats;
pub mod textures {}
pub mod transforms;
pub mod vector;

// Re-export commonly used types
pub use core::cloud_shadows::{
    CloudAnimationParams, CloudShadowQuality, CloudShadowRenderer, CloudShadowUniforms,
};
pub use core::clouds::{
    CloudAnimationPreset, CloudInstance, CloudParams, CloudQuality, CloudRenderMode, CloudRenderer,
    CloudUniforms,
};
pub use core::dof::{CameraDofParams, DofMethod, DofQuality, DofRenderer, DofUniforms};
pub use core::ground_plane::{
    GroundPlaneMode, GroundPlaneParams, GroundPlaneRenderer, GroundPlaneUniforms,
};
pub use core::ibl::{EnvironmentMapType, IBLMaterial, IBLQuality, IBLRenderer, IBLUniforms};
pub use core::dual_source_oit::{
    DualSourceOITMode, DualSourceOITQuality, DualSourceOITRenderer, DualSourceOITStats,
    DualSourceOITUniforms, DualSourceComposeUniforms,
};
pub use core::point_spot_lights::{
    DebugMode, Light, LightPreset, LightType, PointSpotLightRenderer, PointSpotLightUniforms,
    ShadowQuality,
};
pub use core::ltc_area_lights::{
    LTCRectAreaLightRenderer, LTCUniforms, RectAreaLight,
};
pub use core::reflections::{PlanarReflectionRenderer, ReflectionQuality};
pub use core::soft_light_radius::{
    SoftLightFalloffMode, SoftLightPreset, SoftLightRadiusRenderer, SoftLightRadiusUniforms,
};
pub use core::water_surface::{
    WaterSurfaceMode, WaterSurfaceParams, WaterSurfaceRenderer, WaterSurfaceUniforms,
};
pub use error::RenderError;
pub use path_tracing::{TracerEngine, TracerParams};
pub use sdf::{
    CsgOperation, HybridHitResult, HybridScene, SdfPrimitive, SdfPrimitiveType, SdfScene,
    SdfSceneBuilder,
};
pub use shadows::{detect_peter_panning, CascadeStatistics, CascadedShadowMaps, CsmConfig, CsmRenderer};

#[cfg(feature = "extension-module")]
#[pyfunction]
fn configure_csm(
    cascade_count: u32,
    shadow_map_size: u32,
    max_shadow_distance: f32,
    pcf_kernel_size: u32,
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
    enable_evsm: bool,
    debug_mode: u32,
) -> PyResult<()> {
    let config = CpuCsmConfig::new(
        cascade_count,
        shadow_map_size,
        max_shadow_distance,
        pcf_kernel_size,
        depth_bias,
        slope_bias,
        peter_panning_offset,
        enable_evsm,
        debug_mode,
    )
    .map_err(PyValueError::new_err)?;

    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.apply_config(config).map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_enabled(enabled: bool) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_enabled(enabled);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_light_direction(direction: (f32, f32, f32)) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_light_direction([direction.0, direction.1, direction.2]);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_pcf_kernel(kernel_size: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_pcf_kernel(kernel_size)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_bias_params(
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_bias_params(depth_bias, slope_bias, peter_panning_offset)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_debug_mode(mode: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_debug_mode(mode).map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn get_csm_cascade_info() -> PyResult<Vec<(f32, f32, f32)>> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.cascade_info())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn validate_csm_peter_panning() -> PyResult<bool> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.validate_peter_panning())
}

// PyO3 module entry point so Python can `import forge3d._forge3d`
// This must be named exactly `_forge3d` to match [tool.maturin].module-name in pyproject.toml
#[cfg(feature = "extension-module")]
#[pymodule]
fn _forge3d(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Basic metadata so users can sanity-check the native module is loaded
    m.add("__doc__", "forge3d native module")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(configure_csm, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_light_direction, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_pcf_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_bias_params, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_debug_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_csm_cascade_info, m)?)?;
    m.add_function(wrap_pyfunction!(validate_csm_peter_panning, m)?)?;

    // Add main classes
    m.add_class::<crate::scene::Scene>()?;

    Ok(())
}
