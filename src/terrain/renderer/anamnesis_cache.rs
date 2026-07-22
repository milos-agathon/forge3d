//! Sound final-resource caching for the native one-shot terrain renderer.
//!
//! This boundary is intentionally conservative. It is enabled only for the
//! stateless one-shot path and hashes the complete Python parameter object,
//! exact array bytes, material and texture contents, decoded HDR pixels,
//! renderer light overrides, negotiated capabilities, backend/compiler state,
//! and the engine/WGSL build fingerprint. Any input that cannot be serialized
//! causes a false miss and leaves rendering unchanged.

use super::*;
use crate::core::anamnesis::{
    leaf_key, CacheReport, ContentStore, EngineFingerprint, PassKey, PassRequest, Scheduler,
};
use crate::terrain::render_params;
use numpy::PyUntypedArrayMethods;
use pyo3::types::PyAnyMethods;
use std::io;
use std::path::PathBuf;

const DEFAULT_MAX_BYTES: u64 = 10 * 1024 * 1024 * 1024;

pub(super) struct PreparedFrameCache {
    store: ContentStore,
    pipeline: Vec<u8>,
    uniforms: Vec<u8>,
    input_keys: Vec<PassKey>,
    capability: Vec<u8>,
    engine: Vec<u8>,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl PreparedFrameCache {
    /// Execute the native scheduler. A hit returns the verified blob without
    /// invoking `encode`; a miss invokes the real terrain encoder once, reads
    /// back its final resource, and stores it before rehydrating the identical
    /// tracked GPU texture used by downstream callers.
    pub(super) fn execute<F>(
        &self,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        encode: F,
    ) -> PyResult<(crate::Frame, CacheReport)>
    where
        F: FnOnce() -> PyResult<crate::Frame>,
    {
        let mut scheduler = Scheduler::new(self.store.clone());
        let request = PassRequest {
            label: "terrain.one_shot.final",
            pipeline_descriptor_bytes: &self.pipeline,
            uniform_bytes: &self.uniforms,
            input_keys: &self.input_keys,
            capability_fingerprint_bytes: &self.capability,
            engine_fingerprint_bytes: &self.engine,
            estimated_wall_ms: 0.0,
        };
        let (_, blob) = scheduler
            .execute(request, || {
                let frame = encode()
                    .map_err(|error| io::Error::new(io::ErrorKind::Other, error.to_string()))?;
                frame
                    .read_tight_bytes()
                    .map_err(|error| io::Error::new(io::ErrorKind::Other, error.to_string()))
            })
            .map_err(|error| {
                PyRuntimeError::new_err(format!("ANAMNESIS scheduler failed: {error}"))
            })?;
        let frame = crate::Frame::from_tight_bytes(
            device,
            queue,
            self.width,
            self.height,
            self.format,
            &blob,
        )
        .map_err(|error| {
            PyRuntimeError::new_err(format!(
                "ANAMNESIS cached frame rehydrate failed: {error:#}"
            ))
        })?;
        Ok((frame, scheduler.into_report()))
    }
}

fn add_segment(output: &mut Vec<u8>, label: &[u8], bytes: &[u8]) {
    output.extend_from_slice(&(label.len() as u64).to_le_bytes());
    output.extend_from_slice(label);
    output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(bytes);
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

fn cache_path(py: Python<'_>, cache: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    let value = py.import_bound("os")?.call_method1("fspath", (cache,))?;
    value.extract::<PathBuf>()
}

fn pickle_params(py: Python<'_>, params: &render_params::TerrainRenderParams) -> Option<Vec<u8>> {
    let pickle = py.import_bound("pickle").ok()?;
    pickle
        .call_method1("dumps", (params.python_object.bind(py), 5u8))
        .ok()?
        .extract::<Vec<u8>>()
        .ok()
}

fn material_bytes(material_set: &crate::render::material_set::MaterialSet) -> Vec<u8> {
    let mut bytes = Vec::new();
    add_segment(
        &mut bytes,
        b"materials",
        bytemuck::cast_slice(material_set.materials()),
    );
    add_segment(
        &mut bytes,
        b"triplanar_scale",
        &material_set.triplanar_scale.to_bits().to_le_bytes(),
    );
    add_segment(
        &mut bytes,
        b"normal_strength",
        &material_set.normal_strength.to_bits().to_le_bytes(),
    );
    add_segment(
        &mut bytes,
        b"blend_sharpness",
        &material_set.blend_sharpness.to_bits().to_le_bytes(),
    );
    for path in &material_set._texture_paths {
        match path {
            None => add_segment(&mut bytes, b"texture", b"none"),
            Some(path) => {
                add_segment(&mut bytes, b"texture_path", path.as_bytes());
                match std::fs::read(path) {
                    Ok(content) => add_segment(&mut bytes, b"texture_content", &content),
                    Err(_) => add_segment(&mut bytes, b"texture_content", b"missing"),
                }
            }
        }
    }
    bytes
}

fn ibl_bytes(env_maps: &crate::lighting::ibl_wrapper::IBL) -> Vec<u8> {
    let mut bytes = Vec::new();
    add_segment(
        &mut bytes,
        b"intensity",
        &env_maps.intensity.to_bits().to_le_bytes(),
    );
    add_segment(
        &mut bytes,
        b"rotation_deg",
        &env_maps.rotation_deg.to_bits().to_le_bytes(),
    );
    add_segment(
        &mut bytes,
        b"quality",
        format!("{:?}", env_maps.quality).as_bytes(),
    );
    add_segment(
        &mut bytes,
        b"base_resolution",
        &env_maps.base_resolution.to_le_bytes(),
    );
    match env_maps.hdr_image.as_ref() {
        Some(image) => {
            add_segment(&mut bytes, b"hdr_width", &image.width.to_le_bytes());
            add_segment(&mut bytes, b"hdr_height", &image.height.to_le_bytes());
            add_segment(&mut bytes, b"hdr_pixels", bytemuck::cast_slice(&image.data));
        }
        None => add_segment(&mut bytes, b"hdr_pixels", b"none"),
    }
    bytes
}

fn capability_bytes(scene: &TerrainScene) -> Vec<u8> {
    let info = scene.adapter.get_info();
    let mut bytes = Vec::new();
    add_segment(
        &mut bytes,
        b"backend",
        format!("{:?}", info.backend)
            .to_ascii_lowercase()
            .as_bytes(),
    );
    add_segment(
        &mut bytes,
        b"device_features",
        format!("{:?}", scene.device.features()).as_bytes(),
    );
    add_segment(
        &mut bytes,
        b"device_limits",
        format!("{:?}", scene.device.limits()).as_bytes(),
    );
    add_segment(
        &mut bytes,
        b"dx12_compiler",
        std::env::var("WGPU_DX12_COMPILER")
            .unwrap_or_else(|_| "default".to_string())
            .as_bytes(),
    );
    bytes
}

fn override_light_bytes(scene: &TerrainScene) -> Option<Vec<u8>> {
    let guard = scene.light_override.lock().ok()?;
    Some(match guard.as_ref() {
        Some(lights) => bytemuck::cast_slice(lights).to_vec(),
        None => b"parameter-lighting".to_vec(),
    })
}

pub(super) fn prepare_frame_cache(
    py: Python<'_>,
    scene: &TerrainScene,
    cache: Option<&Bound<'_, PyAny>>,
    certificate: Option<&Bound<'_, PyAny>>,
    material_set: &crate::render::material_set::MaterialSet,
    env_maps: &crate::lighting::ibl_wrapper::IBL,
    params: &render_params::TerrainRenderParams,
    heightmap: &PyReadonlyArray2<'_, f32>,
    water_mask: Option<&PyReadonlyArray2<'_, f32>>,
    time_seconds: f32,
) -> PyResult<Option<PreparedFrameCache>> {
    let Some(cache) = cache else {
        return Ok(None);
    };
    // A certificate attests live execution. Reusing a prior frame must never
    // masquerade as a newly executed certified render.
    if certificate_enabled(certificate) {
        return Ok(None);
    }
    let Some(param_bytes) = pickle_params(py, params) else {
        return Ok(None);
    };
    let Ok(height_bytes) = heightmap.as_slice() else {
        return Ok(None);
    };
    let water_bytes = match water_mask {
        Some(mask) => match mask.as_slice() {
            Ok(bytes) => Some(bytemuck::cast_slice(bytes)),
            Err(_) => return Ok(None),
        },
        None => None,
    };
    let Some(light_bytes) = override_light_bytes(scene) else {
        return Ok(None);
    };

    let dimensions = heightmap.shape();
    let mut height_content = Vec::new();
    add_segment(
        &mut height_content,
        b"shape",
        format!("{:?}", dimensions).as_bytes(),
    );
    add_segment(
        &mut height_content,
        b"f32-le",
        bytemuck::cast_slice(height_bytes),
    );
    let mut water_content = Vec::new();
    match (water_mask, water_bytes) {
        (Some(mask), Some(bytes)) => {
            add_segment(
                &mut water_content,
                b"shape",
                format!("{:?}", mask.shape()).as_bytes(),
            );
            add_segment(&mut water_content, b"f32-le", bytes);
        }
        _ => add_segment(&mut water_content, b"water", b"none"),
    }

    let input_keys = vec![
        leaf_key(&height_content),
        leaf_key(&water_content),
        leaf_key(&material_bytes(material_set)),
        leaf_key(&ibl_bytes(env_maps)),
        leaf_key(&light_bytes),
    ];
    let capability = capability_bytes(scene);
    let engine = EngineFingerprint::current().canonical_bytes();
    let mut pipeline = Vec::new();
    add_segment(&mut pipeline, b"label", b"terrain.one_shot.final");
    add_segment(
        &mut pipeline,
        b"format",
        format!("{:?}", scene.color_format).as_bytes(),
    );
    add_segment(
        &mut pipeline,
        b"wgsl_tree",
        EngineFingerprint::current().wgsl_tree_sha256.as_bytes(),
    );
    let mut uniforms = param_bytes;
    add_segment(
        &mut uniforms,
        b"time_seconds",
        &time_seconds.to_bits().to_le_bytes(),
    );
    Ok(Some(PreparedFrameCache {
        store: ContentStore::new(cache_path(py, cache)?, DEFAULT_MAX_BYTES, true).map_err(
            |error| PyRuntimeError::new_err(format!("ANAMNESIS cache open failed: {error}")),
        )?,
        pipeline,
        uniforms,
        input_keys,
        capability,
        engine,
        width: params.size_px.0,
        height: params.size_px.1,
        format: scene.color_format,
    }))
}
