use super::super::*;

#[cfg(feature = "extension-module")]
use serde_json::{Map, Value};
#[cfg(feature = "extension-module")]
use sha2::{Digest, Sha256};

#[cfg(feature = "extension-module")]
use crate::terrain::frame_compiler::{CompiledFrame as NativeCompiledFrame, FrameCompiler};

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "CompiledFrame")]
pub(crate) struct CompiledFrame {
    pub(crate) inner: NativeCompiledFrame,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CompiledFrame {
    #[getter]
    fn frame_index(&self) -> u64 {
        self.inner.frame_index()
    }

    #[getter]
    fn base_seed(&self) -> u64 {
        self.inner.base_seed()
    }

    #[getter]
    fn frame_seed(&self) -> u64 {
        self.inner.frame_seed()
    }

    #[getter]
    fn samples(&self) -> u32 {
        self.inner.samples()
    }

    #[getter]
    fn camera_hash(&self) -> PyResult<String> {
        self.payload_string("camera_hash")
    }

    #[getter]
    fn scene_hash(&self) -> PyResult<String> {
        self.payload_string("scene_hash")
    }

    #[getter]
    fn label_set_hash(&self) -> PyResult<String> {
        self.payload_string("label_set_hash")
    }

    #[getter]
    fn residency_hash(&self) -> PyResult<String> {
        self.payload_string("residency_hash")
    }

    #[getter]
    fn lod_hash(&self) -> PyResult<String> {
        self.payload_string("lod_hash")
    }

    #[getter]
    fn engine_revision(&self) -> PyResult<String> {
        self.payload_string("engine_revision")
    }

    fn to_json(&self) -> String {
        self.inner.canonical_json().to_string()
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = NativeCompiledFrame::from_canonical(json).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
}

#[cfg(feature = "extension-module")]
impl CompiledFrame {
    fn payload_string(&self, field: &str) -> PyResult<String> {
        self.inner
            .payload()
            .get(field)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| PyValueError::new_err(format!("compiled frame missing '{field}'")))
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn frame_seed(base_seed: u64, frame_index: u64) -> u64 {
    crate::terrain::accumulation::frame_seed(base_seed, frame_index)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (frame_index, base_seed, samples, camera_json, scene_json))]
pub(crate) fn compile_frame(
    frame_index: u64,
    base_seed: u64,
    samples: u32,
    camera_json: &str,
    scene_json: &str,
) -> PyResult<CompiledFrame> {
    let inner = FrameCompiler::compile(frame_index, base_seed, samples, camera_json, scene_json)
        .map_err(PyValueError::new_err)?;
    Ok(CompiledFrame { inner })
}

/// Hash an already-rendered RGBA frame into its CHRONOS provenance record.
///
/// Outside CENSOR's render-certificate scope: this executes no rendering; it
/// binds pixels produced by a certified render (`MapScene.render`) to the
/// compiled frame that drove them.
#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn render_compiled_frame(
    compiled_frame: PyRef<'_, CompiledFrame>,
    rgba: numpy::PyReadonlyArray3<'_, u8>,
) -> PyResult<String> {
    let array = rgba.as_array();
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != 4 {
        return Err(PyValueError::new_err(format!(
            "render_compiled_frame expects a (H, W, 4) uint8 array, got shape {shape:?}"
        )));
    }
    let bytes = array.as_slice().ok_or_else(|| {
        PyValueError::new_err("render_compiled_frame requires a C-contiguous (H, W, 4) uint8 array")
    })?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let pixel_hash = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    let payload = compiled_frame.inner.payload();
    let mut out = Map::new();
    for field in [
        "frame_index",
        "base_seed",
        "frame_seed",
        "samples",
        "camera_hash",
        "scene_hash",
        "label_set_hash",
        "residency_hash",
        "lod_hash",
        "engine_revision",
    ] {
        let value = payload
            .get(field)
            .ok_or_else(|| PyValueError::new_err(format!("compiled frame missing '{field}'")))?;
        out.insert(field.to_string(), value.clone());
    }
    out.insert("pixel_hash".to_string(), Value::from(pixel_hash));
    out.insert("compiled_frame".to_string(), payload.clone());
    let normalized = crate::terrain::frame_compiler::normalize_value(&Value::Object(out))
        .map_err(PyValueError::new_err)?;
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|e| PyValueError::new_err(format!("provenance serialization failed: {e}")))?;
    String::from_utf8(bytes)
        .map_err(|e| PyValueError::new_err(format!("provenance JSON is not UTF-8: {e}")))
}
