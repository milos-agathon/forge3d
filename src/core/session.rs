// src/session.rs
// PyO3 session wrapper managing shared GPU context
// Exists to expose a simple GPU session object to Python callers
// RELEVANT FILES: src/gpu.rs, python/forge3d/__init__.py, tests/test_session.py, python/forge3d/terrain_params.py
#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyNotImplementedError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use std::sync::Arc;

/// GPU session managing device lifecycle.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Session")]
pub struct Session {
    #[allow(dead_code)]
    pub(crate) device: Arc<wgpu::Device>,
    #[allow(dead_code)]
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) adapter: Arc<wgpu::Adapter>,
    #[allow(dead_code)]
    pub(crate) instance: Arc<wgpu::Instance>,
    #[allow(dead_code)]
    pub(crate) surface: Option<Arc<wgpu::Surface<'static>>>,
    window: bool,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl Session {
    /// Create a new GPU session.
    ///
    /// Args:
    ///     window: If True, create a windowed session (not implemented yet)
    ///     backend: Optional backend selection ("vulkan", "metal", "dx12", "gl")
    #[new]
    #[pyo3(signature = (window=false, backend=None))]
    pub fn new(window: bool, backend: Option<&str>) -> PyResult<Self> {
        let ctx = super::gpu::ctx();

        if window {
            return Err(PyNotImplementedError::new_err(
                "Windowed sessions not yet supported. Use window=False for offscreen rendering.",
            ));
        }

        if let Some(name) = backend {
            log::warn!(
                "Backend selection ({}) not yet implemented; using global context backend",
                name
            );
        }

        Ok(Self {
            device: Arc::clone(&ctx.device),
            queue: Arc::clone(&ctx.queue),
            adapter: Arc::clone(&ctx.adapter),
            instance: Arc::clone(&ctx.instance),
            surface: None,
            window,
        })
    }

    /// Get session information string.
    pub fn info(&self) -> PyResult<String> {
        let adapter_info = self.adapter.get_info();
        Ok(format!(
            "Session(backend={:?}, device='{}', type={:?}, window={})",
            adapter_info.backend, adapter_info.name, adapter_info.device_type, self.window
        ))
    }

    /// Get adapter name.
    #[getter]
    pub fn adapter_name(&self) -> PyResult<String> {
        Ok(self.adapter.get_info().name.clone())
    }

    /// Get backend name.
    #[getter]
    pub fn backend(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.adapter.get_info().backend))
    }

    /// Get device type.
    #[getter]
    pub fn device_type(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.adapter.get_info().device_type))
    }

    /// Python repr for debugging.
    fn __repr__(&self) -> PyResult<String> {
        self.info()
    }
}
