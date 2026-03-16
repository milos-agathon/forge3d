use super::super::*;

// P5: Screen-space GI Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "ScreenSpaceGI")]
pub struct PyScreenSpaceGI {
    manager: crate::core::screen_space_effects::ScreenSpaceEffectsManager,
    width: u32,
    height: u32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyScreenSpaceGI {
    #[new]
    #[pyo3(signature = (width=1280, height=720))]
    pub fn new(width: u32, height: u32) -> PyResult<Self> {
        let g = crate::core::gpu::ctx();
        let manager = crate::core::screen_space_effects::ScreenSpaceEffectsManager::new(
            g.device.as_ref(),
            width,
            height,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create GI manager: {e}")))?;
        Ok(Self {
            manager,
            width,
            height,
        })
    }

    /// Enable SSAO
    pub fn enable_ssao(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .enable_effect(
                g.device.as_ref(),
                crate::core::screen_space_effects::ScreenSpaceEffect::SSAO,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("enable_ssao failed: {e}")))
    }

    /// Enable SSGI
    pub fn enable_ssgi(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .enable_effect(
                g.device.as_ref(),
                crate::core::screen_space_effects::ScreenSpaceEffect::SSGI,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("enable_ssgi failed: {e}")))
    }

    /// Enable SSR
    pub fn enable_ssr(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .enable_effect(
                g.device.as_ref(),
                crate::core::screen_space_effects::ScreenSpaceEffect::SSR,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("enable_ssr failed: {e}")))
    }

    /// Disable an effect by name: "ssao", "ssgi", or "ssr"
    pub fn disable(&mut self, effect: &str) -> PyResult<()> {
        use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
        let eff = match effect.to_lowercase().as_str() {
            "ssao" => SSE::SSAO,
            "ssgi" => SSE::SSGI,
            "ssr" => SSE::SSR,
            _ => return Err(PyValueError::new_err(format!("unknown effect: {effect}"))),
        };
        self.manager.disable_effect(eff);
        Ok(())
    }

    /// Resize underlying GBuffer to a new size
    pub fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .gbuffer_mut()
            .resize(g.device.as_ref(), width, height)
            .map_err(|e| PyRuntimeError::new_err(format!("resize failed: {e}")))?;
        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Execute enabled GI passes for the current frame
    pub fn execute(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PyScreenSpaceGI.execute"),
            });
        self.manager
            .execute(g.device.as_ref(), &mut encoder, None, None)
            .map_err(|e| PyRuntimeError::new_err(format!("execute failed: {e}")))?;
        g.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}
