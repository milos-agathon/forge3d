use super::super::*;

/// M1: AOV (Arbitrary Output Variable) frame container
/// Holds the auxiliary render outputs captured alongside the beauty pass
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "AovFrame")]
pub struct AovFrame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Albedo texture (base color before lighting)
    albedo_texture: Option<wgpu::Texture>,
    /// Normal texture (world-space normals encoded to [0,1])
    normal_texture: Option<wgpu::Texture>,
    /// Depth texture (linear depth normalized)
    depth_texture: Option<wgpu::Texture>,
    width: u32,
    height: u32,
}

#[cfg(feature = "extension-module")]
impl AovFrame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        albedo_texture: Option<wgpu::Texture>,
        normal_texture: Option<wgpu::Texture>,
        depth_texture: Option<wgpu::Texture>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            device,
            queue,
            albedo_texture,
            normal_texture,
            depth_texture,
            width,
            height,
        }
    }

    fn read_texture_bytes(&self, texture: &wgpu::Texture) -> anyhow::Result<Vec<u8>> {
        read_texture_tight(
            &self.device,
            &self.queue,
            texture,
            (self.width, self.height),
            wgpu::TextureFormat::Rgba8Unorm,
        )
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl AovFrame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "AovFrame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[getter]
    fn has_albedo(&self) -> bool {
        self.albedo_texture.is_some()
    }

    #[getter]
    fn has_normal(&self) -> bool {
        self.normal_texture.is_some()
    }

    #[getter]
    fn has_depth(&self) -> bool {
        self.depth_texture.is_some()
    }

    /// Save albedo AOV to PNG file
    fn save_albedo(&self, path: &str) -> PyResult<()> {
        let texture = self
            .albedo_texture
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Albedo AOV not available"))?;
        let mut data = self
            .read_texture_bytes(texture)
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        image_write::write_png_rgba8(Path::new(path), &data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))?;
        Ok(())
    }

    /// Save normal AOV to PNG file
    fn save_normal(&self, path: &str) -> PyResult<()> {
        let texture = self
            .normal_texture
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Normal AOV not available"))?;
        let mut data = self
            .read_texture_bytes(texture)
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        image_write::write_png_rgba8(Path::new(path), &data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))?;
        Ok(())
    }

    /// Save depth AOV to PNG file
    fn save_depth(&self, path: &str) -> PyResult<()> {
        let texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Depth AOV not available"))?;
        let mut data = self
            .read_texture_bytes(texture)
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        image_write::write_png_rgba8(Path::new(path), &data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))?;
        Ok(())
    }

    /// Save all AOVs to directory with standard naming
    fn save_all(&self, output_dir: &str, base_name: &str) -> PyResult<()> {
        let dir = Path::new(output_dir);
        std::fs::create_dir_all(dir)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create directory: {e}")))?;

        if self.albedo_texture.is_some() {
            let path = dir.join(format!("{}_albedo.png", base_name));
            self.save_albedo(path.to_str().unwrap())?;
        }
        if self.normal_texture.is_some() {
            let path = dir.join(format!("{}_normal.png", base_name));
            self.save_normal(path.to_str().unwrap())?;
        }
        if self.depth_texture.is_some() {
            let path = dir.join(format!("{}_depth.png", base_name));
            self.save_depth(path.to_str().unwrap())?;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "AovFrame(width={}, height={}, albedo={}, normal={}, depth={})",
            self.width,
            self.height,
            self.albedo_texture.is_some(),
            self.normal_texture.is_some(),
            self.depth_texture.is_some()
        )
    }
}
