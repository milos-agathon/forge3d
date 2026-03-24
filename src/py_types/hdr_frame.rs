use super::super::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "HdrFrame")]
pub struct HdrFrame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
}

#[cfg(feature = "extension-module")]
impl HdrFrame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            device,
            queue,
            texture,
            width,
            height,
        }
    }

    pub(crate) fn read_rgba_f32(&self) -> anyhow::Result<Vec<f32>> {
        crate::core::hdr::read_hdr_texture(
            &self.device,
            &self.queue,
            &self.texture,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba16Float,
        )
        .map_err(anyhow::Error::msg)
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub(crate) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HdrFrame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "HdrFrame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn to_numpy_f32<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<f32>> {
        let data = py
            .allow_threads(|| self.read_rgba_f32())
            .map_err(|err| PyRuntimeError::new_err(format!("HDR readback failed: {err:#}")))?;
        let arr =
            ndarray::Array3::from_shape_vec((self.height as usize, self.width as usize, 4), data)
                .map_err(|_| {
                PyRuntimeError::new_err("failed to reshape HDR buffer into numpy array")
            })?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path_obj = Path::new(path);
        let ext = path_obj
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| PyValueError::new_err("expected .exr extension for HDR frame save"))?;
        if !ext.eq_ignore_ascii_case("exr") {
            return Err(PyValueError::new_err(format!(
                "expected .exr extension for HDR frame save, got .{}",
                ext
            )));
        }

        #[cfg(feature = "images")]
        {
            py.allow_threads(|| -> anyhow::Result<()> {
                let data = self.read_rgba_f32()?;
                exr_write::write_exr_rgba_f32(path_obj, self.width, self.height, &data, "beauty")
                    .map_err(anyhow::Error::msg)?;
                Ok(())
            })
            .map_err(|err| PyRuntimeError::new_err(format!("failed to save EXR: {err:#}")))?;
            Ok(())
        }

        #[cfg(not(feature = "images"))]
        {
            Err(PyRuntimeError::new_err(
                "saving HDR frames requires the 'images' feature",
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("HdrFrame(width={}, height={})", self.width, self.height)
    }
}
