use super::super::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "HdrFrame")]
pub struct HdrFrame {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) texture: wgpu::Texture,
    pub(crate) width: u32,
    pub(crate) height: u32,
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

    fn read_rgba_f32(&self) -> anyhow::Result<Vec<f32>> {
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
        let data = self
            .read_rgba_f32()
            .map_err(|err| PyRuntimeError::new_err(format!("HDR readback failed: {err:#}")))?;
        let arr = ndarray::Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            data,
        )
        .map_err(|_| {
            PyRuntimeError::new_err("failed to reshape HDR buffer into numpy array")
        })?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let path_obj = Path::new(path);
        if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
            if !ext.eq_ignore_ascii_case("exr") {
                return Err(PyValueError::new_err(format!(
                    "HdrFrame only supports .exr save, got .{}",
                    ext
                )));
            }
        } else {
            return Err(PyValueError::new_err(
                "HdrFrame.save() requires a path with .exr extension",
            ));
        }

        #[cfg(feature = "images")]
        {
            let data = self
                .read_rgba_f32()
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("HDR readback failed: {err}"))
                })?;

            exr_write::write_exr_rgba_f32(
                path_obj,
                self.width,
                self.height,
                &data,
                "beauty",
            )
            .map_err(|err| {
                PyRuntimeError::new_err(format!("failed to write EXR: {err:#}"))
            })?;
            Ok(())
        }
        #[cfg(not(feature = "images"))]
        {
            Err(PyRuntimeError::new_err(
                "saving EXR files requires the 'images' feature",
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HdrFrame(width={}, height={}, format=Rgba16Float)",
            self.width, self.height
        )
    }
}
