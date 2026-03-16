use super::super::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Frame")]
pub struct Frame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl Frame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            texture,
            width,
            height,
            format,
        }
    }

    fn read_tight_bytes(&self) -> anyhow::Result<Vec<u8>> {
        read_texture_tight(
            &self.device,
            &self.queue,
            &self.texture,
            (self.width, self.height),
            self.format,
        )
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl Frame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "Frame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn format(&self) -> String {
        format!("{:?}", self.format)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let path_obj = Path::new(path);
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
                    if !ext.eq_ignore_ascii_case("png") {
                        return Err(PyValueError::new_err(format!(
                            "expected .png extension for RGBA8 frame save, got .{}",
                            ext
                        )));
                    }
                }
                let mut data = self
                    .read_tight_bytes()
                    .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
                for px in data.chunks_exact_mut(4) {
                    px[3] = 255;
                }
                image_write::write_png_rgba8(path_obj, &data, self.width, self.height).map_err(
                    |err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")),
                )?;
                Ok(())
            }
            wgpu::TextureFormat::Rgba16Float => {
                if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
                    if !ext.eq_ignore_ascii_case("exr") {
                        return Err(PyValueError::new_err(format!(
                            "expected .exr extension for RGBA16F frame save, got .{}",
                            ext
                        )));
                    }
                }
                #[cfg(feature = "images")]
                {
                    let data = crate::core::hdr::read_hdr_texture(
                        &self.device,
                        &self.queue,
                        &self.texture,
                        self.width,
                        self.height,
                        self.format,
                    )
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
                        "saving RGBA16F frames requires the 'images' feature",
                    ))
                }
            }
            other => Err(PyValueError::new_err(format!(
                "unsupported texture format for save: {:?}",
                other
            ))),
        }
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<u8>> {
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                let data = self
                    .read_tight_bytes()
                    .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
                let arr = ndarray::Array3::from_shape_vec(
                    (self.height as usize, self.width as usize, 4),
                    data,
                )
                .map_err(|_| {
                    PyRuntimeError::new_err("failed to reshape RGBA buffer into numpy array")
                })?;
                Ok(arr.into_pyarray_bound(py).into_gil_ref())
            }
            wgpu::TextureFormat::Rgba16Float => Err(PyRuntimeError::new_err(
                "to_numpy for RGBA16F frames is not implemented yet",
            )),
            other => Err(PyValueError::new_err(format!(
                "unsupported texture format for numpy conversion: {:?}",
                other
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame(width={}, height={}, format={:?})",
            self.width, self.height, self.format
        )
    }
}
