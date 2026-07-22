use super::super::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Frame")]
pub struct Frame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture: crate::core::resource_tracker::TrackedTexture,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl Frame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture: crate::core::resource_tracker::TrackedTexture,
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

    pub(crate) fn read_tight_bytes(&self) -> anyhow::Result<Vec<u8>> {
        read_texture_tight(
            &self.device,
            &self.queue,
            &self.texture,
            (self.width, self.height),
            self.format,
        )
    }

    /// Rehydrate a cached, tightly packed single-sample frame into a tracked
    /// GPU texture. Queue texture writes accept tight rows (the 256-byte row
    /// alignment applies to encoder copies, not `Queue::write_texture`).
    pub(crate) fn from_tight_bytes(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        bytes: &[u8],
    ) -> anyhow::Result<Self> {
        let bytes_per_pixel = match format {
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb => 4usize,
            wgpu::TextureFormat::Rgba16Float => 8usize,
            other => anyhow::bail!("unsupported cached frame format: {other:?}"),
        };
        let expected = width as usize * height as usize * bytes_per_pixel;
        anyhow::ensure!(
            width > 0 && height > 0 && bytes.len() == expected,
            "cached frame byte length mismatch: got {}, expected {} for {}x{} {:?}",
            bytes.len(),
            expected,
            width,
            height,
            format
        );
        let texture = crate::core::resource_tracker::tracked_create_texture(
            device.as_ref(),
            &wgpu::TextureDescriptor {
                label: Some("anamnesis.cached_frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * bytes_per_pixel as u32),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(Self::new(device, queue, texture, width, height, format))
    }

    #[cfg(feature = "images")]
    pub(crate) fn read_rgba_f32(&self) -> anyhow::Result<Vec<f32>> {
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb => {
                let data = self.read_tight_bytes()?;
                let mut rgba = Vec::with_capacity(data.len());
                for px in data.chunks_exact(4) {
                    match self.format {
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                            rgba.push(px[2] as f32 / 255.0);
                            rgba.push(px[1] as f32 / 255.0);
                            rgba.push(px[0] as f32 / 255.0);
                            rgba.push(px[3] as f32 / 255.0);
                        }
                        _ => {
                            rgba.push(px[0] as f32 / 255.0);
                            rgba.push(px[1] as f32 / 255.0);
                            rgba.push(px[2] as f32 / 255.0);
                            rgba.push(px[3] as f32 / 255.0);
                        }
                    }
                }
                Ok(rgba)
            }
            wgpu::TextureFormat::Rgba16Float => crate::core::hdr::read_hdr_texture(
                &self.device,
                &self.queue,
                &self.texture,
                self.width,
                self.height,
                self.format,
            )
            .map_err(anyhow::Error::msg),
            other => Err(anyhow::anyhow!(
                "unsupported texture format for float readback: {:?}",
                other
            )),
        }
    }

    #[cfg(feature = "images")]
    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
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
