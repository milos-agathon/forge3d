use super::super::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "AovFrame")]
pub struct AovFrame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    albedo_texture: Option<wgpu::Texture>,
    normal_texture: Option<wgpu::Texture>,
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

    fn read_texture_rgba_f32(&self, texture: &wgpu::Texture) -> anyhow::Result<Vec<f32>> {
        crate::core::hdr::read_hdr_texture(
            &self.device,
            &self.queue,
            texture,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba16Float,
        )
        .map_err(anyhow::Error::msg)
    }

    fn read_albedo_data(&self) -> anyhow::Result<Vec<f32>> {
        let texture = self
            .albedo_texture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Albedo AOV not available"))?;
        self.read_texture_rgba_f32(texture)
    }

    fn read_normal_data(&self) -> anyhow::Result<Vec<f32>> {
        let texture = self
            .normal_texture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Normal AOV not available"))?;
        self.read_texture_rgba_f32(texture)
    }

    fn read_depth_data(&self) -> anyhow::Result<Vec<f32>> {
        let texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Depth AOV not available"))?;
        let rgba = self.read_texture_rgba_f32(texture)?;
        Ok(rgba.chunks_exact(4).map(|px| px[0]).collect())
    }

    fn rgba_to_rgb_array(
        &self,
        rgba: &[f32],
    ) -> anyhow::Result<ndarray::Array3<f32>> {
        let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
        for px in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        ndarray::Array3::from_shape_vec((self.height as usize, self.width as usize, 3), rgb)
            .map_err(|_| anyhow::anyhow!("failed to reshape RGBA buffer into RGB array"))
    }

    fn encode_rgb_png(
        &self,
        rgb: impl Iterator<Item = [f32; 3]>,
    ) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((self.width * self.height * 4) as usize);
        for [r, g, b] in rgb {
            bytes.push((r.clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((g.clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((b.clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push(255);
        }
        bytes
    }

    fn write_png_bytes(&self, path: &str, data: &[u8]) -> PyResult<()> {
        image_write::write_png_rgba8(Path::new(path), data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))
    }

    #[cfg(feature = "images")]
    fn extend_rgba_channels(
        channels: &mut Vec<exr_write::ExrChannelData>,
        prefix: &str,
        data: &[f32],
    ) -> anyhow::Result<()> {
        let expected_pixels = data.len() / 4;
        anyhow::ensure!(
            data.len() == expected_pixels * 4,
            "expected RGBA buffer for EXR export"
        );
        let mut r = Vec::with_capacity(expected_pixels);
        let mut g = Vec::with_capacity(expected_pixels);
        let mut b = Vec::with_capacity(expected_pixels);
        let mut a = Vec::with_capacity(expected_pixels);
        for px in data.chunks_exact(4) {
            r.push(px[0]);
            g.push(px[1]);
            b.push(px[2]);
            a.push(px[3]);
        }
        channels.extend([
            exr_write::ExrChannelData {
                name: format!("{prefix}.R"),
                data: r,
                quantize_linearly: false,
            },
            exr_write::ExrChannelData {
                name: format!("{prefix}.G"),
                data: g,
                quantize_linearly: false,
            },
            exr_write::ExrChannelData {
                name: format!("{prefix}.B"),
                data: b,
                quantize_linearly: false,
            },
            exr_write::ExrChannelData {
                name: format!("{prefix}.A"),
                data: a,
                quantize_linearly: true,
            },
        ]);
        Ok(())
    }

    #[cfg(feature = "images")]
    fn extend_rgb_channels(
        channels: &mut Vec<exr_write::ExrChannelData>,
        prefix: &str,
        data: &[f32],
        suffixes: [&str; 3],
        quantize_linearly: bool,
    ) -> anyhow::Result<()> {
        let expected_pixels = data.len() / 4;
        anyhow::ensure!(
            data.len() == expected_pixels * 4,
            "expected RGBA buffer for EXR export"
        );
        let mut c0 = Vec::with_capacity(expected_pixels);
        let mut c1 = Vec::with_capacity(expected_pixels);
        let mut c2 = Vec::with_capacity(expected_pixels);
        for px in data.chunks_exact(4) {
            c0.push(px[0]);
            c1.push(px[1]);
            c2.push(px[2]);
        }
        channels.extend([
            exr_write::ExrChannelData {
                name: format!("{prefix}.{}", suffixes[0]),
                data: c0,
                quantize_linearly,
            },
            exr_write::ExrChannelData {
                name: format!("{prefix}.{}", suffixes[1]),
                data: c1,
                quantize_linearly,
            },
            exr_write::ExrChannelData {
                name: format!("{prefix}.{}", suffixes[2]),
                data: c2,
                quantize_linearly,
            },
        ]);
        Ok(())
    }

    #[cfg(feature = "images")]
    fn extend_scalar_channel(
        channels: &mut Vec<exr_write::ExrChannelData>,
        prefix: &str,
        data: &[f32],
        suffix: &str,
    ) {
        channels.push(exr_write::ExrChannelData {
            name: format!("{prefix}.{suffix}"),
            data: data.to_vec(),
            quantize_linearly: true,
        });
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

    fn albedo<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArray3<f32>> {
        let rgba = self
            .read_albedo_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let arr = self
            .rgba_to_rgb_array(&rgba)
            .map_err(|err| PyRuntimeError::new_err(format!("reshape failed: {err:#}")))?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn normal<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArray3<f32>> {
        let rgba = self
            .read_normal_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let arr = self
            .rgba_to_rgb_array(&rgba)
            .map_err(|err| PyRuntimeError::new_err(format!("reshape failed: {err:#}")))?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn depth<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArray2<f32>> {
        let data = self
            .read_depth_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let arr =
            ndarray::Array2::from_shape_vec((self.height as usize, self.width as usize), data)
                .map_err(|_| {
                    PyRuntimeError::new_err("failed to reshape depth buffer into numpy array")
                })?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn save_albedo(&self, path: &str) -> PyResult<()> {
        let rgba = self
            .read_albedo_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let png = self.encode_rgb_png(rgba.chunks_exact(4).map(|px| [px[0], px[1], px[2]]));
        self.write_png_bytes(path, &png)
    }

    fn save_normal(&self, path: &str) -> PyResult<()> {
        let rgba = self
            .read_normal_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let png = self.encode_rgb_png(rgba.chunks_exact(4).map(|px| {
            [
                px[0] * 0.5 + 0.5,
                px[1] * 0.5 + 0.5,
                px[2] * 0.5 + 0.5,
            ]
        }));
        self.write_png_bytes(path, &png)
    }

    fn save_depth(&self, path: &str) -> PyResult<()> {
        let depth = self
            .read_depth_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let png = self.encode_rgb_png(depth.iter().copied().map(|value| [value, value, value]));
        self.write_png_bytes(path, &png)
    }

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

    #[cfg(feature = "images")]
    fn save_exr(&self, path: &str, beauty_frame: &crate::Frame) -> PyResult<()> {
        if beauty_frame.dimensions() != (self.width, self.height) {
            return Err(PyValueError::new_err(format!(
                "beauty frame size {:?} does not match AOV size {:?}",
                beauty_frame.dimensions(),
                (self.width, self.height)
            )));
        }

        let beauty = beauty_frame
            .read_rgba_f32()
            .map_err(|err| PyRuntimeError::new_err(format!("beauty readback failed: {err:#}")))?;

        let mut channels = Vec::new();
        Self::extend_rgba_channels(&mut channels, "beauty", &beauty)
            .map_err(|err| PyRuntimeError::new_err(format!("EXR channel build failed: {err:#}")))?;

        if let Some(texture) = self.albedo_texture.as_ref() {
            let rgba = self.read_texture_rgba_f32(texture).map_err(|err| {
                PyRuntimeError::new_err(format!("albedo readback failed: {err:#}"))
            })?;
            Self::extend_rgb_channels(&mut channels, "albedo", &rgba, ["R", "G", "B"], false)
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("EXR channel build failed: {err:#}"))
                })?;
        }
        if let Some(texture) = self.normal_texture.as_ref() {
            let rgba = self.read_texture_rgba_f32(texture).map_err(|err| {
                PyRuntimeError::new_err(format!("normal readback failed: {err:#}"))
            })?;
            Self::extend_rgb_channels(&mut channels, "normal", &rgba, ["X", "Y", "Z"], true)
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("EXR channel build failed: {err:#}"))
                })?;
        }
        if self.depth_texture.is_some() {
            let depth = self.read_depth_data().map_err(|err| {
                PyRuntimeError::new_err(format!("depth readback failed: {err:#}"))
            })?;
            Self::extend_scalar_channel(&mut channels, "depth", &depth, "Z");
        }

        exr_write::write_exr_f32_channels(Path::new(path), self.width, self.height, channels)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write EXR: {err:#}")))?;
        Ok(())
    }

    #[cfg(not(feature = "images"))]
    fn save_exr(&self, _path: &str, _beauty_frame: &crate::Frame) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "saving EXR files requires the 'images' feature",
        ))
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
