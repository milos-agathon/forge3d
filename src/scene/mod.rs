// T41-BEGIN:scene-module
#![allow(dead_code)]
use numpy::{IntoPyArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::path::PathBuf;
use wgpu::util::DeviceExt;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[derive(Debug, Clone)]
pub struct SceneGlobals {
    pub globals: crate::terrain::Globals,
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
}

impl Default for SceneGlobals {
    fn default() -> Self {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0, 2.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = crate::camera::perspective_wgpu(45f32.to_radians(), 4.0 / 3.0, 0.1, 100.0);
        Self {
            globals: crate::terrain::Globals::default(),
            view,
            proj,
        }
    }
}

#[pyclass(module = "_vulkan_forge", name = "Scene")]
pub struct Scene {
    width: u32,
    height: u32,
    grid: u32,

    tp: crate::terrain::pipeline::TerrainPipeline,
    bg0_globals: wgpu::BindGroup,
    bg1_height: wgpu::BindGroup,
    bg2_lut: wgpu::BindGroup,

    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,

    ubo: wgpu::Buffer,
    colormap: crate::terrain::ColormapLUT,
    lut_format: &'static str,

    color: wgpu::Texture,
    color_view: wgpu::TextureView,

    height_view: Option<wgpu::TextureView>,
    height_sampler: Option<wgpu::Sampler>,

    scene: SceneGlobals,
    last_uniforms: crate::terrain::TerrainUniforms,
}

#[pymethods]
impl Scene {
    #[new]
    #[pyo3(text_signature = "(width, height, grid=128, colormap='viridis')")]
    pub fn new(
        width: u32,
        height: u32,
        grid: Option<u32>,
        colormap: Option<String>,
    ) -> PyResult<Self> {
        let grid = grid.unwrap_or(128).max(2);
        // Use shared GPU context
        let g = crate::gpu::ctx();

        // Target
        let color = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());

        // Pipeline
        let tp = crate::terrain::pipeline::TerrainPipeline::create(&g.device, TEXTURE_FORMAT);

        // Mesh
        let (vbuf, ibuf, nidx) = {
            // Same as terrain::build_grid_xyuv (private) — inline minimal copy to avoid re-export churn.
            // Minimal grid that matches T3.1/T3.3 vertex layout: interleaved [x, z, u, v] (Float32x4) => 16-byte stride.
            let n = grid.max(2) as usize;
            let (w, h) = (n, n);
            let scale = 1.5f32;
            let step_x = (2.0 * scale) / (w as f32 - 1.0);
            let step_z = (2.0 * scale) / (h as f32 - 1.0);
            let mut verts = Vec::<f32>::with_capacity(w * h * 4);
            for j in 0..h {
                for i in 0..w {
                    let x = -scale + i as f32 * step_x;
                    let z = -scale + j as f32 * step_z;
                    let u = i as f32 / (w as f32 - 1.0);
                    let v = j as f32 / (h as f32 - 1.0);
                    verts.extend_from_slice(&[x, z, u, v]);
                }
            }
            let mut idx = Vec::<u32>::with_capacity((w - 1) * (h - 1) * 6);
            for j in 0..h - 1 {
                for i in 0..w - 1 {
                    let a = (j * w + i) as u32;
                    let b = (j * w + i + 1) as u32;
                    let c = ((j + 1) * w + i) as u32;
                    let d = ((j + 1) * w + i + 1) as u32;
                    idx.extend_from_slice(&[a, c, b, b, c, d]);
                }
            }
            let vbuf = g
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene-xyuv-vbuf"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ibuf = g
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene-xyuv-ibuf"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::INDEX,
                });
            (vbuf, ibuf, idx.len() as u32)
        };

        // Globals/UBO
        let mut scene = SceneGlobals::default();
        // set correct aspect
        scene.proj = crate::camera::perspective_wgpu(
            45f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            100.0,
        );
        let uniforms = scene.globals.to_uniforms(scene.view, scene.proj);
        let ubo = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene-ubo"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // LUT (+ friendly validation against SUPPORTED)
        let cmap_name = colormap.as_deref().unwrap_or("viridis");
        if !crate::colormap::SUPPORTED.contains(&cmap_name) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unknown colormap '{}'. Supported: {}",
                cmap_name,
                crate::colormap::SUPPORTED.join(", ")
            )));
        }
        let which = crate::colormap::map_name_to_type(cmap_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let (lut, lut_format) =
            crate::terrain::ColormapLUT::new(&g.device, &g.queue, &g.adapter, which)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Dummy height (non-trivial): upload a tiny 2×2 gradient with proper 256-byte row padding.
        // This guarantees the first frame has variance, so the PNG won't compress to a tiny file.
        let (hview, hsamp) = {
            let w = 2u32;
            let h = 2u32;
            let tex = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scene-dummy-height"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Row padding to WebGPU's required alignment for height>1.
            let row_bytes = w * 4;
            let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
            let src_vals: [f32; 4] = [0.00, 0.25, 0.50, 0.75]; // row-major: [[0.00, 0.25],[0.50, 0.75]]
            let src_bytes: &[u8] = bytemuck::cast_slice(&src_vals);
            let mut padded = vec![0u8; (padded_bpr * h) as usize];
            for y in 0..h as usize {
                let s = y * row_bytes as usize;
                let d = y * padded_bpr as usize;
                padded[d..d + row_bytes as usize]
                    .copy_from_slice(&src_bytes[s..s + row_bytes as usize]);
            }
            g.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&Default::default());
            // NOTE: Height is R32Float → must bind with a NonFiltering sampler. Many backends forbid
            // linear filtering on 32-bit float textures. Use NEAREST to satisfy NonFiltering binding.
            let samp = g.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("scene-height-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, samp)
        };

        // Bind groups (cached)
        let bg0_globals = tp.make_bg_globals(&g.device, &ubo);
        let bg1_height = tp.make_bg_height(&g.device, &hview, &hsamp);
        let bg2_lut = tp.make_bg_lut(&g.device, &lut.view, &lut.sampler);

        Ok(Self {
            width,
            height,
            grid,
            tp,
            bg0_globals,
            bg1_height,
            bg2_lut,
            vbuf,
            ibuf,
            nidx,
            ubo,
            colormap: lut,
            lut_format,
            color,
            color_view,
            height_view: Some(hview),
            height_sampler: Some(hsamp),
            scene,
            last_uniforms: uniforms,
        })
    }

    #[pyo3(text_signature = "($self, eye, target, up, fovy_deg, znear, zfar)")]
    pub fn set_camera_look_at(
        &mut self,
        eye: (f32, f32, f32),
        target: (f32, f32, f32),
        up: (f32, f32, f32),
        fovy_deg: f32,
        znear: f32,
        zfar: f32,
    ) -> PyResult<()> {
        use crate::camera;
        let aspect = self.width as f32 / self.height as f32;
        let eye_v = glam::Vec3::new(eye.0, eye.1, eye.2);
        let target_v = glam::Vec3::new(target.0, target.1, target.2);
        let up_v = glam::Vec3::new(up.0, up.1, up.2);
        camera::validate_camera_params(eye_v, target_v, up_v, fovy_deg, znear, zfar)?;
        self.scene.view = glam::Mat4::look_at_rh(eye_v, target_v, up_v);
        self.scene.proj = camera::perspective_wgpu(fovy_deg.to_radians(), aspect, znear, zfar);
        let uniforms = self
            .scene
            .globals
            .to_uniforms(self.scene.view, self.scene.proj);
        let g = crate::gpu::ctx();
        g.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));
        self.last_uniforms = uniforms;
        Ok(())
    }

    #[pyo3(text_signature = "($self, height_r32f)")]
    pub fn set_height_from_r32f(&mut self, height_r32f: &pyo3::types::PyAny) -> PyResult<()> {
        // Accept numpy array float32 (H,W)
        let arr: numpy::PyReadonlyArray2<f32> = height_r32f.extract()?;
        let (h, w) = (arr.shape()[0] as u32, arr.shape()[1] as u32);
        let data = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("height must be C-contiguous float32[H,W]")
        })?;

        let g = crate::gpu::ctx();
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-height-r32f"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // WebGPU requires bytes_per_row to be COPY_BYTES_PER_ROW_ALIGNMENT aligned when height > 1.
        // Build a temporary padded buffer: each row (w*4 bytes) is copied into a padded stride.
        let row_bytes = w * 4;
        let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
        let src_bytes: &[u8] = bytemuck::cast_slice::<f32, u8>(data);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..(h as usize) {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize]
                .copy_from_slice(&src_bytes[s..s + row_bytes as usize]);
        }
        g.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&Default::default());
        let samp = g.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene-height-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        self.height_view = Some(view);
        self.height_sampler = Some(samp);

        // Rebuild only BG1 using cached layout
        let bg1 = self.tp.make_bg_height(
            &g.device,
            self.height_view.as_ref().unwrap(),
            self.height_sampler.as_ref().unwrap(),
        );
        self.bg1_height = bg1;
        Ok(())
    }

    /// Render the current frame to a PNG on disk.
    ///
    /// Parameters
    /// ----------
    /// path : str | os.PathLike
    ///     Destination file path for the PNG image.
    ///
    /// Notes
    /// -----
    /// The written PNG's raw RGBA bytes will match those returned by
    /// `Scene.render_rgba()` on the same frame (row-major, C-contiguous).
    #[pyo3(text_signature = "($self, path)")]
    pub fn render_png(&mut self, path: PathBuf) -> PyResult<()> {
        let g = crate::gpu::ctx();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene-encoder"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&self.tp.pipeline);
            rp.set_bind_group(0, &self.bg0_globals, &[]);
            rp.set_bind_group(1, &self.bg1_height, &[]);
            rp.set_bind_group(2, &self.bg2_lut, &[]);
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.nidx, 0, 0..1);
        }
        g.queue.submit(Some(encoder.finish()));

        // Readback -> PNG (same as TerrainSpike)
        let bpp = 4u32;
        let unpadded = self.width * bpp;
        let padded = crate::gpu::align_copy_bpr(unpadded);
        let size = (padded * self.height) as wgpu::BufferAddress;
        let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(self.height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        g.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        g.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
        for row in 0..self.height {
            let s = (row * padded) as usize;
            let e = s + unpadded as usize;
            pixels.extend_from_slice(&data[s..e]);
        }
        drop(data);
        readback.unmap();
        let img = image::RgbaImage::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Invalid image buffer"))?;
        img.save(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Render the current frame into a NumPy array with shape (H, W, 4), dtype=uint8.
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     C-contiguous (row-major) RGBA byte buffer whose pixels are byte-for-byte
    ///     identical to the PNG produced by `render_png()` for the same frame.
    #[pyo3(text_signature = "($self)")]
    pub fn render_rgba<'py>(
        &mut self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray3<u8>>> {
        // Encode a frame exactly like render_png(), then return (H,W,4) u8
        let g = crate::gpu::ctx();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene-encoder-rgba"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp-rgba"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&self.tp.pipeline);
            rp.set_bind_group(0, &self.bg0_globals, &[]);
            rp.set_bind_group(1, &self.bg1_height, &[]);
            rp.set_bind_group(2, &self.bg2_lut, &[]);
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.nidx, 0, 0..1);
        }
        g.queue.submit(Some(encoder.finish()));

        // Readback -> unpadded RGBA bytes
        let bpp = 4u32;
        let unpadded = self.width * bpp;
        let padded = crate::gpu::align_copy_bpr(unpadded);
        let size = (padded * self.height) as wgpu::BufferAddress;

        let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-readback-rgba"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder-rgba"),
            });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(self.height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        g.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        g.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();

        let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
        for row in 0..self.height {
            let s = (row * padded) as usize;
            let e = s + unpadded as usize;
            pixels.extend_from_slice(&data[s..e]);
        }
        drop(data);
        readback.unmap();

        // Convert to NumPy (H,W,4) u8
        let arr =
            ndarray::Array3::from_shape_vec((self.height as usize, self.width as usize, 4), pixels)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_uniforms_f32<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
        let bytes = bytemuck::bytes_of(&self.last_uniforms);
        let fl: &[f32] = bytemuck::cast_slice(bytes);
        Ok(numpy::PyArray1::from_vec_bound(py, fl.to_vec()))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_lut_format(&self) -> &'static str {
        self.lut_format
    }
}
// T41-END:scene-module
