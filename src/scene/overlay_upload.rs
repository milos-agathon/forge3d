//! Shared raster input upload for Scene and the private label compositor.
use crate::core::overlays::OverlayRenderer;
use crate::core::resource_tracker::AllocationOwner;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

pub(super) struct RasterOverlayTarget<'a> {
    pub allocation_owner: &'a AllocationOwner,
    pub renderer: Option<&'a mut OverlayRenderer>,
    pub width: u32,
    pub height: u32,
    pub height_view: Option<&'a wgpu::TextureView>,
}

impl RasterOverlayTarget<'_> {
    pub fn upload(
        mut self,
        image: &pyo3::types::PyAny,
        alpha: Option<f32>,
        offset_xy: Option<(i32, i32)>,
        scale: Option<f32>,
    ) -> PyResult<()> {
        let _allocation_scope = self.allocation_owner.activate();
        // Validate input array (HxWx3 or HxWx4, uint8)
        let (h, w, c, data) = if let Ok(arr) = image.extract::<numpy::PyReadonlyArray3<u8>>() {
            let shape = arr.shape();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay image must be HxWxC uint8",
                ));
            }
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2] as u32;
            if c != 3 && c != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay channels must be 3 or 4",
                ));
            }
            (h, w, c, arr.as_array().to_owned().into_raw_vec())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected numpy uint8 array HxWxC",
            ));
        };

        // Convert to RGBA8
        let mut rgba: Vec<u8> = Vec::with_capacity((h * w * 4) as usize);
        if c == 4 {
            rgba = data;
        } else {
            // c == 3
            let mut idx = 0usize;
            while idx < data.len() {
                rgba.push(data[idx]);
                rgba.push(data[idx + 1]);
                rgba.push(data[idx + 2]);
                rgba.push(255);
                idx += 3;
            }
        }

        // Create GPU texture and upload
        let g = crate::core::gpu::try_ctx()?;
        let tex = crate::core::resource_tracker::tracked_create_texture(
            &g.device,
            &wgpu::TextureDescriptor {
                label: Some("scene-overlay-rgba8"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // The overlay shader passes colour through to an Rgba8Unorm
                // target without re-encoding, so the overlay must not be
                // sRGB-decoded on sampling either (that darkened every frame).
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        let row_bytes = w * 4;
        let padded_bpr = crate::core::gpu::align_copy_bpr(row_bytes);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
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
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Update overlay renderer state
        if let Some(ref mut ov) = self.renderer {
            // Keep GPU resources alive
            ov.set_overlay_texture(tex, view);
            // Rebind with current height view for altitude/contours
            ov.recreate_bind_group(&g.device, None, self.height_view, None, None)?;

            // Set params
            ov.set_enabled(true);
            if let Some(a) = alpha {
                ov.set_overlay_alpha(a);
            }
            let (off_x, off_y) = offset_xy.unwrap_or((0, 0));
            let scale_v = scale.unwrap_or(1.0).max(1e-3);
            let sample_s = 1.0 / scale_v;
            let uv_off_x = (off_x as f32 / self.width.max(1) as f32) * sample_s;
            let uv_off_y = (off_y as f32 / self.height.max(1) as f32) * sample_s;
            ov.set_overlay_uv(uv_off_x, uv_off_y, sample_s, sample_s);
            ov.upload_uniforms(&g.queue);
        }

        Ok(())
    }
}
