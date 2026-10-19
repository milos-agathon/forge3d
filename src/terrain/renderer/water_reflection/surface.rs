use super::*;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};

impl TerrainScene {
    pub(in crate::terrain::renderer) fn upload_water_surface(
        &self,
        width: u32,
        height: u32,
        params: &render_params::TerrainRenderParams,
        mask: &[f32],
    ) -> Result<(TrackedTexture, wgpu::TextureView)> {
        let optics = &params.water_optics;
        if optics.physical {
            for (name, shape) in [
                (
                    "height",
                    optics
                        .surface_height
                        .as_ref()
                        .map(|r| (r.len(), r[0].len())),
                ),
                (
                    "normals",
                    optics
                        .surface_normals
                        .as_ref()
                        .map(|r| (r.len(), r[0].len())),
                ),
            ] {
                if shape.is_some_and(|s| s != (height as usize, width as usize)) {
                    return Err(anyhow!(
                        "water {name} grid must match heightmap ({height}, {width})"
                    ));
                }
            }
        }
        let supplied = optics.surface_height.is_some() || optics.surface_normals.is_some();
        let (tex_width, tex_height) = (width, height);
        let mut data = Vec::<[f32; 4]>::with_capacity((tex_width * tex_height) as usize);
        for y in 0..tex_height as usize {
            for x in 0..tex_width as usize {
                let h = optics
                    .surface_height
                    .as_ref()
                    .filter(|_| supplied)
                    .map_or(0.0, |r| r[y][x]);
                let n = if let Some(rows) = optics.surface_normals.as_ref().filter(|_| supplied) {
                    let n = rows[y][x];
                    let length = n.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
                    n.map(|v| (f64::from(v) / length) as f32)
                } else if let Some(rows) = optics.surface_height.as_ref().filter(|_| supplied) {
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(width as usize - 1);
                    let y0 = y.saturating_sub(1);
                    let y1 = (y + 1).min(height as usize - 1);
                    // Match the renderer's existing minimum mesh span. Compute
                    // differences in f64 so finite f32 heights do not overflow here.
                    let span = f64::from(params.terrain_span.max(1e-3));
                    let dx = (f64::from(rows[y][x1]) - f64::from(rows[y][x0]))
                        * f64::from(params.z_scale)
                        * f64::from(width - 1)
                        / ((x1 - x0) as f64 * span);
                    let dy = (f64::from(rows[y1][x]) - f64::from(rows[y0][x]))
                        * f64::from(params.z_scale)
                        * f64::from(height - 1)
                        / ((y1 - y0) as f64 * span);
                    let n = glam::DVec3::new(-dx, -dy, 1.0).normalize();
                    [n.x as f32, n.y as f32, n.z as f32]
                } else {
                    [0.0, 0.0, 1.0]
                };
                if !(h * params.z_scale).is_finite()
                    || n.iter().any(|v| !v.is_finite())
                    || n[2] <= 0.0
                {
                    return Err(anyhow!(
                        "water surface exceeds finite world-coordinate or positive-Z normal range"
                    ));
                }
                // Upper-hemisphere stereographic encoding preserves steep normals.
                // Share the existing mask binding: mask, normal.xy, raw height.
                let divisor = 1.0 + f64::from(n[2]);
                let encoded = [
                    (f64::from(n[0]) / divisor) as f32,
                    (f64::from(n[1]) / divisor) as f32,
                ];
                if encoded[0] * encoded[0] + encoded[1] * encoded[1] >= 1.0 {
                    return Err(anyhow!(
                        "water normal loses positive Z at packed GPU precision"
                    ));
                }
                data.push([mask[y * width as usize + x], encoded[0], encoded[1], h]);
            }
        }
        let size = wgpu::Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: 1,
        };
        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.water_surface"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(tex_width * 16),
                rows_per_image: Some(tex_height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((texture, view))
    }
}
