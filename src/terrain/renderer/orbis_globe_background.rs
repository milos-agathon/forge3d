use super::*;
use crate::core::resource_tracker::{tracked_create_texture, TrackedBuffer, TrackedTexture};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::terrain::renderer) struct OrbisGlobeUniforms {
    inverse_view_projection: [f32; 16],
    sphere: [f32; 4],
    base_color: [f32; 4],
    rim_color: [f32; 4],
    sun_direction: [f32; 4],
    space_color: [f32; 4],
    /// ECEF east/north/up unit vectors of the render-frame anchor; they map a
    /// render-space sphere normal back to longitude/latitude.
    earth_east: [f32; 4],
    earth_north: [f32; 4],
    earth_up: [f32; 4],
    /// x = 1 when an equirectangular Earth texture replaces `base_color`.
    earth_flags: [f32; 4],
}

/// Longest edge accepted for the ORBIS Earth texture.
pub(crate) const ORBIS_EARTH_TEXTURE_MAX_WIDTH: u32 = 8192;

/// Create the RGBA8 equirectangular Earth texture (row 0 = 90 N, column 0 =
/// 180 W). Values are stored as authored: the terrain pass writes display
/// (sRGB-encoded) values into the same Rgba8Unorm target.
pub(in crate::terrain::renderer) fn create_orbis_earth_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> anyhow::Result<(TrackedTexture, wgpu::TextureView)> {
    let expected = (width as usize) * (height as usize) * 4;
    anyhow::ensure!(
        width > 0 && height > 0 && rgba.len() == expected,
        "ORBIS Earth texture must be {width}x{height} RGBA8 ({expected} bytes), got {}",
        rgba.len()
    );
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("orbis.globe.earth-texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: texture.inner(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}

pub(in crate::terrain::renderer) fn create_orbis_globe_background_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &TrackedBuffer,
    earth_view: &wgpu::TextureView,
    earth_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("orbis.globe.background.bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(earth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(earth_sampler),
            },
        ],
    })
}

fn enu_basis(anchor: glam::DVec3) -> [[f32; 4]; 3] {
    let up = anchor.normalize();
    let lon = up.y.atan2(up.x);
    let lat = up.z.clamp(-1.0, 1.0).asin();
    let east = glam::DVec3::new(-lon.sin(), lon.cos(), 0.0);
    let north = glam::DVec3::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos());
    let pack = |v: glam::DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    [pack(east), pack(north), pack(up)]
}

/// Clear colour of the globe pass; the atmosphere halo composites over it.
const ORBIS_SPACE_COLOR: [f32; 3] = [0.1, 0.1, 0.15];

impl TerrainScene {
    pub(in crate::terrain::renderer) fn render_orbis_globe_background(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_targets: &crate::terrain::renderer::draw::RenderTargets,
        sky_present: bool,
    ) -> anyhow::Result<bool> {
        if !params.camera_mode.starts_with("clipmap:") {
            return Ok(false);
        }
        if std::env::var_os("ORBIS_DISABLE_GLOBE_BG").is_some() {
            return Ok(false);
        }
        let Some((_center, anchor)) = self.height_streaming_globe_identity() else {
            return Ok(false);
        };
        if !anchor.is_finite() || anchor.length() <= 0.0 {
            anyhow::bail!(
                "ORBIS globe background camera anchor must be finite and non-zero, got {anchor:?}"
            );
        }
        let (eye, view, projection) = Self::build_camera_matrices(params);
        let inverse = (projection * view).inverse();
        if !eye.is_finite() || !inverse.to_cols_array().iter().all(|v| v.is_finite()) {
            anyhow::bail!(
                "ORBIS globe background camera matrices must be finite (eye={eye:?})"
            );
        }
        let sun = params.decoded().light.direction;
        let basis = enu_basis(anchor);
        let uniforms = OrbisGlobeUniforms {
            inverse_view_projection: inverse.to_cols_array(),
            sphere: [
                0.0,
                0.0,
                -(anchor.length() as f32),
                crate::terrain::clipmap::globe::GlobeFrame::WGS84_MEAN_RADIUS_M as f32,
            ],
            base_color: [0.035, 0.095, 0.12, 1.0],
            rim_color: [0.12, 0.34, 0.55, 1.0],
            // Same render-space light as the terrain pass, so the Earth
            // outside the DEM shares the terrain's day side.
            sun_direction: [
                sun[0],
                sun[1],
                sun[2],
                if sky_present { 1.0 } else { 0.0 },
            ],
            space_color: [
                ORBIS_SPACE_COLOR[0],
                ORBIS_SPACE_COLOR[1],
                ORBIS_SPACE_COLOR[2],
                1.0,
            ],
            earth_east: basis[0],
            earth_north: basis[1],
            earth_up: basis[2],
            earth_flags: [
                if self.orbis_earth_textured { 1.0 } else { 0.0 },
                0.0,
                0.0,
                0.0,
            ],
        };
        self.queue
            .write_buffer(&self.orbis_globe_background_uniform, 0, bytemuck::bytes_of(&uniforms));
        let mut cache = self
            .orbis_globe_background_pipeline
            .lock()
            .map_err(|_| anyhow::anyhow!("ORBIS globe background pipeline cache poisoned"))?;
        if cache.as_ref().map(|(samples, _)| *samples) != Some(render_targets.sample_count) {
            *cache = Some((
                render_targets.sample_count,
                Self::create_orbis_globe_background_pipeline(
                    self.device.as_ref(),
                    &self.orbis_globe_background_layout,
                    self.color_format,
                    render_targets.sample_count,
                ),
            ));
        }
        let pipeline = &cache.as_ref().expect("pipeline just inserted").1;
        let load = if sky_present {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: f64::from(ORBIS_SPACE_COLOR[0]),
                g: f64::from(ORBIS_SPACE_COLOR[1]),
                b: f64::from(ORBIS_SPACE_COLOR[2]),
                a: 1.0,
            })
        };
        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = render_targets
            .msaa_view
            .as_ref()
            .map(|_| &render_targets.internal_view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("orbis.globe.background"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.orbis_globe_background_bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(true)
    }
}

impl TerrainRenderer {
    /// Replace the equirectangular Earth texture drawn where the globe has no
    /// terrain source data (row 0 = 90 N, column 0 = 180 W, width = 2 x height).
    pub(crate) fn set_orbis_earth_texture(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            height >= 2 && width == height * 2,
            "ORBIS Earth texture must be equirectangular (width == 2 * height, height >= 2), got {width}x{height}"
        );
        let max_width = ORBIS_EARTH_TEXTURE_MAX_WIDTH
            .min(self.scene.device.limits().max_texture_dimension_2d);
        anyhow::ensure!(
            width <= max_width,
            "ORBIS Earth texture width {width} exceeds the supported maximum {max_width}"
        );
        let (texture, view) =
            create_orbis_earth_texture(&self.scene.device, &self.scene.queue, width, height, rgba)?;
        self.scene.orbis_globe_background_bind_group = create_orbis_globe_background_bind_group(
            &self.scene.device,
            &self.scene.orbis_globe_background_layout,
            &self.scene.orbis_globe_background_uniform,
            &view,
            &self.scene.orbis_earth_sampler,
        );
        self.scene.orbis_earth_texture = texture;
        self.scene.orbis_earth_view = view;
        self.scene.orbis_earth_textured = true;
        Ok(())
    }
}
