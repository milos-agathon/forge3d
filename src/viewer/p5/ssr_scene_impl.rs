// src/viewer/p5/ssr_scene_impl.rs
// SSR scene helpers for the Viewer
// Extracted from mod.rs as part of the viewer refactoring

use wgpu::util::DeviceExt;

use crate::p5::ssr::SsrScenePreset;
use crate::viewer::event_loop::update_ipc_stats;
use crate::viewer::viewer_ssr_scene::{build_ssr_albedo_texture, build_ssr_scene_mesh};
use crate::viewer::Viewer;

impl Viewer {
    pub(crate) fn upload_ssr_scene(&mut self, preset: &SsrScenePreset) -> anyhow::Result<()> {
        let mesh = build_ssr_scene_mesh(preset);
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            anyhow::bail!("SSR scene mesh is empty");
        }

        let vertex_data = bytemuck::cast_slice(&mesh.vertices);
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ssr.scene.vb"),
                contents: vertex_data,
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ssr.scene.ib"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.geom_vb = Some(vb);
        self.geom_ib = Some(ib);
        self.geom_index_count = mesh.indices.len() as u32;
        self.geom_bind_group = None;
        
        update_ipc_stats(
            true,
            mesh.vertices.len() as u32,
            mesh.indices.len() as u32,
            true,
        );

        let tex_size = 1024u32;
        let pixels = build_ssr_albedo_texture(preset, tex_size);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.ssr.scene.albedo"),
            size: wgpu::Extent3d {
                width: tex_size,
                height: tex_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(tex_size * 4),
                rows_per_image: Some(tex_size),
            },
            wgpu::Extent3d {
                width: tex_size,
                height: tex_size,
                depth_or_array_layers: 1,
            },
        );
        self.albedo_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.albedo_texture = Some(texture);
        self.albedo_sampler.get_or_insert_with(|| {
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("viewer.ssr.scene.albedo.sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        });

        self.ensure_geom_bind_group()?;

        Ok(())
    }

    pub(crate) fn apply_ssr_scene_preset(&mut self) -> anyhow::Result<()> {
        let preset = match self.ssr_scene_preset.clone() {
            Some(p) => p,
            None => {
                let preset = SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?;
                self.ssr_scene_preset = Some(preset.clone());
                preset
            }
        };

        self.upload_ssr_scene(&preset)?;
        self.generate_stripe_env_map(&preset)?;
        self.ssr_scene_loaded = true;

        let eye = glam::Vec3::new(preset.camera_distance, preset.camera_height, 0.0);
        let target = glam::Vec3::new(0.0, preset.camera_height * 0.5, 0.0);
        self.camera.set_look_at(eye, target, glam::Vec3::Y);

        Ok(())
    }

    pub(crate) fn generate_stripe_env_map(&mut self, preset: &SsrScenePreset) -> anyhow::Result<()> {
        let size = 256u32;
        let stripe_count = preset.stripe_count.max(1);
        let stripe_width = size / stripe_count;
        
        let mut pixels = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let stripe_idx = x / stripe_width;
                let is_bright = stripe_idx % 2 == 0;
                let idx = ((y * size + x) * 4) as usize;
                
                let (r, g, b) = if is_bright {
                    let bright = (preset.stripe_bright_intensity * 255.0) as u8;
                    (bright, bright, bright)
                } else {
                    let dark = (preset.stripe_dark_intensity * 255.0) as u8;
                    (dark, dark, dark)
                };
                
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.ssr.env.stripe"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for face in 0..6 {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.ssr_env_texture = Some(texture);

        if let Some(ref tex) = self.ssr_env_texture {
            if let Some(ref mut gi) = self.gi {
                gi.set_ssr_env(&self.device, tex);
            }
        }

        Ok(())
    }
}
